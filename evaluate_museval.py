"""Official Full-Song 44.1 kHz Stereo MUSDB18-HQ Evaluation using museval (BSS-Eval v4).

Runs overlap-add separation on every full-length test song in MUSDB18-HQ/test,
upsamples estimates back to 44.1 kHz stereo to match unmodified ground-truth stems,
and reports the official MUSDB18 metric: Median across 50 songs of per-song median 1s-frame SDR.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import resample_poly

from src.utils import build_stft_processor, build_unet_separator, get_device, load_config


def load_stereo_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Load WAV as float32 stereo array of shape (N, 2) in [-1, 1]."""
    sr, audio = wavfile.read(str(path))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    return audio, sr


@torch.no_grad()
def separate_mono_overlap_add(
    wav_np: np.ndarray,
    stft_proc: torch.nn.Module,
    separator: torch.nn.Module,
    chunk_samples: int,
    device: torch.device,
    use_amp: bool,
    batch_chunks: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlap-add separation of a full-length mono signal (N,) at model sample rate."""
    wav = torch.from_numpy(wav_np.astype(np.float32)).to(device)
    hop_samples = chunk_samples // 2
    total_len = wav.shape[0]
    pad_len = (
        max(0, chunk_samples - total_len)
        if total_len < chunk_samples
        else (hop_samples - (total_len - chunk_samples) % hop_samples) % hop_samples
    )
    wav_padded = F.pad(wav, (0, pad_len))

    starts = list(range(0, wav_padded.shape[0] - chunk_samples + 1, hop_samples))
    vocal_acc = torch.zeros_like(wav_padded)
    music_acc = torch.zeros_like(wav_padded)
    weight_acc = torch.zeros_like(wav_padded)
    fade = torch.hann_window(chunk_samples, device=device).clamp(min=1e-3)

    for i in range(0, len(starts), batch_chunks):
        batch_starts = starts[i : i + batch_chunks]
        chunks = torch.stack([wav_padded[s : s + chunk_samples] for s in batch_starts], dim=0)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            spec_c, _, log_mag = stft_proc.stft(chunks)
            mask_v, _, _ = separator(log_mag)
            wv, wm = stft_proc.apply_mask_and_istft(spec_c, mask_v, length=chunk_samples)
        wv = wv.float()
        wm = wm.float()
        for bi, s in enumerate(batch_starts):
            vocal_acc[s : s + chunk_samples] += wv[bi] * fade
            music_acc[s : s + chunk_samples] += wm[bi] * fade
            weight_acc[s : s + chunk_samples] += fade

    vocal_out = (vocal_acc / weight_acc.clamp(min=1e-5))[:total_len].cpu().numpy()
    music_out = (music_acc / weight_acc.clamp(min=1e-5))[:total_len].cpu().numpy()
    return vocal_out, music_out


def separate_stereo_44k(
    mix_44k: np.ndarray,
    orig_sr: int,
    model_sr: int,
    stft_proc: torch.nn.Module,
    separator: torch.nn.Module,
    chunk_samples: int,
    device: torch.device,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate stereo 44.1 kHz song channel-by-channel at model_sr, then upsample back to 44.1 kHz."""
    n_samples = mix_44k.shape[0]
    g = int(np.gcd(orig_sr, model_sr))
    up_down_in = (model_sr // g, orig_sr // g)
    up_down_out = (orig_sr // g, model_sr // g)

    est_v_ch, est_m_ch = [], []
    for ch in range(2):
        sig_in = resample_poly(mix_44k[:, ch], up_down_in[0], up_down_in[1]).astype(np.float32) if orig_sr != model_sr else mix_44k[:, ch]
        v_m, m_m = separate_mono_overlap_add(sig_in, stft_proc, separator, chunk_samples, device, use_amp)
        if orig_sr != model_sr:
            v_44 = resample_poly(v_m, up_down_out[0], up_down_out[1]).astype(np.float32)
            m_44 = resample_poly(m_m, up_down_out[0], up_down_out[1]).astype(np.float32)
        else:
            v_44, m_44 = v_m, m_m
        # Match exact length & preserve >11 kHz stereo air/cymbals in accompaniment (exact 44.1 kHz conservation)
        v_44 = v_44[:n_samples] if len(v_44) >= n_samples else np.pad(v_44, (0, n_samples - len(v_44)))
        m_44 = mix_44k[:, ch] - v_44
        est_v_ch.append(v_44)
        est_m_ch.append(m_44)

    return np.stack(est_v_ch, axis=-1), np.stack(est_m_ch, axis=-1)


def main() -> None:
    import museval

    parser = argparse.ArgumentParser(description="Official Full-Song 44.1 kHz Stereo museval Benchmark on MUSDB18-HQ.")
    parser.add_argument("--config", type=str, default="configs/musdb18_hq_v2.yaml")
    parser.add_argument("--separator-ckpt", type=str, required=True)
    parser.add_argument("--musdb-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--max-songs", type=int, default=0, help="0 = all 50 test songs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device or cfg["device"])
    a_cfg = cfg["audio"]
    cc_major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    use_amp = cc_major >= 7

    ckpt_s = torch.load(args.separator_ckpt, map_location="cpu", weights_only=True)
    retnet_mode = ckpt_s.get("retnet_mode", cfg["model"]["retnet_mode"])

    stft_proc = build_stft_processor(cfg).to(device)
    separator = build_unet_separator(cfg, retnet_mode=retnet_mode).to(device)
    separator.load_state_dict(ckpt_s["separator"])
    separator.eval()

    chunk_samples = (a_cfg["crop_frames"] - 1) * a_cfg["hop_length"]
    test_dir = Path(args.musdb_dir) / "test"
    if not test_dir.exists():
        test_dir = Path(args.musdb_dir)
    song_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir() and (d / "vocals.wav").exists()])
    if args.max_songs > 0:
        song_dirs = song_dirs[: args.max_songs]

    print(
        f"[museval 44.1kHz Stereo] ckpt={args.separator_ckpt} | songs={len(song_dirs)} | "
        f"model_sr={a_cfg['sample_rate']} | device={device} | amp={use_amp}",
        flush=True,
    )

    vocal_song_sdrs, music_song_sdrs = [], []
    vocal_song_sirs, vocal_song_sars = [], []
    per_song = {}

    for idx, sdir in enumerate(song_dirs, 1):
        mix_44, sr = load_stereo_wav(sdir / "mixture.wav")
        voc_44, _ = load_stereo_wav(sdir / "vocals.wav")
        min_l = min(len(mix_44), len(voc_44))
        mix_44 = mix_44[:min_l]
        voc_44 = voc_44[:min_l]
        mus_44 = mix_44 - voc_44

        est_v, est_m = separate_stereo_44k(
            mix_44, sr, a_cfg["sample_rate"], stft_proc, separator, chunk_samples, device, use_amp
        )

        # museval.evaluate expects shape (nsrc, nsampl, nchan)
        refs = np.stack([voc_44, mus_44], axis=0)
        ests = np.stack([est_v, est_m], axis=0)
        sdr, isr, sir, sar = museval.evaluate(refs, ests, win=sr, hop=sr)

        v_med = float(np.nanmedian(sdr[0]))
        m_med = float(np.nanmedian(sdr[1]))
        v_sir = float(np.nanmedian(sir[0]))
        v_sar = float(np.nanmedian(sar[0]))
        vocal_song_sdrs.append(v_med)
        music_song_sdrs.append(m_med)
        vocal_song_sirs.append(v_sir)
        vocal_song_sars.append(v_sar)
        per_song[sdir.name] = {"vocal_sdr": v_med, "music_sdr": m_med, "vocal_sir": v_sir, "vocal_sar": v_sar}

        if idx == 1 or idx % 10 == 0 or idx == len(song_dirs):
            print(
                f"[{idx:02d}/{len(song_dirs)}] {sdir.name[:35]:35s} | "
                f"V_SDR={v_med:+.2f} dB | M_SDR={m_med:+.2f} dB | "
                f"Running Med: V={np.nanmedian(vocal_song_sdrs):+.2f} dB, M={np.nanmedian(music_song_sdrs):+.2f} dB",
                flush=True,
            )

    res = {
        "ckpt": str(args.separator_ckpt),
        "num_songs": len(song_dirs),
        "vocal_sdr_median_of_medians": float(np.nanmedian(vocal_song_sdrs)),
        "vocal_sdr_mean_of_medians": float(np.nanmean(vocal_song_sdrs)),
        "vocal_sir_median": float(np.nanmedian(vocal_song_sirs)),
        "vocal_sar_median": float(np.nanmedian(vocal_song_sars)),
        "music_sdr_median_of_medians": float(np.nanmedian(music_song_sdrs)),
        "music_sdr_mean_of_medians": float(np.nanmean(music_song_sdrs)),
        "per_song": per_song,
    }

    print("=" * 78, flush=True)
    print(f"OFFICIAL MUSDB18-HQ 44.1kHz STEREO MUSEVAL ({len(song_dirs)} FULL SONGS)", flush=True)
    print(f"Checkpoint : {args.separator_ckpt}", flush=True)
    print(
        f"Vocals     : Median SDR = {res['vocal_sdr_median_of_medians']:+.2f} dB | "
        f"Mean SDR = {res['vocal_sdr_mean_of_medians']:+.2f} dB | "
        f"SIR = {res['vocal_sir_median']:+.2f} dB | SAR = {res['vocal_sar_median']:+.2f} dB",
        flush=True,
    )
    print(
        f"Music/Acc. : Median SDR = {res['music_sdr_median_of_medians']:+.2f} dB | "
        f"Mean SDR = {res['music_sdr_mean_of_medians']:+.2f} dB",
        flush=True,
    )
    print("=" * 78, flush=True)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(res, indent=2))
        print(f"Saved detailed per-song museval JSON to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
