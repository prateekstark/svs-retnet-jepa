"""Separate any full-length WAV song into clean Singing Voice (vocals.wav) and Music Accompaniment (music.wav)."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.audio.stft import load_wav, save_wav
from src.utils import build_stft_processor, build_unet_separator, get_device, load_config


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Separate a full-length WAV song into vocals and music.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--separator-ckpt", type=str, required=True)
    parser.add_argument("--input-wav", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/separated_song")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device or cfg["device"])
    a_cfg = cfg["audio"]

    ckpt_s = torch.load(args.separator_ckpt, map_location="cpu", weights_only=True)
    retnet_mode = ckpt_s.get("retnet_mode", cfg["model"]["retnet_mode"])

    stft_proc = build_stft_processor(cfg).to(device)
    separator = build_unet_separator(cfg, retnet_mode=retnet_mode).to(device)
    separator.load_state_dict(ckpt_s["separator"])
    separator.eval()

    wav = load_wav(args.input_wav, target_sr=a_cfg["sample_rate"]).to(device)
    chunk_samples = (a_cfg["crop_frames"] - 1) * a_cfg["hop_length"]
    hop_samples = chunk_samples // 2

    total_len = wav.shape[0]
    pad_len = max(0, chunk_samples - total_len) if total_len < chunk_samples else (hop_samples - (total_len - chunk_samples) % hop_samples) % hop_samples
    wav_padded = F.pad(wav, (0, pad_len))

    vocal_acc = torch.zeros_like(wav_padded)
    music_acc = torch.zeros_like(wav_padded)
    weight_acc = torch.zeros_like(wav_padded)
    fade = torch.hann_window(chunk_samples, device=device).clamp(min=1e-3)

    for start in range(0, wav_padded.shape[0] - chunk_samples + 1, hop_samples):
        chunk = wav_padded[start : start + chunk_samples].unsqueeze(0)
        spec_c, _, log_mag = stft_proc.stft(chunk)
        mask_v, _, _ = separator(log_mag)
        wv, wm = stft_proc.apply_mask_and_istft(spec_c, mask_v, length=chunk_samples)
        vocal_acc[start : start + chunk_samples] += wv.squeeze(0) * fade
        music_acc[start : start + chunk_samples] += wm.squeeze(0) * fade
        weight_acc[start : start + chunk_samples] += fade

    vocal_out = (vocal_acc / weight_acc.clamp(min=1e-5))[:total_len]
    music_out = (music_acc / weight_acc.clamp(min=1e-5))[:total_len]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input_wav).stem
    save_wav(out_dir / f"{stem}_vocals.wav", vocal_out, a_cfg["sample_rate"])
    save_wav(out_dir / f"{stem}_music.wav", music_out, a_cfg["sample_rate"])
    print(f"Separated {args.input_wav} -> {out_dir}/{stem}_vocals.wav & {out_dir}/{stem}_music.wav")


if __name__ == "__main__":
    main()
