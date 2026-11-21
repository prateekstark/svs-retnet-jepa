"""Evaluate Singing Voice Separation on MUSDB18 / Test Split (SDR in dB, SI-SDR in dB, JEPA Manifold Loss) & export WAVs."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.audio.stft import compute_sdr_db, compute_sisdr_db, save_wav
from src.data.musdb import MUSDB18Dataset, SyntheticSVSDataset
from src.models.separator import DualJEPARetNetSeparator
from src.utils import (
    build_stem_jepa,
    build_stft_processor,
    build_unet_separator,
    get_device,
    load_config,
    set_seed,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Singing Voice Separation (SDR/SI-SDR in dB).")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt-vocal", type=str, required=True)
    parser.add_argument("--ckpt-music", type=str, required=True)
    parser.add_argument("--separator-ckpt", type=str, required=True)
    parser.add_argument("--musdb-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs/samples")
    parser.add_argument("--num-wavs", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(args.device or cfg["device"])
    a_cfg = cfg["audio"]

    ckpt_v = torch.load(args.ckpt_vocal, map_location="cpu", weights_only=True)
    ckpt_m = torch.load(args.ckpt_music, map_location="cpu", weights_only=True)
    ckpt_s = torch.load(args.separator_ckpt, map_location="cpu", weights_only=True)
    retnet_mode = ckpt_s.get("retnet_mode", cfg["model"]["retnet_mode"])
    mode = ckpt_s.get("mode", "hybrid_ac")

    stft_proc = build_stft_processor(cfg)
    jepa_v = build_stem_jepa(cfg, retnet_mode=retnet_mode)
    jepa_v.load_state_dict(ckpt_v["model"])
    jepa_m = build_stem_jepa(cfg, retnet_mode=retnet_mode)
    jepa_m.load_state_dict(ckpt_m["model"])

    unet_sep = build_unet_separator(cfg, retnet_mode=retnet_mode)
    unet_sep.load_state_dict(ckpt_s["separator"])

    model = DualJEPARetNetSeparator(
        stft_proc=stft_proc,
        jepa_vocal=jepa_v,
        jepa_music=jepa_m,
        unet_separator=unet_sep,
        mode=mode,
    ).to(device).eval()

    if args.musdb_dir:
        ds_eval = MUSDB18Dataset(
            root_dir=args.musdb_dir,
            split="test",
            sample_rate=a_cfg["sample_rate"],
            hop_length=a_cfg["hop_length"],
            crop_frames=a_cfg["crop_frames"],
            random_crop=False,
        )
    else:
        ds_eval = SyntheticSVSDataset(
            num_samples=60,
            sample_rate=a_cfg["sample_rate"],
            hop_length=a_cfg["hop_length"],
            crop_frames=a_cfg["crop_frames"],
            seed=cfg["seed"] + 500,
        )

    loader = DataLoader(ds_eval, batch_size=12, shuffle=False)
    base_v_sdr, sep_v_sdr, sep_v_sisdr = 0.0, 0.0, 0.0
    base_m_sdr, sep_m_sdr, sep_m_sisdr = 0.0, 0.0, 0.0
    jepa_v_err, jepa_m_err = 0.0, 0.0
    mix_res = 0.0
    n_batches = 0

    for mix_w, vocal_w, music_w in loader:
        mix_w = mix_w.to(device).float()
        vocal_w = vocal_w.to(device).float()
        music_w = music_w.to(device).float()

        est_v, est_m, _ = model.separate_waveforms(mix_w)
        _, _, log_v_est = model.stft_proc.stft(est_v)
        _, _, log_m_est = model.stft_proc.stft(est_m)

        base_v_sdr += compute_sdr_db(mix_w, vocal_w)
        sep_v_sdr += compute_sdr_db(est_v, vocal_w)
        sep_v_sisdr += compute_sisdr_db(est_v, vocal_w)

        base_m_sdr += compute_sdr_db(mix_w, music_w)
        sep_m_sdr += compute_sdr_db(est_m, music_w)
        sep_m_sisdr += compute_sisdr_db(est_m, music_w)

        jv, _ = model.jepa_vocal.verify_separated_spectrogram(log_v_est)
        jm, _ = model.jepa_music.verify_separated_spectrogram(log_m_est)
        jepa_v_err += jv.item()
        jepa_m_err += jm.item()
        mix_res += F.l1_loss(est_v + est_m, mix_w).item()
        n_batches += 1

    n = max(1, n_batches)
    print(f"=== Singing Voice Separation Eval (mode={mode}, retnet_mode={retnet_mode}) ===", flush=True)
    print(f"Vocal SDR (dB, higher is better)  | Mixture: {base_v_sdr/n:+.2f} dB -> Separated: {sep_v_sdr/n:+.2f} dB (SI-SDR: {sep_v_sisdr/n:+.2f} dB)", flush=True)
    print(f"Music SDR (dB, higher is better)  | Mixture: {base_m_sdr/n:+.2f} dB -> Separated: {sep_m_sdr/n:+.2f} dB (SI-SDR: {sep_m_sisdr/n:+.2f} dB)", flush=True)
    print(f"Dual JEPA Manifold Error (V / M)  | P_V: {jepa_v_err/n:.4f} | P_M: {jepa_m_err/n:.4f}", flush=True)
    print(f"Exact Mixture Conservation Error  | {mix_res/n:.2e}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(args.num_wavs, len(ds_eval))):
        mw, vw, muw = ds_eval[i]
        mw_d = mw.unsqueeze(0).to(device).float()
        ev, em, _ = model.separate_waveforms(mw_d)
        save_wav(out_dir / f"song_{i}_mixture.wav", mw, a_cfg["sample_rate"])
        save_wav(out_dir / f"song_{i}_vocal_gt.wav", vw, a_cfg["sample_rate"])
        save_wav(out_dir / f"song_{i}_vocal_separated.wav", ev.squeeze(0), a_cfg["sample_rate"])
        save_wav(out_dir / f"song_{i}_music_gt.wav", muw, a_cfg["sample_rate"])
        save_wav(out_dir / f"song_{i}_music_separated.wav", em.squeeze(0), a_cfg["sample_rate"])

    print(f"Saved {min(args.num_wavs, len(ds_eval))} separated song stem sets to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
