"""Stage 2: Train Dual-JEPA Split-Latent RetNet Singing Voice Separator (Unpaired, Paired, or Hybrid AC)."""

import argparse
import sys
from itertools import cycle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dual-JEPA Split-Latent RetNet Singing Voice Separator.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt-vocal", type=str, required=True)
    parser.add_argument("--ckpt-music", type=str, required=True)
    parser.add_argument("--musdb-dir", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["unpaired", "paired", "hybrid_ac"], default=None)
    parser.add_argument("--paired-fraction", type=float, default=None)
    parser.add_argument("--lambda-jepa-verify", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(args.device or cfg["device"])
    s_cfg = cfg["separator"]
    a_cfg = cfg["audio"]

    mode = args.mode or s_cfg["mode"]
    epochs = args.epochs or s_cfg["epochs"]
    paired_frac = args.paired_fraction if args.paired_fraction is not None else s_cfg["paired_fraction"]
    l_jepa_ver = args.lambda_jepa_verify if args.lambda_jepa_verify is not None else s_cfg["lambda_jepa_verify"]

    ckpt_v = torch.load(args.ckpt_vocal, map_location="cpu", weights_only=True)
    ckpt_m = torch.load(args.ckpt_music, map_location="cpu", weights_only=True)
    retnet_mode = ckpt_v.get("retnet_mode", cfg["model"]["retnet_mode"])

    stft_proc = build_stft_processor(cfg)
    jepa_v = build_stem_jepa(cfg, retnet_mode=retnet_mode)
    jepa_v.load_state_dict(ckpt_v["model"])

    jepa_m = build_stem_jepa(cfg, retnet_mode=retnet_mode)
    jepa_m.load_state_dict(ckpt_m["model"])

    unet_sep = build_unet_separator(cfg, retnet_mode=retnet_mode)
    model = DualJEPARetNetSeparator(
        stft_proc=stft_proc,
        jepa_vocal=jepa_v,
        jepa_music=jepa_m,
        unet_separator=unet_sep,
        mode=mode,
        lambda_stft=s_cfg["lambda_stft"],
        lambda_sdr=s_cfg["lambda_sdr"],
        lambda_jepa_verify=l_jepa_ver,
        lambda_coral=s_cfg["lambda_coral"],
        lambda_remix_cycle=s_cfg["lambda_remix_cycle"],
        lambda_orth=s_cfg["lambda_orth"],
    ).to(device)

    cc_major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    use_amp = cc_major >= 7
    bs = 8 if cc_major >= 7 else 4
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if args.musdb_dir:
        full_ds = MUSDB18Dataset(
            root_dir=args.musdb_dir,
            split="train",
            sample_rate=a_cfg["sample_rate"],
            hop_length=a_cfg["hop_length"],
            crop_frames=a_cfg["crop_frames"],
            random_crop=True,
        )
    else:
        full_ds = SyntheticSVSDataset(
            num_samples=300,
            sample_rate=a_cfg["sample_rate"],
            hop_length=a_cfg["hop_length"],
            crop_frames=a_cfg["crop_frames"],
            seed=cfg["seed"],
        )

    n_total = len(full_ds)
    n_paired = max(bs, int(round(n_total * paired_frac)))
    paired_ds = Subset(full_ds, list(range(n_paired)))

    loader_unpaired = DataLoader(full_ds, batch_size=bs, shuffle=True, drop_last=True)
    loader_paired = DataLoader(paired_ds, batch_size=bs, shuffle=True, drop_last=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=s_cfg["lr"], weight_decay=0.02)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[Separator mode={mode}] retnet_mode={retnet_mode} | paired_frac={paired_frac} ({n_paired}/{n_total}) | "
        f"lambda_jepa_verify={l_jepa_ver} | amp={use_amp} | bs={bs} | trainable_params={sum(p.numel() for p in trainable):,}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        model.jepa_vocal.eval()
        model.jepa_music.eval()
        running = {}
        pair_iter = cycle(loader_paired) if mode in ("paired", "hybrid_ac") else None

        for mix_u, vocal_u, music_u in tqdm(
            loader_unpaired, desc=f"Sep {epoch}/{epochs}", leave=False, disable=not sys.stdout.isatty()
        ):
            mix_u = mix_u.to(device).float()
            vocal_u = vocal_u.to(device).float()
            # Roll music_u by 2 across batch so (vocal_u, music_u_unpaired) come from completely different songs!
            music_u_unpaired = music_u.roll(shifts=2, dims=0).to(device).float()

            mix_p, vocal_p, music_p = None, None, None
            if pair_iter is not None:
                if paired_frac >= 1.0:
                    mix_p, vocal_p, music_p = mix_u, vocal_u, music_u.to(device).float()
                else:
                    mp, vp, mup = next(pair_iter)
                    mix_p, vocal_p, music_p = mp.to(device).float(), vp.to(device).float(), mup.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                loss, stats = model.forward_step(
                    unpaired_vocal_wav=vocal_u,
                    unpaired_music_wav=music_u_unpaired,
                    unseparated_song_wav=mix_u,
                    paired_mix_wav=mix_p,
                    paired_vocal_wav=vocal_p,
                    paired_music_wav=music_p,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v

        n_steps = max(1, len(loader_unpaired))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            summary = " | ".join(f"{k}={v/n_steps:.4f}" for k, v in running.items())
            print(f"Epoch {epoch:03d}/{epochs} | {summary}", flush=True)
            torch.save(
                {"separator": model.separator.state_dict(), "mode": mode, "retnet_mode": retnet_mode, "epoch": epoch},
                out_path,
            )

    print(f"Saved separator checkpoint to {out_path}", flush=True)


if __name__ == "__main__":
    main()
