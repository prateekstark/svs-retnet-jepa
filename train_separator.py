"""Stage 2: Train Dual-JEPA Split-Latent RetNet Singing Voice Separator (Unpaired, Paired, or Hybrid AC)."""

import argparse
import sys
from itertools import cycle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.audio.stft import compute_sdr_db
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
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Held-out song fraction for validation/early stopping")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience in epochs (0 to disable)")
    parser.add_argument("--min-epochs", type=int, default=15, help="Minimum warmup epochs before early stopping can trigger")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum validation metric improvement to reset patience")
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        choices=["auto", "val_loss", "val_sdr"],
        default="auto",
        help="Metric to monitor for early stopping ('auto': val_loss for unpaired, val_sdr for paired/hybrid_ac)",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override auto-selected batch size")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on Ampere/Hopper/Turing GPUs")
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
    es_metric = ("val_loss" if mode == "unpaired" else "val_sdr") if args.early_stop_metric == "auto" else args.early_stop_metric
    higher_is_better = es_metric == "val_sdr"

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
        lambda_mrstft=s_cfg.get("lambda_mrstft", 0.0),
    ).to(device)
    if args.compile and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7:
        model.separator = torch.compile(model.separator, dynamic=True)

    cc_major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    vram_gb = (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) if device.type == "cuda" else 0.0
    use_amp = cc_major >= 7
    if args.batch_size is not None:
        bs = args.batch_size
    elif vram_gb >= 35.0:
        bs = s_cfg.get("batch_size_40g", 20)
    elif vram_gb >= 28.0:
        bs = s_cfg.get("batch_size_32g", 16)
    elif vram_gb >= 20.0:
        bs = s_cfg.get("batch_size_24g", 12)
    elif cc_major >= 7:
        bs = 8
    else:
        bs = 4
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    chunks_per_song = a_cfg.get("chunks_per_song", 12)

    if args.musdb_dir:
        full_ds = MUSDB18Dataset(
            root_dir=args.musdb_dir,
            split="train",
            sample_rate=a_cfg["sample_rate"],
            hop_length=a_cfg["hop_length"],
            crop_frames=a_cfg["crop_frames"],
            chunks_per_song=chunks_per_song,
            random_crop=True,
            val_fraction=args.val_fraction,
            subset="train",
            augment=True,
            preload=True,
        )
        val_ds = MUSDB18Dataset(
            root_dir=args.musdb_dir,
            split="train",
            sample_rate=a_cfg["sample_rate"],
            hop_length=a_cfg["hop_length"],
            crop_frames=a_cfg["crop_frames"],
            random_crop=False,
            val_fraction=args.val_fraction,
            subset="val",
            augment=False,
            preload=True,
        ) if args.val_fraction > 0.0 else None
    else:
        full_ds = SyntheticSVSDataset(
            num_samples=300,
            sample_rate=a_cfg["sample_rate"],
            hop_length=a_cfg["hop_length"],
            crop_frames=a_cfg["crop_frames"],
            seed=cfg["seed"],
        )
        val_ds = SyntheticSVSDataset(
            num_samples=32,
            sample_rate=a_cfg["sample_rate"],
            hop_length=a_cfg["hop_length"],
            crop_frames=a_cfg["crop_frames"],
            seed=cfg["seed"] + 9999,
        ) if args.val_fraction > 0.0 else None

    n_total = len(full_ds)
    n_paired = max(bs, int(round(n_total * paired_frac)))
    paired_ds = Subset(full_ds, list(range(n_paired)))

    pin = device.type == "cuda"
    loader_unpaired = DataLoader(full_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0, pin_memory=pin)
    loader_paired = DataLoader(paired_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=pin) if val_ds is not None else None

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=s_cfg["lr"], weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if higher_is_better else "min",
        factor=0.5,
        patience=max(3, args.patience // 3) if args.patience > 0 else 10,
        min_lr=1e-6,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = out_path.with_name(out_path.stem + "_last.pt")

    start_epoch = 1
    best_metric = float("-inf") if higher_is_better else float("inf")
    no_improve = 0
    resume_src = last_path if last_path.exists() else (out_path if out_path.exists() else None)
    if resume_src is not None:
        try:
            prev = torch.load(resume_src, map_location=device, weights_only=True)
            getattr(model.separator, "_orig_mod", model.separator).load_state_dict(prev["separator"], strict=True)
            if "optimizer" in prev:
                optimizer.load_state_dict(prev["optimizer"])
            if "scheduler" in prev:
                scheduler.load_state_dict(prev["scheduler"])
            start_epoch = prev.get("epoch", 0) + 1
            best_metric = prev.get("best_metric", best_metric)
            no_improve = prev.get("no_improve", 0)
            print(
                f"Resuming {resume_src} from epoch {start_epoch}/{epochs} (best {es_metric}={best_metric:.4f})",
                flush=True,
            )
        except Exception:
            print(f"Starting fresh for {out_path}", flush=True)

    print(
        f"[Separator mode={mode}] retnet_mode={retnet_mode} | paired_frac={paired_frac} ({n_paired}/{n_total}) | "
        f"lambda_jepa_verify={l_jepa_ver} | amp={use_amp} | bs={bs} | es={es_metric} (patience={args.patience}) | "
        f"trainable_params={sum(p.numel() for p in trainable):,}",
        flush=True,
    )

    for epoch in range(start_epoch, epochs + 1):
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
        train_total = running.get("total", 0.0) / n_steps

        # Held-out validation pass (leak-free val_loss + diagnostic vocal/music SDR)
        val_loss = train_total
        val_v_sdr, val_m_sdr = 0.0, 0.0
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_loss_sum, v_sdr_sum, m_sdr_sum, val_batches = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for mix_v, vocal_v, music_v in val_loader:
                    mix_v = mix_v.to(device).float()
                    vocal_v = vocal_v.to(device).float()
                    music_v = music_v.to(device).float()
                    music_v_unp = music_v.roll(shifts=1, dims=0) if mix_v.size(0) > 1 else music_v
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                        v_l, _ = model.forward_step(
                            unpaired_vocal_wav=vocal_v,
                            unpaired_music_wav=music_v_unp,
                            unseparated_song_wav=mix_v,
                            paired_mix_wav=mix_v if mode in ("paired", "hybrid_ac") else None,
                            paired_vocal_wav=vocal_v if mode in ("paired", "hybrid_ac") else None,
                            paired_music_wav=music_v if mode in ("paired", "hybrid_ac") else None,
                        )
                        est_v, est_m, _ = model.separate_waveforms(mix_v)
                    val_loss_sum += float(v_l.item())
                    v_sdr_sum += compute_sdr_db(est_v.float(), vocal_v)
                    m_sdr_sum += compute_sdr_db(est_m.float(), music_v)
                    val_batches += 1
            val_loss = val_loss_sum / max(1, val_batches)
            val_v_sdr = v_sdr_sum / max(1, val_batches)
            val_m_sdr = m_sdr_sum / max(1, val_batches)

        val_mean_sdr = 0.5 * (val_v_sdr + val_m_sdr)
        current_metric = val_mean_sdr if higher_is_better else val_loss
        if epoch >= args.min_epochs:
            scheduler.step(current_metric)

        improved = (
            current_metric > (best_metric + args.min_delta)
            if higher_is_better
            else current_metric < (best_metric - args.min_delta)
        )
        if improved or epoch == 1:
            best_metric = current_metric if improved else best_metric
            no_improve = 0
            raw_sep = getattr(model.separator, "_orig_mod", model.separator)
            torch.save(
                {
                    "separator": raw_sep.state_dict(),
                    "mode": mode,
                    "retnet_mode": retnet_mode,
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "es_metric": es_metric,
                    "val_loss": val_loss,
                    "val_v_sdr": val_v_sdr,
                    "val_m_sdr": val_m_sdr,
                },
                out_path,
            )
        elif epoch >= args.min_epochs:
            no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs or improved:
            summary = " | ".join(f"{k}={v/n_steps:.4f}" for k, v in running.items())
            lr_now = optimizer.param_groups[0]["lr"]
            mark = " *BEST*" if improved else ""
            print(
                f"Epoch {epoch:03d}/{epochs} | {summary} | val_loss={val_loss:.4f} | "
                f"val_sdr=(V:{val_v_sdr:.2f}, M:{val_m_sdr:.2f}) dB | lr={lr_now:.2e} | no_imp={no_improve}{mark}",
                flush=True,
            )
            raw_sep = getattr(model.separator, "_orig_mod", model.separator)
            torch.save(
                {
                    "separator": raw_sep.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "mode": mode,
                    "retnet_mode": retnet_mode,
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "es_metric": es_metric,
                    "no_improve": no_improve,
                },
                last_path,
            )

        if args.patience > 0 and epoch >= args.min_epochs and no_improve >= args.patience:
            print(
                f"Early stopping triggered at epoch {epoch} (no {es_metric} improvement > {args.min_delta} for {args.patience} epochs). Best {es_metric}={best_metric:.4f}",
                flush=True,
            )
            break

    print(f"Best separator checkpoint saved to {out_path} (best {es_metric}={best_metric:.4f})", flush=True)


if __name__ == "__main__":
    main()
