"""Stage 1: Pretrain Domain-Specific RetNet-JEPA Priors on Isolated Singing Vocals or Music Accompaniment."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.musdb import MUSDB18Dataset, SyntheticSVSDataset
from src.utils import build_stem_jepa, build_stft_processor, get_device, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Vocal or Music RetNet-JEPA Prior.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stem", type=str, choices=["vocal", "music"], required=True)
    parser.add_argument("--musdb-dir", type=str, default=None, help="Path to MUSDB18 root (uses synthetic if omitted)")
    parser.add_argument("--retnet-mode", type=str, choices=["bidirectional", "causal"], default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Held-out song fraction for validation/early stopping")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience in epochs (0 to disable)")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum validation loss improvement to reset patience")
    parser.add_argument("--batch-size", type=int, default=None, help="Override auto-selected batch size")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on Ampere/Hopper/Turing GPUs")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(args.device or cfg["device"])
    j_cfg = cfg["jepa_pretrain"]
    a_cfg = cfg["audio"]
    epochs = args.epochs or j_cfg["epochs"]
    chunks_per_song = a_cfg.get("chunks_per_song", 12)

    if args.musdb_dir:
        dataset = MUSDB18Dataset(
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
        )
        val_dataset = MUSDB18Dataset(
            root_dir=args.musdb_dir,
            split="train",
            sample_rate=a_cfg["sample_rate"],
            hop_length=a_cfg["hop_length"],
            crop_frames=a_cfg["crop_frames"],
            random_crop=False,
            val_fraction=args.val_fraction,
            subset="val",
            augment=False,
        ) if args.val_fraction > 0.0 else None
    else:
        dataset = SyntheticSVSDataset(
            num_samples=300,
            sample_rate=a_cfg["sample_rate"],
            hop_length=a_cfg["hop_length"],
            crop_frames=a_cfg["crop_frames"],
            seed=cfg["seed"],
        )
        val_dataset = SyntheticSVSDataset(
            num_samples=48,
            sample_rate=a_cfg["sample_rate"],
            hop_length=a_cfg["hop_length"],
            crop_frames=a_cfg["crop_frames"],
            seed=cfg["seed"] + 9999,
        ) if args.val_fraction > 0.0 else None

    cc_major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    vram_gb = (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) if device.type == "cuda" else 0.0
    use_amp = cc_major >= 7
    if args.batch_size is not None:
        bs = args.batch_size
    elif vram_gb >= 20.0:
        bs = j_cfg.get("batch_size_24g", 48)
    elif cc_major >= 7:
        bs = 32
    else:
        bs = j_cfg["batch_size"]
    nw = 2 if device.type == "cuda" else 0
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=nw, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=(device.type == "cuda")) if val_dataset is not None else None

    stft_proc = build_stft_processor(cfg).to(device)
    model = build_stem_jepa(cfg, retnet_mode=args.retnet_mode).to(device)
    if args.compile and device.type == "cuda" and cc_major >= 7:
        model = torch.compile(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=j_cfg["lr"], weight_decay=j_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, args.patience // 3) if args.patience > 0 else 10, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = out_path.with_name(out_path.stem + "_last.pt")
    start_epoch = 1
    best_val_loss = float("inf")
    no_improve = 0

    resume_src = last_path if last_path.exists() else (out_path if out_path.exists() else None)
    if resume_src is not None:
        try:
            prev = torch.load(resume_src, map_location=device, weights_only=True)
            model.load_state_dict(prev["model"], strict=True)
            if "optimizer" in prev:
                optimizer.load_state_dict(prev["optimizer"])
            if "scheduler" in prev:
                scheduler.load_state_dict(prev["scheduler"])
            start_epoch = prev.get("epoch", 0) + 1
            best_val_loss = prev.get("best_val_loss", float("inf"))
            no_improve = prev.get("no_improve", 0)
            print(f"Resuming {resume_src} from epoch {start_epoch}/{epochs} (best_val={best_val_loss:.4f})", flush=True)
        except Exception:
            print(f"Starting fresh for {out_path}", flush=True)

    print(
        f"[JEPA-{args.stem.upper()}] mode={model.encoder.mode} | amp={use_amp} | bs={bs} | "
        f"patience={args.patience} | trainable_params={sum(p.numel() for p in trainable):,} | device={device}",
        flush=True,
    )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = {}
        for _, vocal_wav, music_wav in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False, disable=not sys.stdout.isatty()):
            target_wav = (vocal_wav if args.stem == "vocal" else music_wav).to(device).float()
            with torch.no_grad():
                _, _, log_mag = stft_proc.stft(target_wav)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                loss, stats = model(log_mag)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            model.update_ema(decay=j_cfg["ema_decay"])
            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v

        n_steps = max(1, len(loader))
        train_total = running.get("total", 0.0) / n_steps

        # Held-out validation pass (right stem + cross-stem discrimination margin)
        val_loss = train_total
        val_wrong_loss = 0.0
        val_margin = 0.0
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_sum, wrong_sum, val_n = 0.0, 0.0, 0
            with torch.no_grad():
                for _, vocal_wav, music_wav in val_loader:
                    right_wav = (vocal_wav if args.stem == "vocal" else music_wav).to(device).float()
                    wrong_wav = (music_wav if args.stem == "vocal" else vocal_wav).to(device).float()
                    _, _, right_mag = stft_proc.stft(right_wav)
                    _, _, wrong_mag = stft_proc.stft(wrong_wav)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                        v_loss, _ = model(right_mag)
                        w_loss, _ = model(wrong_mag)
                    val_sum += float(v_loss.item())
                    wrong_sum += float(w_loss.item())
                    val_n += 1
            val_loss = val_sum / max(1, val_n)
            val_wrong_loss = wrong_sum / max(1, val_n)
            val_margin = val_wrong_loss - val_loss

        scheduler.step(val_loss)
        improved = val_loss < (best_val_loss - args.min_delta)
        if improved:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "retnet_mode": model.encoder.mode,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "val_wrong_loss": val_wrong_loss,
                    "val_margin": val_margin,
                },
                out_path,
            )
        else:
            no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs or improved:
            summary = " | ".join(f"{k}={v/n_steps:.4f}" for k, v in running.items())
            lr_now = optimizer.param_groups[0]["lr"]
            mark = " *BEST*" if improved else ""
            print(
                f"Epoch {epoch:03d}/{epochs} | {summary} | val_loss={val_loss:.4f} | "
                f"val_wrong={val_wrong_loss:.4f} | margin={val_margin:+.4f} | lr={lr_now:.2e} | no_imp={no_improve}{mark}",
                flush=True,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "retnet_mode": model.encoder.mode,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "no_improve": no_improve,
                },
                last_path,
            )

        if args.patience > 0 and no_improve >= args.patience:
            print(
                f"Early stopping triggered at epoch {epoch} (no val_loss improvement > {args.min_delta} for {args.patience} epochs). Best val_loss={best_val_loss:.4f}",
                flush=True,
            )
            break

    print(f"Best {args.stem} JEPA checkpoint saved to {out_path} (best_val_loss={best_val_loss:.4f})", flush=True)


if __name__ == "__main__":
    main()
