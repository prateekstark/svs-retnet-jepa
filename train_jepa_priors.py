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
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(args.device or cfg["device"])
    j_cfg = cfg["jepa_pretrain"]
    a_cfg = cfg["audio"]
    epochs = args.epochs or j_cfg["epochs"]

    if args.musdb_dir:
        dataset = MUSDB18Dataset(
            root_dir=args.musdb_dir,
            split="train",
            sample_rate=a_cfg["sample_rate"],
            hop_length=a_cfg["hop_length"],
            crop_frames=a_cfg["crop_frames"],
            random_crop=True,
        )
    else:
        dataset = SyntheticSVSDataset(
            num_samples=300,
            sample_rate=a_cfg["sample_rate"],
            hop_length=a_cfg["hop_length"],
            crop_frames=a_cfg["crop_frames"],
            seed=cfg["seed"],
        )

    cc_major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    use_amp = cc_major >= 7
    bs = 32 if cc_major >= 7 else j_cfg["batch_size"]
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

    stft_proc = build_stft_processor(cfg).to(device)
    model = build_stem_jepa(cfg, retnet_mode=args.retnet_mode).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=j_cfg["lr"], weight_decay=j_cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    if out_path.exists():
        try:
            prev = torch.load(out_path, map_location=device, weights_only=True)
            model.load_state_dict(prev["model"], strict=True)
            start_epoch = prev.get("epoch", epochs) + 1
            print(f"Resuming {out_path} from epoch {start_epoch}/{epochs}", flush=True)
        except Exception:
            print(f"Starting fresh for {out_path}", flush=True)

    print(
        f"[JEPA-{args.stem.upper()}] mode={model.encoder.mode} | amp={use_amp} | bs={bs} | "
        f"trainable_params={sum(p.numel() for p in trainable):,} | device={device}",
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
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            summary = " | ".join(f"{k}={v/n_steps:.4f}" for k, v in running.items())
            print(f"Epoch {epoch:03d}/{epochs} | {summary}", flush=True)
            torch.save({"model": model.state_dict(), "retnet_mode": model.encoder.mode, "epoch": epoch}, out_path)

    print(f"Saved {args.stem} JEPA checkpoint to {out_path}", flush=True)


if __name__ == "__main__":
    main()
