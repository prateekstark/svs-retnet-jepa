"""MUSDB18 / Multi-Track Singing Voice Separation Dataset Loader (Paired, Few-Shot, & Unpaired Pools)."""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from src.audio.stft import load_wav


class MUSDB18Dataset(Dataset):
    """Loads MUSDB18-HQ song directories containing vocals.wav and accompaniment/drums+bass+other."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        sample_rate: int = 16000,
        hop_length: int = 256,
        crop_frames: int = 128,
        chunks_per_song: int = 12,
        random_crop: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        split_dir = self.root_dir / split if (self.root_dir / split).exists() else self.root_dir
        self.sample_rate = sample_rate
        self.target_samples = (crop_frames - 1) * hop_length
        self.chunks_per_song = chunks_per_song if split == "train" else 4
        self.random_crop = random_crop

        self.song_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir() and (d / "vocals.wav").exists()])
        if not self.song_dirs:
            raise FileNotFoundError(f"No MUSDB18 song folders with vocals.wav found in {split_dir}")

        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.song_dirs) * self.chunks_per_song

    def _load_song_stems(self, song_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if song_idx not in self._cache:
            sdir = self.song_dirs[song_idx]
            vocal = load_wav(sdir / "vocals.wav", target_sr=self.sample_rate)
            if (sdir / "accompaniment.wav").exists():
                music = load_wav(sdir / "accompaniment.wav", target_sr=self.sample_rate)
            elif (sdir / "mixture.wav").exists():
                mix = load_wav(sdir / "mixture.wav", target_sr=self.sample_rate)
                min_l = min(len(mix), len(vocal))
                vocal = vocal[:min_l]
                music = mix[:min_l] - vocal
            else:
                music = torch.zeros_like(vocal)
                for stem in ("drums.wav", "bass.wav", "other.wav"):
                    if (sdir / stem).exists():
                        st = load_wav(sdir / stem, target_sr=self.sample_rate)
                        min_l = min(len(music), len(st))
                        music[:min_l] = music[:min_l] + st[:min_l]
            min_len = min(len(vocal), len(music))
            self._cache[song_idx] = (vocal[:min_len], music[:min_len])
        return self._cache[song_idx]

    def _crop_chunk(self, wav: torch.Tensor, start: int) -> torch.Tensor:
        if len(wav) < self.target_samples:
            return torch.nn.functional.pad(wav, (0, self.target_samples - len(wav)))
        return wav[start : start + self.target_samples]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mix_wav, vocal_wav, music_wav) of shape (target_samples,)."""
        song_idx = idx // self.chunks_per_song
        chunk_idx = idx % self.chunks_per_song
        vocal, music = self._load_song_stems(song_idx)

        max_start = max(1, len(vocal) - self.target_samples)
        if self.random_crop:
            # Pick a region with non-trivial vocal or music energy
            start = int(torch.randint(0, max_start, (1,)).item())
            for _ in range(4):
                v_c = vocal[start : start + self.target_samples]
                if v_c.pow(2).mean().item() > 1e-4:
                    break
                start = int(torch.randint(0, max_start, (1,)).item())
        else:
            start = min(max_start - 1, (max_start // max(1, self.chunks_per_song)) * chunk_idx)

        v_crop = self._crop_chunk(vocal, start)
        m_crop = self._crop_chunk(music, start)
        return v_crop + m_crop, v_crop, m_crop


class SyntheticSVSDataset(Dataset):
    """Synthesizes vibrato harmonic singing vocals + percussive/chordal music accompaniment."""

    def __init__(
        self,
        num_samples: int = 300,
        sample_rate: int = 16000,
        hop_length: int = 256,
        crop_frames: int = 128,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.target_samples = (crop_frames - 1) * hop_length
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed + idx)
        t = torch.linspace(0.0, self.target_samples / self.sample_rate, self.target_samples)

        # 1. Singing Voice: F0 in [180, 340] Hz with 5.5 Hz vibrato + harmonic overtones + syllable envelope
        f0 = float(torch.empty(1).uniform_(180.0, 340.0, generator=g).item())
        vib_rate = float(torch.empty(1).uniform_(4.8, 6.5, generator=g).item())
        phase = torch.cumsum(2.0 * 3.14159 * (f0 + 12.0 * torch.sin(2.0 * 3.14159 * vib_rate * t)) / self.sample_rate, dim=0)
        vocal = torch.zeros_like(t)
        for h in range(1, 7):
            vocal = vocal + (1.0 / (h ** 1.1)) * torch.sin(h * phase)
        syllable_env = 0.5 + 0.5 * torch.sin(2.0 * 3.14159 * 2.5 * t + float(torch.rand(1, generator=g).item()) * 6.0)
        vocal = vocal * syllable_env
        vocal = 0.35 * vocal / vocal.abs().max().clamp(min=1e-5)

        # 2. Music Accompaniment: Low bass [65-110 Hz] + high chord + periodic percussive drum bursts
        f_bass = float(torch.empty(1).uniform_(65.0, 110.0, generator=g).item())
        bass = 0.5 * torch.sin(2.0 * 3.14159 * f_bass * t) + 0.25 * torch.sin(4.0 * 3.14159 * f_bass * t)
        beat_env = torch.exp(-18.0 * ((t * 4.0) % 1.0))
        noise = torch.randn(self.target_samples, generator=g)
        drums = beat_env * noise * 0.45
        music = bass + drums
        music = 0.35 * music / music.abs().max().clamp(min=1e-5)

        mix = vocal + music
        return mix, vocal, music
