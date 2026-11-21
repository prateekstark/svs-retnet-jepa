"""Complex STFT Processor, Ratio-Mask iSTFT Synthesis, Multi-Resolution STFT Loss, & MUSDB18 SDR Metrics."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile


def load_wav(path: str | Path, target_sr: int = 16000) -> torch.Tensor:
    sr, audio = wavfile.read(str(path))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    wav = torch.from_numpy(audio)
    if sr != target_sr:
        new_len = int(round(wav.shape[0] * target_sr / sr))
        wav = F.interpolate(
            wav.view(1, 1, -1), size=new_len, mode="linear", align_corners=False
        ).view(-1)
    return wav


def save_wav(path: str | Path, wav: torch.Tensor, sample_rate: int = 16000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = wav.detach().cpu().view(-1).numpy()
    peak = max(float(np.abs(audio).max()), 1e-5)
    if peak > 1.0:
        audio = audio / peak * 0.98
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, pcm16)


def compute_sdr_db(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-7) -> float:
    """Official MUSDB18/MDX global Signal-to-Distortion Ratio (SDR in dB, higher is better)."""
    # est, ref: (B, T_samples) or (T_samples,)
    if est.dim() == 1:
        est = est.unsqueeze(0)
        ref = ref.unsqueeze(0)
    min_len = min(est.shape[-1], ref.shape[-1])
    est = est[..., :min_len]
    ref = ref[..., :min_len]

    num = torch.sum(ref.pow(2), dim=-1) + eps
    den = torch.sum((ref - est).pow(2), dim=-1) + eps
    sdr = 10.0 * torch.log10(num / den)
    return float(sdr.mean().item())


def compute_sisdr_db(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-7) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR in dB, higher is better)."""
    if est.dim() == 1:
        est = est.unsqueeze(0)
        ref = ref.unsqueeze(0)
    min_len = min(est.shape[-1], ref.shape[-1])
    est = est[..., :min_len] - est[..., :min_len].mean(dim=-1, keepdim=True)
    ref = ref[..., :min_len] - ref[..., :min_len].mean(dim=-1, keepdim=True)

    dot = torch.sum(est * ref, dim=-1, keepdim=True)
    ref_energy = torch.sum(ref.pow(2), dim=-1, keepdim=True) + eps
    proj = (dot / ref_energy) * ref
    noise = est - proj
    sisdr = 10.0 * torch.log10((torch.sum(proj.pow(2), dim=-1) + eps) / (torch.sum(noise.pow(2), dim=-1) + eps))
    return float(sisdr.mean().item())


def sdr_waveform_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable waveform L1 + negative SNR/SDR loss."""
    min_len = min(est.shape[-1], ref.shape[-1])
    est = est[..., :min_len]
    ref = ref[..., :min_len]
    l1 = F.l1_loss(est, ref)
    num = torch.sum(ref.pow(2), dim=-1) + eps
    den = torch.sum((ref - est).pow(2), dim=-1) + eps
    neg_sdr = -10.0 * torch.log10(num / den).mean() * 0.05
    return l1 + neg_sdr


class STFTProcessor(nn.Module):
    """Handles forward Complex STFT, Log-Magnitude normalization, and Exact Phase Ratio-Mask iSTFT."""

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_freq_bins: int = 512,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freq_bins = n_freq_bins
        self.register_buffer("window", torch.hann_window(n_fft))

    def stft(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """wav: (B, L_samples) -> (complex_stft: (B, 513, T), mag: (B, 1, 512, T), log_mag: (B, 1, 512, T))."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        spec_c = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        mag_full = spec_c.abs()  # (B, 513, T)
        mag = mag_full[:, : self.n_freq_bins, :].unsqueeze(1)  # (B, 1, 512, T)
        log_mag = torch.log1p(mag)
        return spec_c, mag, log_mag

    def apply_mask_and_istft(
        self,
        mix_complex_stft: torch.Tensor,  # (B, 513, T)
        mask_vocal: torch.Tensor,        # (B, 1, 512, T) in [0, 1]
        length: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply soft ratio mask M_V and (1 - M_V) to mixture complex STFT and invert to waveforms.
        Guarantees wav_vocal + wav_music == wav_mix exactly!
        """
        m_v = mask_vocal.squeeze(1)  # (B, 512, T)
        # Pad 513th Nyquist frequency bin by replicating bin 512
        m_v_full = F.pad(m_v, (0, 0, 0, 1), mode="replicate")  # (B, 513, T)
        m_m_full = 1.0 - m_v_full

        stft_v = mix_complex_stft * m_v_full
        stft_m = mix_complex_stft * m_m_full

        wav_v = torch.istft(
            stft_v,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            length=length,
        )
        wav_m = torch.istft(
            stft_m,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            length=length,
        )
        return wav_v, wav_m
