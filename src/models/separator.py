"""Dual-JEPA Split-Latent RetNet + 2D U-Net Ratio-Mask Singing Voice Separator."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.audio.stft import STFTProcessor, sdr_waveform_loss
from src.models.retnet_jepa import (
    RetNetBlock,
    StemRetNetJEPA,
    build_2d_sincos_pos_embed,
)


def coral_moment_loss(z_pred: torch.Tensor, z_real: torch.Tensor) -> torch.Tensor:
    """Match mean, variance, and second-order Deep CORAL covariance between z_pred and z_real."""
    mu_p, mu_r = z_pred.mean(dim=(0, 1)), z_real.mean(dim=(0, 1))
    std_p = torch.sqrt(z_pred.var(dim=(0, 1), unbiased=False) + 1e-4)
    std_r = torch.sqrt(z_real.var(dim=(0, 1), unbiased=False) + 1e-4)
    moment = F.mse_loss(mu_p, mu_r) + F.mse_loss(std_p, std_r)

    d = z_pred.shape[-1]
    zp = z_pred.reshape(-1, d)
    zr = z_real.reshape(-1, d)
    zp = zp - zp.mean(dim=0, keepdim=True)
    zr = zr - zr.mean(dim=0, keepdim=True)
    cov_p = (zp.T @ zp) / max(1, zp.shape[0] - 1)
    cov_r = (zr.T @ zr) / max(1, zr.shape[0] - 1)
    return moment + 2.0 * F.mse_loss(cov_p, cov_r)


def stft_mag_loss(
    mag_est: torch.Tensor,
    log_mag_est: torch.Tensor,
    mag_ref: torch.Tensor,
    log_mag_ref: torch.Tensor,
) -> torch.Tensor:
    """Combined linear-magnitude L1 + log-magnitude L1 + 2D harmonic derivative sharpness loss."""
    l1_lin = F.l1_loss(mag_est, mag_ref)
    l1_log = F.l1_loss(log_mag_est, log_mag_ref)
    df_e = log_mag_est[:, :, 1:, :] - log_mag_est[:, :, :-1, :]
    df_r = log_mag_ref[:, :, 1:, :] - log_mag_ref[:, :, :-1, :]
    dt_e = log_mag_est[:, :, :, 1:] - log_mag_est[:, :, :, :-1]
    dt_r = log_mag_ref[:, :, :, 1:] - log_mag_ref[:, :, :, :-1]
    grad = F.l1_loss(df_e, df_r) + F.l1_loss(dt_e, dt_r)
    return l1_lin + l1_log + 0.3 * grad


class ConvBlock2D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: Tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SplitLatentRetNetUNet(nn.Module):
    """Hybrid 2D U-Net + Split-Latent RetNet predicting complementary STFT ratio masks (M_V, M_M)."""

    def __init__(
        self,
        n_freq_bins: int = 512,
        crop_frames: int = 128,
        patch_size: Tuple[int, int] = (32, 4),
        d_model: int = 256,
        num_layers: int = 6,
        decoder_layers: int = 4,
        num_heads: int = 8,
        ffn_mult: int = 3,
        unet_channels: int = 64,
        mode: str = "bidirectional",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pf, pt = patch_size
        self.pf, self.pt = pf, pt
        self.n_freq_patches = n_freq_bins // pf
        self.n_time_patches = crop_frames // pt
        self.num_patches = self.n_freq_patches * self.n_time_patches
        self.d_model = d_model
        c = unet_channels

        # High-resolution U-Net encoder skips for fine harmonic & transient preservation
        self.enc_skip1 = ConvBlock2D(1, c, stride=(1, 1))          # (B, C, 512, 128)
        self.enc_skip2 = ConvBlock2D(c, c * 2, stride=(2, 2))      # (B, 2C, 256, 64)

        # RetNet Patch Trunk
        self.patch_embed = nn.Conv2d(1, d_model, kernel_size=(pf, pt), stride=(pf, pt))
        self.register_buffer(
            "pos_embed", build_2d_sincos_pos_embed(self.n_freq_patches, self.n_time_patches, d_model)
        )
        trunk_layers = max(2, num_layers - 2)
        branch_layers = max(2, decoder_layers // 2)
        self.trunk = nn.ModuleList(
            [RetNetBlock(d_model, num_heads, ffn_mult, mode=mode, dropout=dropout) for _ in range(trunk_layers)]
        )

        # Split-Latent Orthogonal Vocal & Music RetNet Branches
        self.vocal_branch = nn.ModuleList(
            [RetNetBlock(d_model, num_heads, ffn_mult, mode=mode, dropout=dropout) for _ in range(branch_layers)]
        )
        self.music_branch = nn.ModuleList(
            [RetNetBlock(d_model, num_heads, ffn_mult, mode=mode, dropout=dropout) for _ in range(branch_layers)]
        )
        self.norm_v = nn.RMSNorm(d_model)
        self.norm_m = nn.RMSNorm(d_model)
        self.orth_proj_v = nn.Linear(d_model, 128, bias=False)
        self.orth_proj_m = nn.Linear(d_model, 128, bias=False)

        # Unpatchify to (256, 64) and fuse with U-Net skips
        self.unpatch_v = nn.Linear(d_model, pf * pt)
        self.unpatch_m = nn.Linear(d_model, pf * pt)

        self.dec2 = ConvBlock2D(c * 2 + 2, c * 2, stride=(1, 1))
        self.dec1 = ConvBlock2D(c * 2 + c + 2, c, stride=(1, 1))
        self.mask_head = nn.Conv2d(c, 2, kernel_size=3, padding=1)

    def _tokens_to_grid(self, z: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        b = z.shape[0]
        p = linear(z).view(b, self.n_time_patches, self.n_freq_patches, self.pf, self.pt)
        return p.permute(0, 2, 3, 1, 4).contiguous().view(
            b, 1, self.n_freq_patches * self.pf, self.n_time_patches * self.pt
        )

    def forward(
        self, log_mag_mix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """log_mag_mix: (B, 1, 512, 128) -> (mask_vocal: (B, 1, 512, 128), z_v: (B, N, D), z_m: (B, N, D))."""
        b = log_mag_mix.shape[0]
        s1 = self.enc_skip1(log_mag_mix)  # (B, C, 512, 128)
        s2 = self.enc_skip2(s1)           # (B, 2C, 256, 64)

        x = self.patch_embed(log_mag_mix)
        x = x.permute(0, 3, 2, 1).contiguous().view(b, self.num_patches, self.d_model) + self.pos_embed
        for blk in self.trunk:
            x = blk(x)

        zv, zm = x, x
        for blk_v, blk_m in zip(self.vocal_branch, self.music_branch):
            zv = blk_v(zv)
            zm = blk_m(zm)
        zv = self.norm_v(zv)
        zm = self.norm_m(zm)

        grid_v = self._tokens_to_grid(zv, self.unpatch_v)  # (B, 1, 512, 128)
        grid_m = self._tokens_to_grid(zm, self.unpatch_m)  # (B, 1, 512, 128)

        # Fuse at half-resolution (256, 64) then full-resolution (512, 128)
        gv_half = F.avg_pool2d(grid_v, kernel_size=2, stride=2)
        gm_half = F.avg_pool2d(grid_m, kernel_size=2, stride=2)
        d2 = self.dec2(torch.cat([s2, gv_half, gm_half], dim=1))
        d2_up = F.interpolate(d2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d2_up, s1, grid_v, grid_m], dim=1))

        logits = self.mask_head(d1)  # (B, 2, 512, 128)
        # Complementary softmax/sigmoid ratio mask M_V + M_M = 1
        mask_v = torch.sigmoid(logits[:, 0:1] - logits[:, 1:2] + grid_v - grid_m)
        return mask_v, zv, zm

    def orthogonality_loss(self, zv: torch.Tensor, zm: torch.Tensor) -> torch.Tensor:
        pv = self.orth_proj_v(zv.mean(dim=1))
        pm = self.orth_proj_m(zm.mean(dim=1))
        return F.cosine_similarity(pv, pm, dim=-1).abs().mean()


class DualJEPARetNetSeparator(nn.Module):
    """Complete Singing Voice Separator verified by Frozen Vocal JEPA (P_V) & Music JEPA (P_M)."""

    def __init__(
        self,
        stft_proc: STFTProcessor,
        jepa_vocal: StemRetNetJEPA,
        jepa_music: StemRetNetJEPA,
        unet_separator: SplitLatentRetNetUNet,
        mode: str = "hybrid_ac",
        lambda_stft: float = 1.5,
        lambda_sdr: float = 0.5,
        lambda_jepa_verify: float = 0.4,
        lambda_coral: float = 0.3,
        lambda_remix_cycle: float = 0.8,
        lambda_orth: float = 0.2,
    ) -> None:
        super().__init__()
        if mode not in ("unpaired", "paired", "hybrid_ac"):
            raise ValueError(f"Unsupported separator mode: {mode}")
        self.stft_proc = stft_proc
        self.jepa_vocal = jepa_vocal
        self.jepa_music = jepa_music
        for p in self.jepa_vocal.parameters():
            p.requires_grad = False
        for p in self.jepa_music.parameters():
            p.requires_grad = False
        self.jepa_vocal.eval()
        self.jepa_music.eval()

        self.separator = unet_separator
        self.mode = mode
        self.lambda_stft = lambda_stft
        self.lambda_sdr = lambda_sdr
        self.lambda_jepa_verify = lambda_jepa_verify
        self.lambda_coral = lambda_coral
        self.lambda_remix_cycle = lambda_remix_cycle
        self.lambda_orth = lambda_orth

    def separate_waveforms(
        self, wav_mix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separate wav_mix (B, L_samples) -> (wav_vocal, wav_music, mask_vocal)."""
        orig_len = wav_mix.shape[-1]
        spec_c, _, log_mag_mix = self.stft_proc.stft(wav_mix)
        t_frames = log_mag_mix.shape[-1]
        # Pad time frames to multiple of patch_size[1] * 2 if needed
        pt_mult = self.separator.pt * 2
        pad_t = (pt_mult - (t_frames % pt_mult)) % pt_mult
        if pad_t > 0:
            log_mag_in = F.pad(log_mag_mix, (0, pad_t))
        else:
            log_mag_in = log_mag_mix

        mask_v, _, _ = self.separator(log_mag_in)
        if pad_t > 0:
            mask_v = mask_v[..., :t_frames]

        wav_v, wav_m = self.stft_proc.apply_mask_and_istft(spec_c, mask_v, length=orig_len)
        return wav_v, wav_m, mask_v

    def forward_step(
        self,
        unpaired_vocal_wav: torch.Tensor,
        unpaired_music_wav: torch.Tensor,
        unseparated_song_wav: torch.Tensor,
        paired_mix_wav: Optional[torch.Tensor] = None,
        paired_vocal_wav: Optional[torch.Tensor] = None,
        paired_music_wav: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = torch.tensor(0.0, device=unpaired_vocal_wav.device)
        metrics: Dict[str, float] = {}

        # --- 1. Unpaired Cross-Pool Dynamic Mixing + Real-Song Dual-JEPA Manifold Adaptation ---
        if self.mode in ("unpaired", "hybrid_ac"):
            # A. Cross-Pool Synthetic Mix: mix independent vocal v ~ D_V and music m ~ D_M
            syn_mix_wav = unpaired_vocal_wav + unpaired_music_wav
            _, mag_v_pool, log_v_pool = self.stft_proc.stft(unpaired_vocal_wav)
            _, mag_m_pool, log_m_pool = self.stft_proc.stft(unpaired_music_wav)
            spec_c_syn, mag_syn, log_syn = self.stft_proc.stft(syn_mix_wav)

            mask_v_syn, zv_syn, zm_syn = self.separator(log_syn)
            mag_v_est = mask_v_syn * mag_syn
            mag_m_est = (1.0 - mask_v_syn) * mag_syn
            log_v_est = torch.log1p(mag_v_est)
            log_m_est = torch.log1p(mag_m_est)

            syn_stft_loss = stft_mag_loss(mag_v_est, log_v_est, mag_v_pool, log_v_pool) + stft_mag_loss(
                mag_m_est, log_m_est, mag_m_pool, log_m_pool
            )
            wav_v_syn, wav_m_syn = self.stft_proc.apply_mask_and_istft(
                spec_c_syn, mask_v_syn, length=syn_mix_wav.shape[-1]
            )
            syn_sdr_l = sdr_waveform_loss(wav_v_syn, unpaired_vocal_wav) + sdr_waveform_loss(
                wav_m_syn, unpaired_music_wav
            )
            total = total + self.lambda_stft * syn_stft_loss + self.lambda_sdr * syn_sdr_l
            metrics["unpaired_syn_stft"] = syn_stft_loss.item()

            # B. Real Unseparated Song Adaptation (No Ground-Truth Stems!)
            spec_c_song, mag_song, log_song = self.stft_proc.stft(unseparated_song_wav)
            mask_v_song, zv_song, zm_song = self.separator(log_song)
            mag_v_song = mask_v_song * mag_song
            mag_m_song = (1.0 - mask_v_song) * mag_song
            log_v_song = torch.log1p(mag_v_song)
            log_m_song = torch.log1p(mag_m_song)

            # Orthogonality between z_V and z_M
            orth_l = self.separator.orthogonality_loss(zv_song, zm_song)
            total = total + self.lambda_orth * orth_l
            metrics["orth"] = orth_l.item()

            # Dual Frozen JEPA Manifold Verification (P_V on vocal estimate, P_M on music estimate)
            if self.lambda_jepa_verify > 0.0:
                jepa_v_l, z_ev_hat = self.jepa_vocal.verify_separated_spectrogram(log_v_song)
                jepa_m_l, z_em_hat = self.jepa_music.verify_separated_spectrogram(log_m_song)
                jepa_dual = jepa_v_l + jepa_m_l
                total = total + self.lambda_jepa_verify * jepa_dual
                metrics["jepa_dual"] = jepa_dual.item()

                if self.lambda_coral > 0.0:
                    with torch.no_grad():
                        z_ev_real = self.jepa_vocal.encoder(log_v_pool)
                        z_em_real = self.jepa_music.encoder(log_m_pool)
                    coral_l = coral_moment_loss(z_ev_hat, z_ev_real) + coral_moment_loss(z_em_hat, z_em_real)
                    total = total + self.lambda_coral * coral_l
                    metrics["coral"] = coral_l.item()

            # Cross-Song Re-Mix Cycle Consistency: mix vocal(song_i) + music(song_{i+1}) -> separate -> recover
            if self.lambda_remix_cycle > 0.0:
                wav_v_song, wav_m_song = self.stft_proc.apply_mask_and_istft(
                    spec_c_song, mask_v_song, length=unseparated_song_wav.shape[-1]
                )
                wav_m_rolled = wav_m_song.roll(shifts=1, dims=0)
                remix_wav = wav_v_song.detach() + wav_m_rolled.detach()
                _, mag_remix, log_remix = self.stft_proc.stft(remix_wav)
                mask_v_remix, _, _ = self.separator(log_remix)
                mag_v_rec = mask_v_remix * mag_remix
                mag_m_rec = (1.0 - mask_v_remix) * mag_remix
                remix_l = F.l1_loss(torch.log1p(mag_v_rec), log_v_song.detach()) + F.l1_loss(
                    torch.log1p(mag_m_rec), log_m_song.roll(shifts=1, dims=0).detach()
                )
                total = total + self.lambda_remix_cycle * remix_l
                metrics["remix_cycle"] = remix_l.item()

        # --- 2. Supervised Paired Studio Stems (Mode C / Hybrid AC) ---
        if self.mode in ("paired", "hybrid_ac") and paired_mix_wav is not None:
            spec_c_p, mag_p, log_p = self.stft_proc.stft(paired_mix_wav)
            _, mag_vp_gt, log_vp_gt = self.stft_proc.stft(paired_vocal_wav)
            _, mag_mp_gt, log_mp_gt = self.stft_proc.stft(paired_music_wav)

            mask_vp, zvp, zmp = self.separator(log_p)
            mag_vp_est = mask_vp * mag_p
            mag_mp_est = (1.0 - mask_vp) * mag_p
            log_vp_est = torch.log1p(mag_vp_est)
            log_mp_est = torch.log1p(mag_mp_est)

            p_stft_l = stft_mag_loss(mag_vp_est, log_vp_est, mag_vp_gt, log_vp_gt) + stft_mag_loss(
                mag_mp_est, log_mp_est, mag_mp_gt, log_mp_gt
            )
            wav_vp_est, wav_mp_est = self.stft_proc.apply_mask_and_istft(
                spec_c_p, mask_vp, length=paired_mix_wav.shape[-1]
            )
            p_sdr_l = sdr_waveform_loss(wav_vp_est, paired_vocal_wav) + sdr_waveform_loss(
                wav_mp_est, paired_music_wav
            )
            total = total + self.lambda_stft * p_stft_l + self.lambda_sdr * p_sdr_l
            metrics["paired_stft"] = p_stft_l.item()
            metrics["paired_sdr_l"] = p_sdr_l.item()

            if self.mode == "paired" and self.lambda_jepa_verify > 0.0:
                jv, _ = self.jepa_vocal.verify_separated_spectrogram(log_vp_est)
                jm, _ = self.jepa_music.verify_separated_spectrogram(log_mp_est)
                total = total + self.lambda_jepa_verify * (jv + jm)
                metrics["jepa_dual"] = (jv + jm).item()

        metrics["total"] = total.item()
        return total, metrics
