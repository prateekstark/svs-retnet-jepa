# Dual-JEPA Split-Latent RetNet for Singing Voice & Music Separation

A non-adversarial framework for **Unpaired, Semi-Supervised, and Supervised Singing Voice Separation (SVS)** using **Dual Frozen RetNet-JEPA Manifold Verifiers ($P_V, P_M$)**, **Split-Latent Orthogonal RetNet + 2D U-Net Ratio Masking**, and **Cross-Song Re-Mix Cycle Consistency**.

## Core Architecture

1. **Stage 1 — Dual Domain-Specific RetNet-JEPA Priors (`train_jepa_priors.py`):**
   - **Vocal RetNet-JEPA $(E_V, \bar{E}_V, P_V)$:** Pretrained on isolated singing vocals / acapella speech to learn human vocal harmonic and formant transition dynamics.
   - **Music RetNet-JEPA $(E_M, \bar{E}_M, P_M)$:** Pretrained on isolated instrumental tracks to learn percussive transients, basslines, and chord dynamics.

2. **Stage 2 — Split-Latent RetNet + 2D U-Net Ratio-Mask Separator (`train_separator.py`):**
   - Predicts complementary STFT ratio masks $M_V(f, t) \in (0, 1)$ and $M_M(f, t) = 1 - M_V(f, t)$ on the mixture's complex STFT, guaranteeing **exact phase preservation and $10^{-7}$ mixture conservation** ($\hat{x}_V(t) + \hat{x}_M(t) = x_{\text{mix}}(t)$) via `torch.istft`.
   - Enforces **Cosine Orthogonality** $|\cos(\text{Proj}(z_V), \text{Proj}(z_M))| \to 0$ at the split-latent bottleneck.
   - Verified on unseparated real songs via **Dual Frozen JEPA Manifold Verification** ($\mathcal{L}_{P_V}(\hat{S}_V) + \mathcal{L}_{P_M}(\hat{S}_M)$), **Deep CORAL**, and **Cross-Song Re-Mix Cycle Consistency**.

## Quick Start

```bash
# Run full 5-experiment suite (Unpaired, Unpaired No-JEPA, Paired, Hybrid AC, 10% Few-Shot Hybrid AC)
# Pass MUSDB_DIR=/path/to/musdb18hq to run on MUSDB18-HQ, or omit for synthetic benchmark:
chmod +x run_svs_pipeline.sh
bash run_svs_pipeline.sh

# Separate any full-length WAV song into clean vocals and music accompaniment:
python3 separate_song.py --separator-ckpt checkpoints/sep_hybrid_ac.pt --input-wav /path/to/song.wav --out-dir outputs/my_song
```
