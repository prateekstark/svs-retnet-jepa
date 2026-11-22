#!/usr/bin/env bash
# Complete Publishable MUSDB18-HQ Benchmark Runner (Dual-JEPA SVS v2)
# Designed for a single 24 GB GPU (RTX 4090 / A100) on RunPod, Vast.ai, or Lambda Labs.
# Usage:
#   bash run_musdb18_hq_cloud.sh /path/to/musdb18hq
# If MUSDB18-HQ is not yet downloaded, it automatically fetches it from Zenodo (~22.7 GB).
set -euo pipefail

MUSDB_DIR="${1:-./data/musdb18hq}"
CFG="configs/musdb18_hq_v2.yaml"
CKPT_DIR="./checkpoints_hq_v2"
OUT_DIR="./outputs_hq_v2"
mkdir -p "${CKPT_DIR}" "${OUT_DIR}"

if [ ! -d "${MUSDB_DIR}/train" ] || [ ! -d "${MUSDB_DIR}/test" ]; then
    echo "=== [0/4] Downloading Full MUSDB18-HQ (~22.7 GB) from Zenodo ==="
    mkdir -p "${MUSDB_DIR}"
    curl -f -L -C - --retry 5 -o "${MUSDB_DIR}/musdb18hq.zip" \
        "https://zenodo.org/api/records/3338373/files/musdb18hq.zip/content"
    unzip -o -q "${MUSDB_DIR}/musdb18hq.zip" -d "${MUSDB_DIR}"
    rm -f "${MUSDB_DIR}/musdb18hq.zip"
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=== [1/4] Stage 1: Vocal & Music RetNet-JEPA Priors (Early Stopping, patience=20) ==="
python3 train_jepa_priors.py --config "${CFG}" --stem vocal --musdb-dir "${MUSDB_DIR}" \
    --epochs 150 --patience 20 --val-fraction 0.14 --out "${CKPT_DIR}/jepa_vocal.pt"
python3 train_jepa_priors.py --config "${CFG}" --stem music --musdb-dir "${MUSDB_DIR}" \
    --epochs 150 --patience 20 --val-fraction 0.14 --out "${CKPT_DIR}/jepa_music.pt"

echo "=== [2/4] Stage 2: Full Paper Ablation & Data-Efficiency Suite (Early Stopping, patience=15) ==="
# 1. Proposed 0%-Paired Dual-JEPA Separator
python3 train_separator.py --config "${CFG}" \
    --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" \
    --musdb-dir "${MUSDB_DIR}" --mode unpaired --epochs 200 --patience 15 --val-fraction 0.14 \
    --out "${CKPT_DIR}/sep_unpaired_dual_jepa.pt"

# 2. Ablation: 0%-Paired WITHOUT Dual-JEPA Verification (lambda_jepa_verify=0)
python3 train_separator.py --config "${CFG}" \
    --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" \
    --musdb-dir "${MUSDB_DIR}" --mode unpaired --lambda-jepa-verify 0.0 --epochs 200 --patience 15 --val-fraction 0.14 \
    --out "${CKPT_DIR}/sep_unpaired_no_jepa.pt"

# 3. Proposed 10% Few-Shot Hybrid AC (Dual-JEPA + 10% paired studio songs)
python3 train_separator.py --config "${CFG}" \
    --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" \
    --musdb-dir "${MUSDB_DIR}" --mode hybrid_ac --paired-fraction 0.10 --epochs 200 --patience 15 --val-fraction 0.14 \
    --out "${CKPT_DIR}/sep_hybrid_ac_10pct.pt"

# 4. Proposed 25% Few-Shot Hybrid AC
python3 train_separator.py --config "${CFG}" \
    --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" \
    --musdb-dir "${MUSDB_DIR}" --mode hybrid_ac --paired-fraction 0.25 --epochs 200 --patience 15 --val-fraction 0.14 \
    --out "${CKPT_DIR}/sep_hybrid_ac_25pct.pt"

# 5. Fully Supervised 100% Paired Baseline
python3 train_separator.py --config "${CFG}" \
    --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" \
    --musdb-dir "${MUSDB_DIR}" --mode paired --paired-fraction 1.0 --lambda-jepa-verify 0.0 --epochs 200 --patience 15 --val-fraction 0.14 \
    --out "${CKPT_DIR}/sep_paired_100pct.pt"

echo "=== [3/4] Stage 3: Official 50-Song MUSDB18-HQ Test Evaluation ==="
for NAME in sep_unpaired_no_jepa sep_unpaired_dual_jepa sep_hybrid_ac_10pct sep_hybrid_ac_25pct sep_paired_100pct; do
    echo ">>> Evaluating ${NAME}..."
    python3 evaluate_svs.py --config "${CFG}" \
        --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" \
        --separator-ckpt "${CKPT_DIR}/${NAME}.pt" \
        --musdb-dir "${MUSDB_DIR}" --out-dir "${OUT_DIR}/${NAME}" --num-wavs 5 \
        | tee "${OUT_DIR}/${NAME}_results.txt"
done

echo "=== [4/4] COMPLETE! All results & separated WAVs in ${OUT_DIR} ==="
