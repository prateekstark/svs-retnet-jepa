#!/usr/bin/env bash
# Complete Publishable MUSDB18-HQ Benchmark Runner (Dual-JEPA SVS v2)
# Budget-hardened for a single 24-32 GB GPU (RTX 4090 / RTX 5090 / A100) on Vast.ai / RunPod.
# Runs highest-impact paper models first and evaluates each model immediately on the 50-song test split.
# Usage:
#   bash run_musdb18_hq_cloud.sh ./data/musdb18hq
set -euo pipefail

MUSDB_DIR="${1:-./data/musdb18hq}"
CFG="configs/musdb18_hq_v2.yaml"
CKPT_DIR="${CKPT_DIR:-./checkpoints_hq_v2}"
OUT_DIR="${OUT_DIR:-./outputs_hq_v2}"
mkdir -p "${CKPT_DIR}" "${OUT_DIR}"

FOUND_TRAIN=$(find "${MUSDB_DIR}" -maxdepth 3 -type d -name train 2>/dev/null | head -n 1 || true)
if [ -z "${FOUND_TRAIN}" ]; then
    echo "=== [0/3] Downloading Full MUSDB18-HQ (~22.7 GB) from Zenodo ==="
    mkdir -p "${MUSDB_DIR}"
    for i in 1 2 3 4 5 6 7 8; do
        curl -f -L -C - --http1.1 --connect-timeout 30 --max-time 1800 \
            -o "${MUSDB_DIR}/musdb18hq.zip" \
            "https://zenodo.org/api/records/3338373/files/musdb18hq.zip/content" && break
        echo "Retrying Zenodo download (attempt ${i})..."
        sleep 5
    done
    unzip -o -q "${MUSDB_DIR}/musdb18hq.zip" -d "${MUSDB_DIR}"
    rm -f "${MUSDB_DIR}/musdb18hq.zip"
    FOUND_TRAIN=$(find "${MUSDB_DIR}" -maxdepth 3 -type d -name train | head -n 1)
fi

MUSDB_ROOT="$(dirname "${FOUND_TRAIN}")"
echo "MUSDB18-HQ verified at: ${MUSDB_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

eval_model() {
    local NAME="$1"
    echo ">>> Evaluating ${NAME} on 50-song MUSDB18-HQ test split..."
    python3 evaluate_svs.py --config "${CFG}" \
        --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" \
        --separator-ckpt "${CKPT_DIR}/${NAME}.pt" \
        --musdb-dir "${MUSDB_ROOT}" --out-dir "${OUT_DIR}/${NAME}" --num-wavs 5 \
        | tee "${OUT_DIR}/${NAME}_results.txt"
}

echo "=== [1/3] Stage 1: Vocal & Music RetNet-JEPA Priors (val_margin early stopping, min 25 ep, patience 15) ==="
python3 train_jepa_priors.py --config "${CFG}" --stem vocal --musdb-dir "${MUSDB_ROOT}" \
    --epochs 120 --min-epochs 25 --patience 15 --val-fraction 0.14 --out "${CKPT_DIR}/jepa_vocal.pt"
python3 train_jepa_priors.py --config "${CFG}" --stem music --musdb-dir "${MUSDB_ROOT}" \
    --epochs 120 --min-epochs 25 --patience 15 --val-fraction 0.14 --out "${CKPT_DIR}/jepa_music.pt"

echo "=== [2/3] Stage 2: Paper Benchmark Suite (Ordered by Paper Impact, Immediate Eval After Each Model) ==="

# Model 1 (Flagship #1): 0%-Paired Dual-JEPA Separator
python3 train_separator.py --config "${CFG}" \
    --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" \
    --musdb-dir "${MUSDB_ROOT}" --mode unpaired --epochs 150 --min-epochs 15 --patience 12 --val-fraction 0.14 \
    --out "${CKPT_DIR}/sep_unpaired_dual_jepa.pt"
eval_model sep_unpaired_dual_jepa

# Model 2 (Core Ablation): 0%-Paired WITHOUT Dual-JEPA Verification (lambda_jepa_verify=0)
python3 train_separator.py --config "${CFG}" \
    --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" \
    --musdb-dir "${MUSDB_ROOT}" --mode unpaired --lambda-jepa-verify 0.0 --epochs 150 --min-epochs 15 --patience 12 --val-fraction 0.14 \
    --out "${CKPT_DIR}/sep_unpaired_no_jepa.pt"
eval_model sep_unpaired_no_jepa

# Model 3 (Flagship #2): 10% Few-Shot Hybrid AC (Dual-JEPA + 10% paired studio songs)
python3 train_separator.py --config "${CFG}" \
    --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" \
    --musdb-dir "${MUSDB_ROOT}" --mode hybrid_ac --paired-fraction 0.10 --epochs 150 --min-epochs 15 --patience 12 --val-fraction 0.14 \
    --out "${CKPT_DIR}/sep_hybrid_ac_10pct.pt"
eval_model sep_hybrid_ac_10pct

# Model 4 (Upper Bound Baseline): 100% Fully Supervised Paired
python3 train_separator.py --config "${CFG}" \
    --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" \
    --musdb-dir "${MUSDB_ROOT}" --mode paired --paired-fraction 1.0 --lambda-jepa-verify 0.0 --epochs 150 --min-epochs 15 --patience 12 --val-fraction 0.14 \
    --out "${CKPT_DIR}/sep_paired_100pct.pt"
eval_model sep_paired_100pct

# Model 5 (Data-Efficiency Curve Point): 25% Few-Shot Hybrid AC
python3 train_separator.py --config "${CFG}" \
    --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" \
    --musdb-dir "${MUSDB_ROOT}" --mode hybrid_ac --paired-fraction 0.25 --epochs 150 --min-epochs 15 --patience 12 --val-fraction 0.14 \
    --out "${CKPT_DIR}/sep_hybrid_ac_25pct.pt"
eval_model sep_hybrid_ac_25pct

echo "=== [3/3] COMPLETE! Summary of all 5 models: ==="
cat "${OUT_DIR}"/*_results.txt
