#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "/content/drive/MyDrive" ]; then
    DRIVE_SAVE_DIR="${DRIVE_SAVE_DIR:-/content/drive/MyDrive/svs-retnet-jepa-runs}"
    mkdir -p "${DRIVE_SAVE_DIR}"
    CKPT_DIR="${CKPT_DIR:-${DRIVE_SAVE_DIR}/checkpoints}"
    OUT_DIR="${OUT_DIR:-${DRIVE_SAVE_DIR}/outputs}"
    RESULTS_FILE="${DRIVE_SAVE_DIR}/svs_results.txt"
    echo "[Google Drive detected] Saving checkpoints, WAVs, and results to: ${DRIVE_SAVE_DIR}"
else
    CKPT_DIR="${CKPT_DIR:-${REPO_DIR}/checkpoints}"
    OUT_DIR="${OUT_DIR:-${REPO_DIR}/outputs}"
    RESULTS_FILE="${REPO_DIR}/svs_results.txt"
fi
PY="${PYTHON:-$(command -v python3 || command -v python)}"

MUSDB_ARGS=()
if [ -n "${MUSDB_DIR:-}" ] && [ -d "${MUSDB_DIR}" ]; then
    MUSDB_ARGS=(--musdb-dir "${MUSDB_DIR}")
fi

EPOCHS_JEPA="${EPOCHS_JEPA:-40}"
EPOCHS_SEP="${EPOCHS_SEP:-40}"

mkdir -p "${CKPT_DIR}" "${OUT_DIR}"
cd "${REPO_DIR}"

echo "=== [1/3] Stage 1: Pretraining Dual RetNet-JEPA Priors (Vocal JEPA & Music JEPA) ==="
$PY train_jepa_priors.py --stem vocal "${MUSDB_ARGS[@]}" --epochs "${EPOCHS_JEPA}" --out "${CKPT_DIR}/jepa_vocal.pt"
$PY train_jepa_priors.py --stem music "${MUSDB_ARGS[@]}" --epochs "${EPOCHS_JEPA}" --out "${CKPT_DIR}/jepa_music.pt"

echo "=== [2/3] Stage 2: Training Dual-JEPA Split-Latent RetNet Separators & Ablations ==="
$PY train_separator.py --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" "${MUSDB_ARGS[@]}" --mode unpaired  --epochs "${EPOCHS_SEP}" --out "${CKPT_DIR}/sep_unpaired.pt"
$PY train_separator.py --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" "${MUSDB_ARGS[@]}" --mode unpaired  --lambda-jepa-verify 0.0 --epochs "${EPOCHS_SEP}" --out "${CKPT_DIR}/sep_unpaired_no_jepa.pt"
$PY train_separator.py --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" "${MUSDB_ARGS[@]}" --mode paired    --epochs "${EPOCHS_SEP}" --out "${CKPT_DIR}/sep_paired.pt"
$PY train_separator.py --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" "${MUSDB_ARGS[@]}" --mode hybrid_ac --epochs "${EPOCHS_SEP}" --out "${CKPT_DIR}/sep_hybrid_ac.pt"
$PY train_separator.py --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" "${MUSDB_ARGS[@]}" --mode hybrid_ac --paired-fraction 0.10 --epochs "${EPOCHS_SEP}" --out "${CKPT_DIR}/sep_hybrid_ac_10pct.pt"

echo "=== [3/3] Evaluating SDR / SI-SDR (dB) & Exporting Separated Singing + Music Stems ==="
{
    echo "--- 1. Unpaired Dual-JEPA + Re-Mix Cycle + CORAL (0 paired songs) ---"
    $PY evaluate_svs.py --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" --separator-ckpt "${CKPT_DIR}/sep_unpaired.pt"         "${MUSDB_ARGS[@]}" --out-dir "${OUT_DIR}/samples_unpaired"
    echo "--- 2. Ablation: Unpaired WITHOUT Dual-JEPA Verification ---"
    $PY evaluate_svs.py --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" --separator-ckpt "${CKPT_DIR}/sep_unpaired_no_jepa.pt" "${MUSDB_ARGS[@]}" --out-dir "${OUT_DIR}/samples_unpaired_no_jepa"
    echo "--- 3. Paired Studio Stems Only (100% paired) ---"
    $PY evaluate_svs.py --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" --separator-ckpt "${CKPT_DIR}/sep_paired.pt"           "${MUSDB_ARGS[@]}" --out-dir "${OUT_DIR}/samples_paired"
    echo "--- 4. Proposed Hybrid AC (Full Dual-JEPA + Re-Mix Cycle + Paired) ---"
    $PY evaluate_svs.py --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" --separator-ckpt "${CKPT_DIR}/sep_hybrid_ac.pt"        "${MUSDB_ARGS[@]}" --out-dir "${OUT_DIR}/samples_hybrid_ac"
    echo "--- 5. Few-Shot Hybrid AC (Only 10% Paired Studio Songs) ---"
    $PY evaluate_svs.py --ckpt-vocal "${CKPT_DIR}/jepa_vocal.pt" --ckpt-music "${CKPT_DIR}/jepa_music.pt" --separator-ckpt "${CKPT_DIR}/sep_hybrid_ac_10pct.pt"  "${MUSDB_ARGS[@]}" --out-dir "${OUT_DIR}/samples_hybrid_ac_10pct"
} | tee "${REPO_DIR}/svs_results.txt"

if [ "${RESULTS_FILE}" != "${REPO_DIR}/svs_results.txt" ]; then
    cp -f "${REPO_DIR}/svs_results.txt" "${RESULTS_FILE}"
fi

echo "=== Complete! Metrics saved in ${RESULTS_FILE} and WAV stems in ${OUT_DIR}/ ==="
