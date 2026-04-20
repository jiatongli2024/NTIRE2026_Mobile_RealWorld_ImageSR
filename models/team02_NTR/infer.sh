#!/bin/bash
# =============================================================================
# NTR @ UIUC — OSEDiff Inference for NTIRE 2026 Mobile SR Challenge
# Best method: OSEDiff (wavelet color alignment) + sharpen + saturation
#
# Usage:
#   bash infer.sh <LR_INPUT_DIR> <SR_OUTPUT_DIR>
#
# Example:
#   bash infer.sh /path/to/test/LR_images /path/to/output/SR_images
#
# Prerequisites:
#   1. Clone OSEDiff:  git clone https://github.com/cswry/OSEDiff
#   2. Download model weights (see README.md)
#   3. Install dependencies:  pip install -r requirements.txt
#   4. Set OSEDIFF_DIR and HF_HOME below (or via env vars)
# =============================================================================

set -euo pipefail

# ---- Input arguments --------------------------------------------------------
INPUT_DIR="${1:-}"
OUTPUT_DIR_FINAL="${2:-}"

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR_FINAL" ]; then
    echo "Usage: bash infer.sh <LR_INPUT_DIR> <SR_OUTPUT_DIR>"
    echo ""
    echo "  LR_INPUT_DIR   — directory with LR input images (.png)"
    echo "  SR_OUTPUT_DIR  — directory to write final SR outputs"
    exit 1
fi

# ---- Paths (override via environment variables) -----------------------------
# Path to the OSEDiff repository (cloned from https://github.com/cswry/OSEDiff)
OSEDIFF_DIR="/home/NTIRE26/data2/NTIRE2026/Mobile/NTIRE2026_Mobile_RealWorld_ImageSR/Team_code/Team_03/code/OSEDiff"

# Path where model weights are stored (HuggingFace cache root)
HF_HOME="/home/NTIRE26/data2/NTIRE2026/Mobile/NTIRE2026_Mobile_RealWorld_ImageSR/Team_code/Team_03/code/hf_cache"

# Python executable (uses current environment's python by default)
PYTHON="${PYTHON:-python3}"

# GPU index
export CUDA_VISIBLE_DEVICES=2
export HF_HOME

# ---- Derived paths ----------------------------------------------------------
SD21_PATH=$(find "$HF_HOME" -name "model_index.json" 2>/dev/null | head -1 | xargs -I{} dirname {})
SD21_PATH="/home/NTIRE26/data2/NTIRE2026/Mobile/NTIRE2026_Mobile_RealWorld_ImageSR/Team_code/Team_03/code/hf_cache/sd21"
if [ -z "$SD21_PATH" ]; then
    echo "ERROR: Could not find Stable Diffusion 2.1 model under $HF_HOME"
    echo "       Download it with:"
    echo "         python -c \"from huggingface_hub import snapshot_download; snapshot_download('Manojb/stable-diffusion-2-1-base', local_dir='$HF_HOME/sd21')\""
    exit 1
fi

RAM_BASE="$HF_HOME/ram_models/ram_swin_large_14m.pth"
RAM_FT="$HF_HOME/ram_models/DAPE.pth"
OSEDIFF_PKL="$OSEDIFF_DIR/preset/models/osediff.pkl"
OUTPUT_DIR_RAW="${OUTPUT_DIR_FINAL}_wavelet_raw"

# ---- Validation -------------------------------------------------------------
for f in "$RAM_BASE" "$RAM_FT" "$OSEDIFF_PKL"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing model file: $f"
        echo "       See README.md for download instructions."
        exit 1
    fi
done

N_INPUT=$(ls "$INPUT_DIR"/*.png 2>/dev/null | wc -l)
echo "=== NTR OSEDiff Mobile SR Inference ==="
echo "Input:       $INPUT_DIR ($N_INPUT images)"
echo "Output:      $OUTPUT_DIR_FINAL"
echo "OSEDiff dir: $OSEDIFF_DIR"
echo "SD 2.1:      $SD21_PATH"
echo "GPU:         CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

mkdir -p "$OUTPUT_DIR_RAW" "$OUTPUT_DIR_FINAL"

# ---- Step 1: OSEDiff SR inference (wavelet color alignment) -----------------
echo "[1/2] Running OSEDiff (wavelet alignment, process_size=512, FP16)..."
cd "$OSEDIFF_DIR"

$PYTHON test_osediff.py \
    --input_image     "$INPUT_DIR" \
    --output_dir      "$OUTPUT_DIR_RAW" \
    --pretrained_model_name_or_path "$SD21_PATH" \
    --osediff_path    "$OSEDIFF_PKL" \
    --ram_path        "$RAM_BASE" \
    --ram_ft_path     "$RAM_FT" \
    --upscale         4 \
    --mixed_precision fp16 \
    --align_method    wavelet \
    --process_size    512 \
    --vae_encoder_tiled_size 1024 \
    --vae_decoder_tiled_size 224 \
    --latent_tiled_size 96 \
    --latent_tiled_overlap 32

echo ""
echo "[2/2] Applying post-processing (unsharp mask + saturation boost)..."

$PYTHON "$(dirname "$0")/postprocess.py" \
    --input_dir  "$OUTPUT_DIR_RAW" \
    --output_dir "$OUTPUT_DIR_FINAL"

N_OUT=$(ls "$OUTPUT_DIR_FINAL"/*.png 2>/dev/null | wc -l)
echo ""
echo "=== Done. $N_OUT SR images written to $OUTPUT_DIR_FINAL ==="
