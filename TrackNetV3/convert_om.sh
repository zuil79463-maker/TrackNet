#!/usr/bin/env bash
# Convert the exported ONNX model to an Ascend OM model with ATC on the Orange Pi AIpro.
# Usage: bash TrackNetV3/convert_om.sh [input.onnx] [output_prefix]
#   SOC_VERSION=Ascend310B4 overrides the chip auto-detected from npu-smi.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX="${1:-$DIR/ckpts/TrackNet_best.onnx}"
OUTPUT="${2:-$DIR/ckpts/TrackNet_best}"
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

[ -f "$ONNX" ] || { echo "Missing $ONNX. Run export_onnx.py on the PC and copy the file here." >&2; exit 1; }
if ! command -v atc >/dev/null && [ -f "$ASCEND_ENV" ]; then
    set +u; source "$ASCEND_ENV"; set -u
fi
command -v atc >/dev/null || { echo "atc not found; install CANN toolkit or set ASCEND_ENV." >&2; exit 1; }

SOC="${SOC_VERSION:-}"
if [ -z "$SOC" ] && command -v npu-smi >/dev/null; then
    CHIP="$(npu-smi info 2>/dev/null | grep -oE '310B[0-9]+' | head -n 1 || true)"
    [ -n "$CHIP" ] && SOC="Ascend$CHIP"
fi
[ -n "$SOC" ] || { echo "Cannot detect the chip. Check npu-smi info and set SOC_VERSION, e.g. Ascend310B4." >&2; exit 1; }

echo "ATC: $ONNX -> $OUTPUT.om (soc_version=$SOC). This can take several minutes."
# Static ONNX shape from export_onnx.py; float32 input/output, NPU computes in FP16 internally.
atc --framework=5 \
    --model="$ONNX" \
    --output="$OUTPUT" \
    --input_format=NCHW \
    --soc_version="$SOC" \
    --log=error
echo "Done: $OUTPUT.om. Run: bash TrackNetV3/run_tracking.sh"
