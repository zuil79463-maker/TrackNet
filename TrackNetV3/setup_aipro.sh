#!/usr/bin/env bash
# Prepare the Orange Pi AIpro: Python venv, runtime packages, RealSense SDK and USB permissions.
# Usage (from the repository root, as the normal user, e.g. HwHiAiUser):
#   bash TrackNetV3/setup_aipro.sh
# Environment overrides: PYTHON=/path/to/python3  ASCEND_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$DIR")"
VENV="$ROOT/.venv"
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
RULES_URL="https://raw.githubusercontent.com/realsenseai/librealsense/master/config/99-realsense-libusb.rules"

echo "== System =="
uname -m
[ "$(uname -m)" = "aarch64" ] || echo "Warning: expected aarch64 (Orange Pi AIpro)." >&2

if [ -f "$ASCEND_ENV" ]; then
    set +u; source "$ASCEND_ENV"; set -u
    echo "CANN environment: $ASCEND_ENV"
else
    echo "Warning: $ASCEND_ENV not found. Install CANN (preinstalled on the official AIpro image)." >&2
fi
command -v npu-smi >/dev/null && npu-smi info || echo "Warning: npu-smi unavailable." >&2

# Use the same Python that pyACL supports (the official image's default python3).
PYTHON="${PYTHON:-$(command -v python3)}"
echo "Python: $PYTHON ($("$PYTHON" --version 2>&1))"
"$PYTHON" -c "import acl" 2>/dev/null && echo "pyACL: OK" || \
    echo "Warning: pyACL cannot be imported by $PYTHON; NPU inference will be unavailable." >&2

echo "== System packages =="
if command -v apt-get >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3-venv libusb-1.0-0 libgl1 libglib2.0-0 curl || \
        echo "Warning: some apt packages failed to install." >&2
fi

echo "== Python venv: $VENV =="
# --system-site-packages keeps CANN Python modules (acl, te, ...) visible inside the venv.
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$DIR/requirements-aipro.txt" || \
    "$VENV/bin/python" -m pip install numpy opencv-python Pillow

if ! "$VENV/bin/python" -c "import pyrealsense2" 2>/dev/null; then
    echo "pyrealsense2 wheel unavailable for this platform." >&2
    echo "Build it from source (about 30-60 min): bash TrackNetV3/install_librealsense.sh" >&2
fi

echo "== RealSense USB permissions =="
if [ ! -f /etc/udev/rules.d/99-realsense-libusb.rules ]; then
    if curl -fsSL "$RULES_URL" -o /tmp/99-realsense-libusb.rules; then
        sudo cp /tmp/99-realsense-libusb.rules /etc/udev/rules.d/
        sudo udevadm control --reload-rules && sudo udevadm trigger
        echo "udev rules installed; replug the D435i."
    else
        echo "Warning: cannot download udev rules; run as root or install them manually." >&2
    fi
fi

echo "== Check =="
"$VENV/bin/python" - <<'EOF'
import importlib
for name in ('numpy', 'cv2', 'PIL', 'pyrealsense2', 'acl'):
    try:
        importlib.import_module(name)
        print(f'{name}: OK')
    except Exception as error:
        print(f'{name}: MISSING ({error})')
EOF
echo "Next: copy ckpts/TrackNet_best.onnx here, then run: bash TrackNetV3/convert_om.sh"
