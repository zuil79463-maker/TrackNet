#!/usr/bin/env bash
# Build librealsense + pyrealsense2 from source on the Orange Pi AIpro (aarch64).
# The RSUSB (libusb) backend avoids kernel patches; color streaming is all this project needs.
# Usage: bash TrackNetV3/install_librealsense.sh        (LIBRS_REF=v2.xx.x to pin a release tag)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$DIR")"
PY="$ROOT/.venv/bin/python"
LIBRS_REF="${LIBRS_REF:-master}"
SRC="${LIBRS_SRC:-$HOME/librealsense}"
JOBS="${JOBS:-$(nproc)}"

[ -x "$PY" ] || { echo "Missing .venv; run bash TrackNetV3/setup_aipro.sh first" >&2; exit 1; }
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')"

PYDEV="$("$PY" -c 'import sys; print("python%d.%d-dev" % sys.version_info[:2])')"
sudo apt-get update
sudo apt-get install -y git cmake build-essential pkg-config libssl-dev libusb-1.0-0-dev libudev-dev python3-dev
sudo apt-get install -y "$PYDEV" || echo "Note: $PYDEV not in apt (fine for conda Python)."

if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 --branch "$LIBRS_REF" https://github.com/realsenseai/librealsense.git "$SRC"
fi

# Building with 4 GB RAM can run out of memory; add swap or set JOBS=2 if the compiler is killed.
cmake -S "$SRC" -B "$SRC/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE="$PY" \
    -DPYTHON_INSTALL_DIR="$SITE" \
    -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build "$SRC/build" -j "$JOBS"
sudo cmake --install "$SRC/build"
sudo ldconfig

sudo cp "$SRC/config/99-realsense-libusb.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

"$PY" -c "import pyrealsense2 as rs; print('pyrealsense2', rs.__version__ if hasattr(rs, '__version__') else 'OK')"
echo "Replug the D435i, then check: bash TrackNetV3/run_tracking.sh --list-devices"
