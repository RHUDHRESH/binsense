#!/usr/bin/env bash
# One-shot Raspberry Pi 4 setup for BinSense. Needs internet ONCE; after this everything runs offline.
#   bash setup_pi.sh            (set BINSENSE_SKIP_APT=1 to skip the apt step when you have no sudo)
set -euo pipefail
cd "$(dirname "$0")"

echo "== system =="; uname -m; grep PRETTY /etc/os-release; python3 --version
[ "$(uname -m)" = "aarch64" ] || { echo "Need 64-bit Pi OS (aarch64)."; exit 1; }

if [ -z "${BINSENSE_SKIP_APT:-}" ]; then
  sudo apt-get update -y
  sudo apt-get install -y python3-venv python3-pip libopenblas0 libjpeg62-turbo
fi

PKGS=(--extra-index-url https://download.pytorch.org/whl/cpu "torch==2.12.1" "executorch==1.3.1" numpy pillow psutil opencv-python-headless)
rm -rf ~/binsense-venv
if python3 -m venv ~/binsense-venv 2>/dev/null && [ -x ~/binsense-venv/bin/pip ]; then
  source ~/binsense-venv/bin/activate
  pip install --upgrade pip
  pip install "${PKGS[@]}"
else
  # No venv/pip on this image and no sudo: use uv, a single user-level binary.
  echo "== python venv module not available; installing with uv (no root needed) =="
  rm -rf ~/binsense-venv
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv venv ~/binsense-venv --python "$(command -v python3)"
  source ~/binsense-venv/bin/activate
  uv pip install --index-strategy unsafe-best-match "${PKGS[@]}"
fi

python - <<'EOF'
import torch, executorch, platform
from executorch.extension.pybindings import portable_lib as pl
print("torch", torch.__version__, "| python", platform.python_version(), "| threads", pl._threadpool_get_thread_count())
EOF
echo "== setup OK =="
