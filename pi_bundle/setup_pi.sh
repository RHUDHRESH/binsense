#!/usr/bin/env bash
# One-shot Raspberry Pi 4 setup for BinSense. Needs internet ONCE; after this everything runs offline.
#   bash setup_pi.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== system =="; uname -m; cat /etc/os-release | grep PRETTY; python3 --version
[ "$(uname -m)" = "aarch64" ] || { echo "Need 64-bit Pi OS (aarch64)."; exit 1; }

if [ -z "${BINSENSE_SKIP_APT:-}" ]; then
  sudo apt-get update -y
  sudo apt-get install -y python3-venv python3-pip libopenblas0 libjpeg62-turbo
fi

python3 -m venv ~/binsense-venv
source ~/binsense-venv/bin/activate
pip install --upgrade pip
# Same versions as the export machine (ExecuTorch 1.3.1 ships aarch64 wheels for Python 3.10-3.13).
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.12.1" "executorch==1.3.1" numpy pillow psutil opencv-python-headless
# Keep a local wheel cache so the install can be repeated with the network unplugged.
mkdir -p ~/binsense-wheels && pip download -d ~/binsense-wheels --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.12.1" "executorch==1.3.1" numpy pillow psutil opencv-python-headless >/dev/null 2>&1 || true

python - <<'EOF'
import torch, executorch, platform
from executorch.extension.pybindings import portable_lib as pl
print("torch", torch.__version__, "| python", platform.python_version(), "| threads", pl._threadpool_get_thread_count())
EOF
echo "== setup OK. Next: source ~/binsense-venv/bin/activate && bash tune_pi.sh && bash bench_pi.sh =="
