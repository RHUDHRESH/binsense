#!/usr/bin/env bash
# Build ExecuTorch 1.3.1 from source ON the Raspberry Pi 4, because the prebuilt aarch64 wheels crash on
# Cortex-A72 ("Illegal instruction"). Adapted from Arm's course script build_etdump_runtime.sh.
# Needs: git cmake ninja-build python3-dev (apt). Log: ~/et_build.log   Done marker: ~/ET_BUILD_DONE
set -uo pipefail
exec > ~/et_build.log 2>&1
echo "[$(date)] build start"; vcgencmd measure_temp
source ~/binsense-venv/bin/activate
SRC=~/executorch-src/executorch
if [ ! -d "$SRC/.git" ]; then
  mkdir -p ~/executorch-src
  git clone --branch v1.3.1 --depth 1 --recurse-submodules --shallow-submodules https://github.com/pytorch/executorch.git "$SRC" || { echo "CLONE FAILED"; exit 1; }
fi
git -C "$SRC" submodule update --init --recursive --depth 1
pip install -q --extra-index-url https://download.pytorch.org/whl/cpu "executorch==1.3.1"      # brings the build-time Python deps
pip install -q --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==2.12.1+cpu"
pip install -q setuptools wheel pyyaml zstd certifi "cmake<4" 2>/dev/null
mkdir -p ~/tmpbuild && export TMPDIR=~/tmpbuild   # /tmp is RAM-backed on this OS and fills up during the build
export CMAKE_BUILD_PARALLEL_LEVEL=4
export CFLAGS="-march=armv8-a" CXXFLAGS="-march=armv8-a"
export CMAKE_ARGS="-DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_DEVTOOLS=OFF -DEXECUTORCH_ENABLE_EVENT_TRACER=OFF -DEXECUTORCH_BUILD_OPENVINO=OFF -DEXECUTORCH_BUILD_COREML=OFF -DEXECUTORCH_BUILD_EXTENSION_TRAINING=OFF -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=OFF -DEXECUTORCH_BUILD_EXTENSION_LLM=OFF -DEXECUTORCH_BUILD_KERNELS_LLM=OFF -DEXECUTORCH_BUILD_KERNELS_LLM_AOT=OFF -DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=OFF"
echo "[$(date)] compiling (this is the long part)"
pip install --force-reinstall --no-build-isolation --no-deps "$SRC" || { echo "BUILD FAILED"; exit 2; }
echo "[$(date)] testing"
cd ~/binsense
"$VIRTUAL_ENV/bin/python3" - <<'EOF'
import time, torch
from executorch.extension.pybindings import portable_lib as pl
print("pybindings OK, threads", pl._threadpool_get_thread_count())
for name, res in (("models/stu_mobilenetv2_100_96_fp32.pte", 96), ("models/mobileclip2-s0_192_fp32.pte", 192)):
    m = pl._load_for_executorch(name); x = torch.rand(1, 3, res, res); m.forward((x,))
    t = time.perf_counter(); m.forward((x,)); print(name, "RUNS in", round((time.perf_counter() - t) * 1000, 1), "ms")
EOF
[ $? -eq 0 ] && touch ~/ET_BUILD_DONE
echo "[$(date)] build end"; vcgencmd measure_temp; vcgencmd get_throttled
