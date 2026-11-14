#!/usr/bin/env bash
# The stock ExecuTorch 1.3.1 wheel crashes on the Pi 4 (Illegal instruction). Try other released wheels.
source ~/binsense-venv/bin/activate
PY="$VIRTUAL_ENV/bin/python3"
echo "available: $(pip index versions executorch 2>/dev/null | grep -i 'available' | cut -c1-200)"
echo "gcc: $(gcc --version 2>/dev/null | head -1 || echo none)   cmake: $(cmake --version 2>/dev/null | head -1 || echo none)   g++: $(command -v g++ || echo none)"
for v in "$@"; do
  pip install -q --no-deps --extra-index-url https://download.pytorch.org/whl/cpu "executorch==$v" >/dev/null 2>&1 || { echo "$v: no wheel"; continue; }
  "$PY" -c "from executorch.extension.pybindings import portable_lib as pl; print('IMPORT_OK threads', pl._threadpool_get_thread_count())" > /tmp/et_try.txt 2>&1
  code=$?
  echo "$v: exit $code  $(tail -1 /tmp/et_try.txt | cut -c1-120)"
  if [ $code -eq 0 ]; then
    cd ~/binsense
    "$PY" - <<'EOF' 2>&1 | tail -3
import time, torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch
for name, res in (("models/stu_mobilenetv2_100_96_fp32.pte", 96), ("models/mobileclip2-s0_192_fp32.pte", 192)):
    try:
        m = _load_for_executorch(name); x = torch.rand(1, 3, res, res); m.forward((x,))
        t = time.perf_counter(); out = m.forward((x,))[0]; print(name, "RUNS, out", tuple(out.shape), round((time.perf_counter() - t) * 1000, 1), "ms")
    except Exception as e:
        print(name, "FAILED", type(e).__name__, str(e)[:100])
EOF
  fi
done
