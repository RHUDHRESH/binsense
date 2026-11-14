#!/usr/bin/env bash
# Find which import crashes with "Illegal instruction" on this CPU.
source ~/binsense-venv/bin/activate
PY="$VIRTUAL_ENV/bin/python3"
echo "cpu features: $(grep -m1 Features /proc/cpuinfo | cut -d: -f2)"
echo "cpu part: $(grep -m1 'CPU part' /proc/cpuinfo)"
for stmt in "import numpy; print('numpy', numpy.__version__)" \
            "import torch; print('torch', torch.__version__)" \
            "import torch; x=torch.rand(1,3,8,8); print('torch math ok', float(torch.nn.functional.conv2d(x, torch.rand(2,3,3,3)).sum()) > 0)" \
            "import executorch; print('executorch pkg ok')" \
            "from executorch.extension.pybindings import portable_lib as pl; print('pybindings ok')" \
            "from executorch.extension.pybindings import portable_lib as pl; print('threads', pl._threadpool_get_thread_count())" \
            "import cv2; print('cv2', cv2.__version__)" \
            "import PIL; print('pillow', PIL.__version__)"; do
  out=$("$PY" -c "$stmt" 2>&1 | tail -1); code=$?
  echo "[$code] $stmt  =>  $out"
done
pip list 2>/dev/null | grep -i -E "^(torch|executorch|numpy|opencv|torchao) "
