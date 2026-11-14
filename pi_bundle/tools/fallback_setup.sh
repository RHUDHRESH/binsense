#!/usr/bin/env bash
# Plain-PyTorch fallback on the Pi (used while ExecuTorch is being built from source for this CPU).
source ~/binsense-venv/bin/activate
pip install -q open_clip_torch timm 2>&1 | tail -2
cd ~/binsense
export BINSENSE_BACKEND=torch
"$VIRTUAL_ENV/bin/python3" run_eval.py --pte models/mobileclip2-s0_192_fp32.pte --pack packs/organiser_192.npz --images organiser --csv organiser/labels.csv --threads 4 --repeat 2 --out results --tag pi_torch_192 2>&1 | grep -v -E "Warning|warn" | tail -34
pkill -f ui_server.py 2>/dev/null
nohup "$VIRTUAL_ENV/bin/python3" ui_server.py 8000 > ~/ui.log 2>&1 &
sleep 4; curl -s -o /dev/null -w "pi ui: %{http_code}\n" http://localhost:8000/
