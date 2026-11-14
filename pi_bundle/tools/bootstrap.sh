#!/usr/bin/env bash
# One command on a fresh Raspberry Pi (needs internet once). Fetches BinSense, installs it, runs the demo.
#   curl -sL https://raw.githubusercontent.com/RHUDHRESH/binsense/main/pi_bundle/tools/bootstrap.sh | bash
set -uo pipefail
echo "== BinSense bootstrap =="
tr -d '\0' < /proc/device-tree/model 2>/dev/null; echo
sudo apt-get update -y >/dev/null 2>&1
sudo apt-get install -y git >/dev/null 2>&1
rm -rf ~/binsense_repo
git clone --depth 1 https://github.com/RHUDHRESH/binsense ~/binsense_repo || { echo "clone failed"; exit 1; }
mkdir -p ~/binsense
cp -r ~/binsense_repo/pi_bundle/. ~/binsense/
cd ~/binsense
sed -i 's/\r$//' *.sh *.py *.service 2>/dev/null
bash setup_pi.sh || { echo "SETUP FAILED"; exit 2; }
source ~/binsense-venv/bin/activate
bash tune_pi.sh
echo "== cameras =="; ls /dev/video* 2>/dev/null || echo "no camera device found"
echo "== first run =="
"$VIRTUAL_ENV/bin/python3" demo.py
echo
echo "Next:  cd ~/binsense && source ~/binsense-venv/bin/activate"
echo "       python3 show.py        (whole proof, step by step)"
echo "       python3 live.py        (webcam)"
echo "       bash bench_pi.sh       (full benchmark)"
