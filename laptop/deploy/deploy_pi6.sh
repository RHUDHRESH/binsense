#!/usr/bin/env bash
# Copy the bundle to the Pi over its IPv6 link-local address, install without sudo, run the first demo.
set -uo pipefail
H=${1:-'pi@fe80::da3a:ddff:fec0:7287%9'}; KEY="$HOME/.ssh/binsense_pi"; LOG=results/logs/logs/pi_deploy.log; mkdir -p results/logs; : > "$LOG"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "copying bundle ..."
tar -C pi_bundle --exclude=results --exclude=__pycache__ --exclude=my_photos -cf - . | $SSH "$H" 'mkdir -p ~/binsense && tar -C ~/binsense -xf - && cd ~/binsense && sed -i "s/\r$//" *.sh *.py *.service 2>/dev/null; echo "files: $(ls | wc -l)  size: $(du -sh . | cut -f1)"' 2>&1 | tee -a "$LOG"
say "installing (no sudo) ..."
$SSH "$H" 'cd ~/binsense && BINSENSE_SKIP_APT=1 bash setup_pi.sh' >> "$LOG" 2>&1 && say "install OK" || { say "INSTALL FAILED"; tail -20 "$LOG"; exit 2; }
say "first run on the Pi ..."
$SSH "$H" 'cd ~/binsense && bash run_demo.sh --images organiser' 2>&1 | tee -a "$LOG" | tail -60
say "DONE"
