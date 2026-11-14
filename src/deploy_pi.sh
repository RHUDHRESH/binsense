#!/usr/bin/env bash
# Wait for the Pi to appear on the laptop hotspot, then copy the bundle, install, and run the first demo.
#   bash src/deploy_pi.sh [user]
set -uo pipefail
USER_=${1:-pi}; KEY="$HOME/.ssh/binsense_pi"; LOG=results/pi_deploy.log; mkdir -p results; : > "$LOG"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_binsense -o BatchMode=yes -o ConnectTimeout=8"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for the Pi on the hotspot (192.168.137.x) ..."
HOST=""
for try in $(seq 1 120); do
  for i in $(seq 2 60) 100 150 200 250; do (ping -n 1 -w 200 192.168.137.$i >/dev/null 2>&1 &); done; sleep 4
  for ip in $(arp -a | grep "192.168.137\." | grep dynamic | awk '{print $1}'); do
    if $SSH $USER_@$ip 'echo ok' >/dev/null 2>&1; then HOST=$ip; break; fi
  done
  [ -n "$HOST" ] && break
  if $SSH $USER_@binsense.local 'echo ok' >/dev/null 2>&1; then HOST=binsense.local; break; fi
done
[ -z "$HOST" ] && { say "FAILED: no Pi answered SSH as $USER_ within 10 minutes"; arp -a | grep "192.168.137\." | tee -a "$LOG"; exit 1; }
echo "$HOST" > results/pi_host.txt
say "connected: $USER_@$HOST"
$SSH $USER_@$HOST 'tr -d "\0" < /proc/device-tree/model; echo; grep PRETTY /etc/os-release; uname -m; free -h | sed -n 2p; df -h / | tail -1; vcgencmd measure_temp; ls /dev/video* 2>/dev/null | tr "\n" " "; echo; lsusb' 2>&1 | tee -a "$LOG"

say "copying pi_bundle ..."
tar -C pi_bundle --exclude=results --exclude=__pycache__ -cf - . | $SSH $USER_@$HOST 'mkdir -p ~/binsense && tar -C ~/binsense -xf - && cd ~/binsense && sed -i "s/\r$//" *.sh *.py *.service 2>/dev/null; ls | wc -l; du -sh .' 2>&1 | tee -a "$LOG"

say "installing (apt + ExecuTorch, several minutes) ..."
$SSH $USER_@$HOST 'cd ~/binsense && bash setup_pi.sh' >> "$LOG" 2>&1 && say "install OK" || { say "INSTALL FAILED (see $LOG)"; tail -15 "$LOG"; exit 2; }

say "first run on the Pi: demo ..."
$SSH $USER_@$HOST 'cd ~/binsense && source ~/binsense-venv/bin/activate && bash tune_pi.sh >/dev/null 2>&1; ./run_demo.sh' 2>&1 | tee -a "$LOG" | tail -45
say "DONE"
