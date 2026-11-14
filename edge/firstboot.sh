#!/bin/bash
# Runs once as root on the Pi's first boot (started by cloud-init). Installs BinSense with no one logged in.
# Progress: /home/pi/firstboot.log     Finished marker: /home/pi/FIRSTBOOT_DONE
exec > /home/pi/firstboot.log 2>&1
echo "[$(date)] firstboot start"
for i in $(seq 1 90); do curl -sfI https://github.com >/dev/null && break; sleep 4; done
apt-get update -y
apt-get install -y git python3-venv python3-pip libopenblas0 libjpeg62-turbo
sudo -u pi -H bash -c '
  rm -rf ~/binsense_repo
  git clone --depth 1 https://github.com/RHUDHRESH/binsense ~/binsense_repo || exit 1
  mkdir -p ~/binsense && cp -r ~/binsense_repo/pi_bundle/. ~/binsense/
  cd ~/binsense && sed -i "s/\r$//" *.sh *.py *.service 2>/dev/null
  BINSENSE_SKIP_APT=1 bash setup_pi.sh
' || echo "SETUP FAILED"
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
sudo -u pi -H bash -c '
  cd ~/binsense && source ~/binsense-venv/bin/activate
  "$VIRTUAL_ENV/bin/python3" demo.py > ~/demo_first_run.log 2>&1
  nohup "$VIRTUAL_ENV/bin/python3" ui_server.py 8000 > ~/ui.log 2>&1 &
'
chown pi:pi /home/pi/firstboot.log
touch /home/pi/FIRSTBOOT_DONE; chown pi:pi /home/pi/FIRSTBOOT_DONE
echo "[$(date)] firstboot done"
