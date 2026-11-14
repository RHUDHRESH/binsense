#!/usr/bin/env bash
# Hardware inspection and tuning tools for the Raspberry Pi 4. Run once after first boot (needs internet):
#   bash provision_tools.sh
set -uo pipefail

sudo apt-get update -y
sudo apt-get install -y \
  htop btop iotop sysstat \
  linux-perf stress-ng sysbench \
  cpufrequtils lm-sensors \
  i2c-tools python3-gpiozero python3-lgpio \
  rpicam-apps python3-picamera2 \
  git tmux curl jq tree usbutils pciutils \
  python3-venv python3-pip python3-psutil

echo
echo "== board =="
tr -d '\0' < /proc/device-tree/model; echo
grep -m1 Revision /proc/cpuinfo
echo "== cpu =="
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|max MHz|min MHz|Flags" | cut -c1-200
echo "== memory / storage =="
free -h | head -2
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT | head -8
echo "== firmware view (vcgencmd) =="
for c in "measure_temp" "measure_clock arm" "measure_volts core" "get_throttled" "get_config arm_freq" "get_config over_voltage" "get_mem arm" "get_mem gpu"; do
  printf '%-24s %s\n' "$c" "$(vcgencmd $c 2>/dev/null)"
done
echo "== governor =="
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies 2>/dev/null
echo "== os =="
grep PRETTY /etc/os-release; uname -m; python3 --version

cat <<'EOF'

Useful commands from here:
  btop                                  live CPU per core, memory, temperature
  watch -n1 'vcgencmd measure_temp; vcgencmd measure_clock arm; vcgencmd get_throttled'
  sudo perf stat -e cycles,instructions,cache-misses python run_eval.py ...     hardware counters for one run
  stress-ng --cpu 4 --timeout 60s --metrics-brief                               heat and throttle test
  sysbench cpu --threads=4 run                                                  raw CPU score
  i2cdetect -y 1                                                                devices on the I2C pins
  pinout                                                                        GPIO header map
  rpicam-hello --list-cameras                                                   attached cameras
get_throttled bits: 0x1 under-voltage, 0x2 frequency capped, 0x4 throttled, 0x8 soft temp limit
                    (0x10000-0x80000 = the same has happened since boot)
EOF
