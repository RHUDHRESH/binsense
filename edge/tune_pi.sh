#!/usr/bin/env bash
# Safe, reversible Pi 4 tuning (no config.txt edits, nothing survives a reboot).
set -uo pipefail
echo "before: governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) clock=$(vcgencmd measure_clock arm) temp=$(vcgencmd measure_temp) throttled=$(vcgencmd get_throttled)"
# 1. Hold the CPU at full clock instead of letting it ramp up per request.
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
# 2. Quiet the box: stop services that wake up and steal a core mid-benchmark.
for s in bluetooth cups cups-browsed ModemManager triggerhappy avahi-daemon; do sudo systemctl stop "$s" 2>/dev/null; done
# 3. Drop file caches so memory numbers are clean.
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
echo "after:  governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) clock=$(vcgencmd measure_clock arm) temp=$(vcgencmd measure_temp)"
free -m | head -2
