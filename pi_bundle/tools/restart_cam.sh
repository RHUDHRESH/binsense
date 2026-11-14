#!/usr/bin/env bash
# Restart the Pi's web/camera server.  bash tools/restart_cam.sh [camera-only]
cd ~/binsense
for p in $(pgrep -f "tools/keep_ui.sh") $(pgrep -f "ui_server.py 8000"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
sleep 1
[ "${1:-}" = "camera-only" ] && export BINSENSE_CAMERA_ONLY=1
setsid nohup bash tools/keep_ui.sh > /dev/null 2>&1 < /dev/null &
sleep 8
echo "listening: $(ss -ltn | grep -c ':8000 ')"
curl -s -o /dev/null -m 15 -w "camera frame: %{http_code} %{size_download} bytes\n" http://localhost:8000/cam.jpg
