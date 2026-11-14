#!/usr/bin/env bash
# Keeps the Pi camera/web server running; restarts it if it stops.
cd ~/binsense
while true; do ~/binsense-venv/bin/python3 ui_server.py 8000 >> ~/ui.log 2>&1; sleep 2; done
