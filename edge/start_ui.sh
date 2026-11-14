#!/usr/bin/env bash
# Start the BinSense web UI on the Pi, then open http://<pi-address>:8000 from the laptop.
cd "$(dirname "$0")"; source ~/binsense-venv/bin/activate
exec "$VIRTUAL_ENV/bin/python3" ui_server.py 8000
