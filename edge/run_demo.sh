#!/usr/bin/env bash
# Runs the terminal demo with the bundle's environment.
cd "$(dirname "$0")"; source ~/binsense-venv/bin/activate
exec "$VIRTUAL_ENV/bin/python3" demo.py "$@"
