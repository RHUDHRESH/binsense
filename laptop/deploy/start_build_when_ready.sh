#!/usr/bin/env bash
# Wait until the Pi has git, cmake and Python headers (after the sudo apt step), then start the ExecuTorch build there.
H='pi@fe80::da3a:ddff:fec0:7287%9'; SSH="ssh -i $HOME/.ssh/binsense_pi -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes -o ConnectTimeout=8"
tar -C pi_bundle -cf - build_executorch.sh | $SSH "$H" 'tar -C ~/binsense -xf - && sed -i "s/\r$//" ~/binsense/build_executorch.sh'
for i in $(seq 1 240); do
  if $SSH "$H" 'command -v git >/dev/null && command -v cmake >/dev/null && ls /usr/include/python3.13/Python.h >/dev/null 2>&1' 2>/dev/null; then
    $SSH "$H" 'nohup bash ~/binsense/build_executorch.sh >/dev/null 2>&1 & echo started'
    echo "[$(date +%H:%M:%S)] build started on the Pi"; exit 0
  fi
  sleep 10
done
echo "build tools never appeared"; exit 1
