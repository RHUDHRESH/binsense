#!/usr/bin/env python3
"""Attach label groups (multi-select tags) to a decision pack.
   add_groups.py pack.npz '{"recyclable": ["cardboard","glass","metal","paper","plastic"], "landfill": ["trash"]}'
"""
import json, sys
import numpy as np
z = dict(np.load(sys.argv[1], allow_pickle=False)); meta = json.loads(str(z["meta"]))
groups = json.loads(sys.argv[2])
unknown = {m for ms in groups.values() for m in ms} - set(meta["labels"])
if unknown: sys.exit(f"labels not in pack: {sorted(unknown)}")
meta["groups"] = groups; z["meta"] = np.array(json.dumps(meta)); np.savez(sys.argv[1], **z)
print("groups added:", list(groups))
