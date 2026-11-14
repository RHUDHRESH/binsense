#!/usr/bin/env python3
"""Fetch TrashNet (resized, 43 MB, MIT licence) and lay out a balanced sample as data/trashnet/<label>/*.jpg.

Every N-th file goes to data/trashnet (test); the rest go to data/trashnet_pool (calibration / support).
"""
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 40
z = zipfile.ZipFile(hf_hub_download("garythung/trashnet", "dataset-resized.zip", repo_type="dataset"))
by = defaultdict(list)
for n in sorted(z.namelist()):
    p = Path(n)
    if p.suffix.lower() == ".jpg" and not p.name.startswith(".") and "__MACOSX" not in n:
        by[p.parent.name].append(n)
for lab, names in by.items():
    step = max(1, len(names) // per_class)
    test = set(names[::step][:per_class])
    for n in names:
        d = Path("data/trashnet" if n in test else "data/trashnet_pool") / lab
        d.mkdir(parents=True, exist_ok=True)
        (d / Path(n).name).write_bytes(z.read(n))
    print(lab, len(names), "->", len(test), "test")
