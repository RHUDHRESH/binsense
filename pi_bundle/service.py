#!/usr/bin/env python3
"""BinSense sorting service: watches a folder, decides, logs, and never guesses silently.

  python service.py --config config.json          (runs until stopped; safe to start at boot)

Deployment behaviours (each one checkable from the files it writes):
  * abstain   - below `min_confidence` the item goes to review/ for a person, it is not guessed
  * audit log - every decision is appended to audit.jsonl with model hash, confidence and timing
  * bad input - unreadable or oversized files go to rejected/ with the reason; the service keeps running
  * health    - health.json is rewritten every cycle (uptime, counts, latency, temperature, throttling)
  * offline   - no network calls anywhere; labels and thresholds come from files
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from binsense import Cascade, BinSense  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULTS = {"pte": "models/own_mobileclip2_s0_256_fp32.pte", "pack": "packs/own_mobileclip2_s0_256.npz",
            "gate_pte": "models/own_mobilenet_v2_224_fp32.pte", "gate_pack": "packs/own_mobilenet_v2_224.npz",
            "gate_threshold": 0.9, "min_confidence": 0.7, "threads": 4, "max_bytes": 20_000_000,
            "inbox": "inbox", "sorted": "sorted", "review": "review", "rejected": "rejected",
            "audit": "audit.jsonl", "health": "health.json", "poll_seconds": 0.2}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def vc(*a: str) -> str | None:
    try:
        return subprocess.run(["vcgencmd", *a], capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--once", action="store_true", help="process what is in the inbox, then exit")
    args = ap.parse_args()
    cfg = {**DEFAULTS, **(json.loads(args.config.read_text()) if args.config else {})}
    base = args.config.parent if args.config else Path(__file__).parent
    P = lambda k: (base / cfg[k])  # noqa: E731
    for k in ("inbox", "sorted", "review", "rejected"):
        P(k).mkdir(parents=True, exist_ok=True)

    model = BinSense(P("pte"), P("pack"), threads=cfg["threads"])
    ident = {"model": model.meta["model"], "pte": P("pte").name, "pte_sha256": sha(P("pte")), "pack_sha256": sha(P("pack"))}
    if cfg.get("gate_pte") and P("gate_pte").exists():
        model = Cascade(BinSense(P("gate_pte"), P("gate_pack")), model, cfg["gate_threshold"])
        ident["gate_pte_sha256"] = sha(P("gate_pte"))
    stats = {"started": time.time(), "decided": 0, "review": 0, "rejected": 0, "last_ms": None}
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    print("BinSense service ready:", json.dumps(ident))

    def move(src: Path, folder: Path) -> str:
        folder.mkdir(parents=True, exist_ok=True)
        dst = folder / f"{int(time.time() * 1000)}_{src.name}"
        shutil.move(str(src), dst)
        return str(dst.relative_to(base))

    with P("audit").open("a", buffering=1) as audit:
        while not stop["now"]:
            files = sorted(p for p in P("inbox").iterdir() if p.is_file())
            for f in files:
                rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "file": f.name, **ident}
                try:
                    if f.suffix.lower() not in EXT:
                        raise ValueError(f"unsupported type {f.suffix}")
                    if f.stat().st_size > cfg["max_bytes"]:
                        raise ValueError("file too large")
                    d = model.decide(f)
                    c = d["choice"]
                    ok = c["confidence"] >= cfg["min_confidence"]
                    rec.update(decision=c["answer"] if ok else "REVIEW", best_guess=c["answer"],
                               confidence=round(c["confidence"], 4), multi=d["multi"]["labels"] + d["multi"]["tags"],
                               path=d.get("path", "single"), ms={k: round(v, 2) for k, v in d["ms"].items()})
                    rec["moved_to"] = move(f, P("sorted") / c["answer"] if ok else P("review"))
                    stats["decided" if ok else "review"] += 1
                    stats["last_ms"] = rec["ms"]["inference"]
                except Exception as e:  # noqa: BLE001  one bad file must never stop the line
                    rec.update(decision="REJECTED", error=f"{type(e).__name__}: {e}"[:200])
                    try:
                        rec["moved_to"] = move(f, P("rejected"))
                    except Exception:  # noqa: BLE001
                        pass
                    stats["rejected"] += 1
                audit.write(json.dumps(rec) + "\n")
                print(rec["ts"], f.name, "->", rec["decision"], rec.get("confidence", ""), rec.get("ms", {}).get("inference", ""))
            P("health").write_text(json.dumps({**stats, "uptime_s": round(time.time() - stats["started"]),
                                               "temp": vc("measure_temp"), "throttled": vc("get_throttled"), **ident}, indent=1))
            if args.once:
                break
            time.sleep(cfg["poll_seconds"])
    print("stopped cleanly")


if __name__ == "__main__":
    main()
