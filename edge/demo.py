#!/usr/bin/env python3
"""BinSense live terminal demo: proves the board is offline, then classifies photos on-device.

  python demo.py                      one pass over images/
  python demo.py --loop 3 --delay 1   three passes, one second per image
  python demo.py --forever            until Ctrl+C
  python demo.py --watch inbox        classify photos as they are dropped into inbox/

Needs only executorch, torch, numpy, pillow, psutil. Writes demo_results.csv and demo_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COLOR = {"g": "32", "y": "33", "r": "31", "c": "36", "b": "1", "d": "2"}
USE_COLOR = True


def col(text, k: str) -> str:
    return f"\033[{COLOR[k]}m{text}\033[0m" if USE_COLOR else str(text)


def head(title: str) -> None:
    print("\n" + col(f"== {title} " + "=" * max(4, 66 - len(title)), "b"))


def run(*cmd: str) -> str | None:
    """Output of a system command, or None when it is missing or fails."""
    try:
        r = subprocess.run(list(cmd), capture_output=True, text=True, timeout=3)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001  (command not present on this machine)
        return None


def find(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() or q.exists() else HERE / q


def sha12(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def pct(v: list[float], q: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def peak_mb() -> float | None:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KB
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1e6
        except Exception:  # noqa: BLE001
            return None


def tcp_ok(host: str, port: int) -> bool:
    try:
        socket.create_connection((host, port), timeout=2).close()
        return True
    except OSError:
        return False


def dns_ok(name: str) -> bool:
    out: list[bool] = []

    def look():
        try:
            socket.getaddrinfo(name, 443)
            out.append(True)
        except OSError:
            pass
    t = threading.Thread(target=look, daemon=True)  # a dead resolver can hang far longer than 2 s
    t.start()
    t.join(3)
    return bool(out)


def offline_block() -> tuple[str, dict]:
    import psutil

    head("1. OFFLINE PROOF")
    try:
        board = Path("/proc/device-tree/model").read_text(errors="ignore").strip("\x00\n ")
    except OSError:
        board = f"{platform.machine()} (not a Raspberry Pi)"
    print(f"  host      {socket.gethostname()}\n  board     {board}")
    print(f"  os        {platform.system()} {platform.release()} / Python {platform.python_version()}")
    print(f"  cpus      {os.cpu_count()}")
    print("  network interfaces:")
    try:
        up = {k: v.isup for k, v in psutil.net_if_stats().items()}
        for name, addrs in psutil.net_if_addrs().items():
            ips = [a.address for a in addrs if a.family == socket.AF_INET] or ["no IPv4"]
            print(f"    {name[:26]:<26} {'up' if up.get(name) else 'down':<5} {', '.join(ips)}")
    except Exception as e:  # noqa: BLE001
        print(f"    n/a ({type(e).__name__})")
    route = "n/a (no `ip` command here)"
    if shutil.which("ip"):
        r = run("ip", "route", "show", "default")
        route = "n/a" if r is None else (r.splitlines()[0] if r else "none")
    print(f"  default route   {route}")
    checks = {"TCP 1.1.1.1:443": tcp_ok("1.1.1.1", 443), "TCP 8.8.8.8:53": tcp_ok("8.8.8.8", 53),
              "DNS github.com": dns_ok("github.com")}
    for k, ok in checks.items():
        print(f"  {k:<16}{col('REACHABLE', 'y') if ok else col('unreachable', 'g')}")
    online = any(checks.values())
    verdict = "WARNING: internet is reachable" if online else "OFFLINE: no route to the internet"
    print("  " + col(verdict, "y" if online else "g"))
    return verdict, {"board": board, "default_route": route, "online": online,
                     "checks": {k: ("reachable" if v else "unreachable") for k, v in checks.items()}}


def model_block(args):
    import logging
    import warnings
    from importlib import metadata

    warnings.simplefilter("ignore")  # third-party import chatter would bury the demo output
    logging.disable(logging.WARNING)
    import numpy as np
    import torch
    from binsense import BinSense, Cascade

    head("2. MODELS (loaded from local files)")
    full = BinSense(find(args.pte), find(args.pack), threads=args.threads or None)
    parts, engine = [("careful", full)], full
    if args.gate_pte and find(args.gate_pte).exists() and find(args.gate_pack).exists():
        gate = BinSense(find(args.gate_pte), find(args.gate_pack))
        engine = Cascade(gate, full, args.threshold)
        parts.insert(0, ("fast gate", gate))
    else:
        print(col("  gate model not found: running the careful model alone", "y"))
    info = []
    for role, m in parts:
        mb, digest = m.pte.stat().st_size / 1e6, sha12(m.pte)
        m.embed(np.zeros((1, 3, m.res, m.res), np.float32))  # warm-up so the first photo is not the slow one
        print(f"  {role:<10}{m.pte.name:<34}{mb:6.1f} MB  sha256 {digest}  load {m.load_ms:.0f} ms  input {m.res}px")
        print(col(f"            {str(m.meta.get('model', '?'))[:90]}", "d"))
        info.append({"role": role, "model": m.meta.get("model"), "pte": m.pte.name, "size_mb": round(mb, 2),
                     "sha256_12": digest, "load_ms": round(m.load_ms, 1), "resolution": m.res})
    try:
        et = metadata.version("executorch")
    except Exception:  # noqa: BLE001
        et = "?"
    print(f"  threads {full.threads}   ExecuTorch {et}   torch {torch.__version__}   labels: {', '.join(full.labels)}")
    print(f"  fast answer needs confidence >= {args.threshold}; below {args.review} goes to REVIEW")
    return engine, {"models": info, "threads": full.threads, "executorch": et, "torch": torch.__version__}


def hardware_line() -> str:
    import psutil

    def vc(*a: str) -> str:
        out = run("vcgencmd", *a)
        return out.split("=")[-1] if out else "n/a"
    clock = vc("measure_clock", "arm")
    clock = f"{int(clock) / 1e6:.0f} MHz" if clock.isdigit() else clock
    cores = " ".join(f"{c:.0f}%" for c in psutil.cpu_percent(percpu=True))
    rss = psutil.Process().memory_info().rss / 1e6
    return col(f"  [hw] temp {vc('measure_temp')}  arm {clock}  throttled {vc('get_throttled')}  "
               f"cpu {cores}  rss {rss:.0f} MB", "c")


def folder_items(folder: Path, args):
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXT)
    if not files:
        print(col(f"  no images found in {folder}", "r"))
    n = 0
    while files and (args.forever or n < args.loop):
        n += 1
        yield from files


def watch_items(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    seen = {p for p in folder.iterdir()}
    sizes: dict[Path, int] = {}
    print(f"  watching {folder} for new photos (Ctrl+C to stop)")
    while True:
        for p in sorted(folder.iterdir()):
            if p in seen or not p.is_file() or p.suffix.lower() not in EXT:
                continue
            size = p.stat().st_size
            if size and sizes.get(p) == size:  # same size on two polls: the copy has finished
                seen.add(p)
                yield p
            sizes[p] = size
        time.sleep(0.3)


def main() -> None:
    global USE_COLOR
    ap = argparse.ArgumentParser(description="BinSense offline demo")
    ap.add_argument("--pte", default="models/own_mobileclip2_s0_256_fp32.pte")
    ap.add_argument("--pack", default="packs/own_mobileclip2_s0_256.npz")
    ap.add_argument("--gate-pte", default="models/stu_mobilenetv2_100_96_fp32.pte")
    ap.add_argument("--gate-pack", default="packs/stu_mobilenetv2_100_96.npz")
    ap.add_argument("--threshold", type=float, default=0.9, help="gate confidence needed for a fast answer")
    ap.add_argument("--review", type=float, default=0.7, help="below this confidence the item goes to review")
    ap.add_argument("--images", default="images")
    ap.add_argument("--threads", type=int, default=0, help="0 = ExecuTorch default")
    ap.add_argument("--loop", type=int, default=1)
    ap.add_argument("--forever", action="store_true")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--watch", default=None, metavar="FOLDER")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    USE_COLOR = not args.no_color and sys.stdout.isatty()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":
        os.system("")  # switches the Windows console to ANSI mode

    def stop(*_):
        raise KeyboardInterrupt
    for s in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, s):
            signal.signal(getattr(signal, s), stop)

    print(col("BinSense - on-device waste sorting demo", "b"))
    verdict, net = offline_block()
    try:
        engine, models = model_block(args)
    except KeyboardInterrupt:
        sys.exit("\nstopped before the models were loaded")
    except Exception as e:  # noqa: BLE001
        sys.exit(col(f"\ncould not load the models: {type(e).__name__}: {e}", "r"))

    import psutil
    psutil.cpu_percent(percpu=True)  # first call only sets the baseline
    head("3. CLASSIFICATION")
    print(col(f"  {'#':>4}  {'file':<24}{'predicted':<11}{'conf':>6}  {'path':<8}{'infer':>9}  tags", "d"))
    source = watch_items(find(args.watch)) if args.watch else folder_items(find(args.images), args)
    rows, inf, busy, errors, t_start = [], [], 0.0, 0, time.time()
    try:
        for i, path in enumerate(source, 1):
            truth = path.parent.name if path.parent.name in engine.labels else None
            t0 = time.perf_counter()
            try:
                d = engine.decide(path)
            except Exception as e:  # noqa: BLE001  one bad file must never stop the demo
                errors += 1
                print(f"  {i:>4}  {path.name[:23]:<24}" + col(f"SKIPPED unreadable image ({type(e).__name__})", "r"))
                continue
            busy += time.perf_counter() - t0
            c, way = d["choice"], d.get("path", "careful")
            review = c["confidence"] < args.review
            correct = None if truth is None else c["answer"] == truth
            tags = " + ".join(d["multi"]["labels"] + d["multi"]["tags"])
            inf.append(d["ms"]["inference"])
            rows.append({"n": i, "image": path.name, "label": truth or "", "predicted": c["answer"],
                         "confidence": round(c["confidence"], 4), "correct": "" if correct is None else correct,
                         "review": review, "path": way, "preprocess_ms": round(d["ms"]["preprocess"], 2),
                         "inference_ms": round(d["ms"]["inference"], 2), "multi_select": tags})
            mark = "" if correct is None else (col("✓", "g") if correct else col(f"✗ (is {truth})", "r"))
            print(f"  {i:>4}  {path.name[:23]:<24}{c['answer']:<11}{c['confidence'] * 100:5.1f}%  "
                  f"{col(f'{way:<8}', 'g' if way == 'fast' else 'c')}{d['ms']['inference']:6.1f} ms  {tags}  "
                  f"{mark}{col('  REVIEW', 'y') if review else ''}")
            if len(rows) % 10 == 0:
                print(hardware_line())
            if args.delay > 0:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print(col("\n  stopped by Ctrl+C", "y"))
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl+C must not cut the summary short
        summary(args, rows, inf, busy, errors, t_start, verdict, net, models)


def summary(args, rows, inf, busy, errors, t_start, verdict, net, models) -> None:
    if not rows or len(rows) % 10:
        print(hardware_line())
    head("5. SUMMARY")
    known = [r for r in rows if r["correct"] != ""]
    ok = sum(1 for r in known if r["correct"])
    fast = sum(1 for r in rows if r["path"] == "fast")
    rev = sum(1 for r in rows if r["review"])
    n, peak = len(rows), peak_mb()
    s = {"images": n, "unreadable": errors, "scored": len(known), "correct": ok,
         "accuracy": round(ok / len(known), 4) if known else None,
         "fast_share": round(fast / n, 4) if n else None, "review": rev,
         "inference_ms": ({"median": round(pct(inf, .5), 2), "p90": round(pct(inf, .9), 2),
                           "min": round(min(inf), 2), "max": round(max(inf), 2)} if inf else None),
         "images_per_second": round(n / busy, 2) if busy else None,
         "peak_memory_mb": round(peak, 1) if peak else None, "wall_seconds": round(time.time() - t_start, 1),
         "gate_threshold": args.threshold, "review_threshold": args.review, "offline_verdict": verdict,
         "network": net, **models, "machine": platform.machine(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"  images processed   {n}" + (f"   ({errors} unreadable, skipped)" if errors else ""))
    if known:
        print(f"  correct            {ok} / {len(known)}   accuracy {col(f'{100 * ok / len(known):.1f}%', 'b')}")
    else:
        print("  correct            n/a (no labels known)")
    if n:
        m = s["inference_ms"]
        print(f"  fast model share   {fast} / {n}  ({100 * fast / n:.0f}%)")
        print(f"  sent to review     {rev}")
        print(f"  inference ms       median {m['median']:.1f}   p90 {m['p90']:.1f}   min {m['min']:.1f}   max {m['max']:.1f}")
        print(f"  images per second  {s['images_per_second']}")
    print(f"  peak memory        {s['peak_memory_mb'] if peak else 'n/a'} MB")
    print("  " + col(verdict, "g" if verdict.startswith("OFFLINE") else "y"))
    try:
        with (HERE / "demo_results.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["n"])
            w.writeheader()
            w.writerows(rows)
        (HERE / "demo_summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
        print(col(f"  wrote demo_results.csv and demo_summary.json in {HERE}", "d"))
    except OSError as e:
        print(col(f"  could not write result files: {e}", "y"))


if __name__ == "__main__":
    main()
