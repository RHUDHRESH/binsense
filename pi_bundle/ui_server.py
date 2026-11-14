#!/usr/bin/env python3
"""BinSense local web UI. No internet, no extra packages: open http://localhost:8000

  python ui_server.py            # then open the address it prints
Pick a dataset or drop photos in, type your own labels, move the sliders, click Run all.
"""
from __future__ import annotations

import csv
import json
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from binsense import BinSense, softmax  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODELS, TEXT = {}, {}


def stem(p: str) -> str:
    s = Path(p).stem
    for suf in ("_fp32", "_int8", "_portable"):
        s = s.removesuffix(suf)
    return s


def pack_meta(p: Path) -> dict:
    z = np.load(p, allow_pickle=False)
    m = json.loads(str(z["meta"])); m["text"] = "label_emb" in z.files
    return m


def catalog() -> dict:
    ptes = sorted(p.name for p in (BASE / "models").glob("*.pte") if "portable" not in p.name)
    packs = {p.name: pack_meta(p) for p in sorted((BASE / "packs").glob("*.npz"))}
    out = []
    for pte in ptes:
        res = next((int(t) for t in Path(pte).stem.split("_") if t.isdigit() and len(t) in (2, 3)), 0)
        ok = [n for n, m in packs.items()
              if Path(n).stem == stem(pte) or (pte.startswith("mobileclip2-s0") and m["text"] and m["preprocess"]["size"] == res)]
        if ok:
            out.append({"pte": pte, "mb": round((BASE / "models" / pte).stat().st_size / 1e6, 1), "res": res, "packs": ok,
                        "kind": "text labels (any labels, no training)" if pte.startswith("mobileclip2-s0") else "trained on TrashNet (6 fixed labels)"})
    return {"models": out, "packs": {n: {"labels": m["labels"], "text": m["text"], "size": m["preprocess"]["size"]} for n, m in packs.items()}}


def datasets() -> dict:
    out = {}
    for d in sorted(p for p in BASE.iterdir() if p.is_dir() and p.name not in {"models", "packs", "results", "__pycache__", "sorted", "review", "rejected", "inbox"}):
        files = sorted((p for p in d.rglob("*") if p.suffix.lower() in EXT), key=lambda p: (len(p.stem), p.stem))
        if not files:
            continue
        truth = {}
        if (d / "labels.csv").exists():
            truth = {r[0]: r[1] for r in csv.reader((d / "labels.csv").open(encoding="utf-8")) if len(r) > 1 and r[0] != "file"}
        out[d.name] = [{"path": str(f.relative_to(BASE)).replace("\\", "/"), "name": f.name,
                        "truth": truth.get(f.name, f.parent.name if f.parent != d else "")} for f in files]
    return out


def get_model(pte: str, pack: str) -> BinSense:
    k = (pte, pack, (BASE / "packs" / pack).stat().st_mtime)
    if k not in MODELS:
        MODELS[k] = BinSense(BASE / "models" / pte, BASE / "packs" / pack)
    return MODELS[k]


def run_one(m: BinSense, path: Path, sharp: float) -> dict:
    t0 = time.perf_counter(); x = m.preprocess(path); t1 = time.perf_counter(); e = m.embed(x); t2 = time.perf_counter()
    if m.label_emb is not None:
        raw = m.label_emb @ e; p = softmax(m.scale * sharp * raw); how = "similarity between the photo and each label's text"
        if m.head is not None:
            p = (1 - m.head_weight) * p + m.head_weight * softmax(m.head[0] @ e + m.head[1])
    else:
        raw = m.head[0] @ e + m.head[1]; p = softmax(raw * sharp); how = "score from the trained decision table"
    return {"file": m.pte.name, "mb": round(m.pte.stat().st_size / 1e6, 1), "model": m.meta["model"], "input": list(x.shape),
            "embedding": int(len(e)), "pre_ms": round((t1 - t0) * 1000, 1), "infer_ms": round((t2 - t1) * 1000, 1), "how": how,
            "labels": m.labels, "raw": [round(float(v), 3) for v in raw], "probs": [round(float(v), 4) for v in p],
            "nouls": {k: round(m.noul_from(e, k), 3) for k in m.noul}}


def classify(q: dict) -> dict:
    path = (BASE / q["image"]).resolve()
    if BASE not in path.parents or not path.exists():
        raise ValueError("image not found")
    main = get_model(q["pte"], q["pack"]); sharp = float(q.get("sharp", 1)); steps = []; used = "single model"
    final = None
    if q.get("gate_pte") and q.get("gate_pack"):
        g = get_model(q["gate_pte"], q["gate_pack"])
        if g.labels == main.labels:
            s = run_one(g, path, sharp); s["role"] = "Fast model"; steps.append(s)
            if max(s["probs"]) >= float(q.get("threshold", 0.9)):
                final, used = s, "fast model"
    if final is None:
        s = run_one(main, path, sharp); s["role"] = "Careful model" if steps else "Model"; steps.append(s); final = s
        used = "careful model" if len(steps) > 1 else "single model"
    probs = dict(zip(final["labels"], final["probs"])); top = max(probs, key=probs.get); conf = probs[top]
    multi = main.multi_from({"answer": top, "confidence": conf, "probabilities": probs}, final["nouls"],
                            float(q.get("label_min", 0.25)), float(q.get("tag_min", 0.5)))
    review = conf < float(q.get("review", 0.7))
    return {"image": q["image"], "steps": steps, "answer": top, "confidence": round(conf, 4), "review": review, "used": used,
            "multi": multi["labels"] + multi["tags"], "tags": multi["tag_probabilities"],
            "infer_ms": round(sum(s["infer_ms"] for s in steps), 1), "pre_ms": round(sum(s["pre_ms"] for s in steps), 1)}


def run_all(q: dict) -> dict:
    rows, t0 = [], time.perf_counter()
    for it in datasets().get(q["dataset"], []):
        try:
            r = classify({**q, "image": it["path"]})
            rows.append({"image": it["name"], "path": it["path"], "label": it["truth"], "predicted": "REVIEW" if r["review"] else r["answer"],
                         "best_guess": r["answer"], "confidence": r["confidence"], "multi": " + ".join(r["multi"]), "used": r["used"],
                         "infer_ms": r["infer_ms"], "correct": (r["answer"] == it["truth"]) if it["truth"] else None})
        except Exception as e:  # noqa: BLE001
            rows.append({"image": it["name"], "path": it["path"], "label": it["truth"], "predicted": f"ERROR {type(e).__name__}", "best_guess": "",
                         "confidence": 0, "multi": "", "used": "", "infer_ms": 0, "correct": None})
    known = [r for r in rows if r["correct"] is not None]; ms = sorted(r["infer_ms"] for r in rows) or [0]
    summ = {"images": len(rows), "scored": len(known), "correct": sum(r["correct"] for r in known),
            "accuracy": round(100 * sum(r["correct"] for r in known) / max(1, len(known)), 1), "review": sum(r["predicted"] == "REVIEW" for r in rows),
            "fast_share": round(100 * sum(r["used"] == "fast model" for r in rows) / max(1, len(rows))), "ms_median": ms[len(ms) // 2], "ms_max": ms[-1],
            "seconds": round(time.perf_counter() - t0, 2), "model": q["pte"], "model_mb": round((BASE / "models" / q["pte"]).stat().st_size / 1e6, 1),
            "labels": q["pack"], "memory_mb": memory_mb()}
    out = BASE / "results"; out.mkdir(exist_ok=True)
    with (out / f"ui_{q['dataset']}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "path"] if rows else ["image"], extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    (out / f"ui_{q['dataset']}.json").write_text(json.dumps(summ, indent=1))
    return {"rows": rows, "summary": summ, "saved": f"results/ui_{q['dataset']}.csv"}


def memory_mb() -> float:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6)
    except Exception:  # noqa: BLE001
        return 0


def system() -> dict:
    def vc(*a):
        try:
            return subprocess.run(["vcgencmd", *a], capture_output=True, text=True, timeout=2).stdout.strip() or "n/a"
        except Exception:  # noqa: BLE001
            return "n/a"
    online = False
    for host, port in (("1.1.1.1", 443), ("8.8.8.8", 53)):
        try:
            socket.create_connection((host, port), timeout=1.5).close(); online = True
        except OSError:
            pass
    board = "not a Raspberry Pi"
    try:
        board = Path("/proc/device-tree/model").read_text().strip("\0")
    except OSError:
        pass
    return {"online": online, "board": board, "host": socket.gethostname(), "temp": vc("measure_temp"), "clock": vc("measure_clock", "arm"),
            "throttled": vc("get_throttled"), "memory_mb": memory_mb(), "can_edit_labels": can_text()}


def can_text() -> bool:
    try:
        import open_clip  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def build_pack(q: dict) -> dict:
    """Make a label pack from plain text: one label per line, 'name: phrase, phrase, phrase'. Needs the text model (laptop)."""
    import open_clip, torch
    labels = {}
    for line in q["text"].splitlines():
        if ":" in line:
            n, ph = line.split(":", 1); labels[n.strip()] = [x.strip() for x in ph.split(",") if x.strip()] or [n.strip()]
        elif line.strip():
            labels[line.strip()] = [line.strip()]
    if len(labels) < 2:
        raise ValueError("need at least two labels, one per line")
    if "m" not in TEXT:
        clip, _, _ = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained="dfndr2b"); TEXT["m"] = clip.eval(); TEXT["tok"] = open_clip.get_tokenizer("MobileCLIP2-S0")
    T = ["a photo of {}.", "a close-up photo of {}.", "a photo of {} thrown away as waste.", "{} in a bin.", "a picture of discarded {}."]

    @torch.inference_mode()
    def emb(phrases):
        f = TEXT["m"].encode_text(TEXT["tok"]([t.format(p) for p in phrases for t in T])); f = f / f.norm(dim=-1, keepdim=True); m = f.mean(0)
        return (m / m.norm()).numpy().astype(np.float32)
    arr = np.stack([emb(v) for v in labels.values()]); made = []
    name = "".join(c for c in q.get("name", "custom") if c.isalnum() or c in "-_") or "custom"
    for res in (256, 192, 160):
        ref = next((p for p in (BASE / "packs").glob(f"*_{res}.npz") if pack_meta(p)["text"]), None)
        if ref is None:
            continue
        meta = {k: v for k, v in pack_meta(ref).items() if k in ("model", "pretrained", "preprocess", "logit_scale")}
        meta.update(labels=list(labels), noul={}, score={}, phrases=labels)
        np.savez(BASE / "packs" / f"{name}_{res}.npz", meta=np.array(json.dumps(meta)), label_emb=arr); made.append(f"{name}_{res}.npz")
    return {"made": made, "labels": list(labels)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                return self.send((BASE / "ui.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self.send({**catalog(), "datasets": datasets(), "system": system()})
            if u.path == "/api/system":
                return self.send(system())
            if u.path == "/img":
                p = (BASE / unquote(parse_qs(u.query)["p"][0])).resolve()
                if BASE in p.parents and p.suffix.lower() in EXT and p.exists():
                    return self.send(p.read_bytes(), "image/" + p.suffix.lower().strip(".").replace("jpg", "jpeg"))
            self.send({"error": "not found"}, code=404)
        except Exception as e:  # noqa: BLE001
            self.send({"error": f"{type(e).__name__}: {e}"}, code=500)

    def do_POST(self):
        u = urlparse(self.path); n = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(n)
        try:
            if u.path == "/api/upload":
                qs = parse_qs(u.query); ds = "".join(c for c in qs.get("dataset", ["my_photos"])[0] if c.isalnum() or c in "-_") or "my_photos"
                name = Path(unquote(qs["name"][0])).name
                if Path(name).suffix.lower() not in EXT:
                    raise ValueError("not an image file")
                (BASE / ds).mkdir(exist_ok=True); (BASE / ds / name).write_bytes(body); return self.send({"saved": f"{ds}/{name}", "dataset": ds})
            q = json.loads(body or b"{}")
            if u.path == "/api/classify":
                return self.send(classify(q))
            if u.path == "/api/run_all":
                return self.send(run_all(q))
            if u.path == "/api/build_pack":
                return self.send(build_pack(q))
            if u.path == "/api/truth":
                d = BASE / "".join(c for c in q["dataset"] if c.isalnum() or c in "-_"); f = d / "labels.csv"
                t = {r[0]: r[1] for r in csv.reader(f.open(encoding="utf-8")) if len(r) > 1 and r[0] != "file"} if f.exists() else {}
                t[Path(q["name"]).name] = q["label"]
                with f.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh); w.writerow(["file", "label"]); w.writerows(sorted(t.items()))
                return self.send({"ok": True})
            self.send({"error": "not found"}, code=404)
        except Exception as e:  # noqa: BLE001
            self.send({"error": f"{type(e).__name__}: {e}"}, code=400)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    print(f"BinSense UI running. Open  http://localhost:{port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
