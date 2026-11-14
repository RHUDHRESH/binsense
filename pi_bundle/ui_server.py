#!/usr/bin/env python3
"""BinSense local web UI. No internet, no extra packages: open http://localhost:8000

  python ui_server.py            # then open the address it prints
Pick a dataset or drop photos in, type your own labels, move the sliders, click Run all.
"""
from __future__ import annotations

import csv
import json
import socket
import os
import subprocess
import sys
import threading
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


SKIP = {"models", "packs", "results", "__pycache__", "sorted", "review", "rejected", "inbox", "support5", "support20", "organiser_raw", "trashnet_pool"}
EXTRA = BASE.parent / "data"            # the project's downloaded photo sets, when running from the laptop
MAX_PER_SET = 300


def allowed(path: Path) -> bool:
    path = path.resolve()
    return (BASE in path.parents or EXTRA.resolve() in path.parents) and path.suffix.lower() in EXT and path.exists()


def datasets() -> dict:
    out = {}
    roots = [d for d in sorted(BASE.iterdir()) if d.is_dir() and d.name not in SKIP]
    if EXTRA.exists():
        roots += [d for d in sorted(EXTRA.iterdir()) if d.is_dir() and d.name not in SKIP]
    for d in roots:
        files = sorted((f for f in d.rglob("*") if f.suffix.lower() in EXT), key=lambda f: (f.parent.name, len(f.stem), f.stem))
        if not files:
            continue
        if len(files) > MAX_PER_SET:      # keep a spread across sub-folders, not just the first folder
            files = files[:: max(1, len(files) // MAX_PER_SET)][:MAX_PER_SET]
        truth = {}
        if (d / "labels.csv").exists():
            truth = {r[0]: r[1] for r in csv.reader((d / "labels.csv").open(encoding="utf-8")) if len(r) > 1 and r[0] != "file"}
        name = d.name if d.name not in out else d.name + "_2"
        out[name] = [{"path": (str(f.relative_to(BASE)) if BASE in f.parents else "../data/" + str(f.relative_to(EXTRA))).replace("\\", "/"),
                      "name": f.name, "truth": truth.get(f.name, f.parent.name if f.parent != d else "")} for f in files]
    return out


def get_model(pte: str, pack: str) -> BinSense:
    if os.environ.get("BINSENSE_CAMERA_ONLY"):   # set on a device whose ExecuTorch runtime is not ready yet
        raise ValueError("the model runtime is still being built on this device; only the camera is available")
    k = (pte, pack, (BASE / "packs" / pack).stat().st_mtime)
    if k not in MODELS:
        MODELS[k] = BinSense(BASE / "models" / pte, BASE / "packs" / pack)
    return MODELS[k]


THREADS = {"n": 0}


def set_threads(n: int):
    """Change how many CPU cores ExecuTorch uses. Models are reloaded so the change takes effect."""
    if n and n != THREADS["n"]:
        if os.environ.get("BINSENSE_BACKEND") == "torch":
            import torch
            torch.set_num_threads(int(n)); THREADS["n"] = int(n); return
        from executorch.extension.pybindings import portable_lib as pl
        pl._unsafe_reset_threadpool(int(n)); THREADS["n"] = int(n); MODELS.clear()


def prep(m: BinSense, path: Path, q: dict) -> np.ndarray:
    """Photo -> tensor, with the tinkering options: crop style, brightness, contrast, blur."""
    from PIL import Image, ImageEnhance, ImageFilter
    crop, br, co, blur = q.get("crop", "centre"), float(q.get("brightness", 1)), float(q.get("contrast", 1)), float(q.get("blur", 0))
    if crop == "centre" and br == 1 and co == 1 and blur == 0:
        return m.preprocess(path)
    im = Image.open(path).convert("RGB"); r = m.res
    if br != 1:
        im = ImageEnhance.Brightness(im).enhance(br)
    if co != 1:
        im = ImageEnhance.Contrast(im).enhance(co)
    if blur > 0:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    w, h = im.size
    if crop == "whole":                       # squash the whole photo in, nothing cut off
        im = im.resize((r, r), Image.BILINEAR)
    else:
        if crop == "zoom":                    # keep only the middle 60% before resizing
            im = im.crop((int(w * .2), int(h * .2), int(w * .8), int(h * .8))); w, h = im.size
        sc = r / min(w, h); im = im.resize((max(r, round(w * sc)), max(r, round(h * sc))), Image.BILINEAR); w, h = im.size
        l, t = (w - r) // 2, (h - r) // 2; im = im.crop((l, t, l + r, t + r))
    x = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
    return ((x - m.mean) / m.std)[None]


def run_one(m: BinSense, path: Path, sharp: float, q: dict | None = None) -> dict:
    t0 = time.perf_counter(); x = prep(m, path, q or {}); t1 = time.perf_counter(); e = m.embed(x); t2 = time.perf_counter()
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
    if not allowed(path):
        raise ValueError("image not found")
    set_threads(int(q.get("threads", 0)))
    main = get_model(q["pte"], q["pack"]); sharp = float(q.get("sharp", 1)); steps = []; used = "single model"
    final = None
    if q.get("gate_pte") and q.get("gate_pack"):
        g = get_model(q["gate_pte"], q["gate_pack"])
        if g.labels == main.labels:
            s = run_one(g, path, sharp, q); s["role"] = "Fast model"; steps.append(s)
            if max(s["probs"]) >= float(q.get("threshold", 0.9)):
                final, used = s, "fast model"
    if final is None:
        s = run_one(main, path, sharp, q); s["role"] = "Careful model" if steps else "Model"; steps.append(s); final = s
        used = "careful model" if len(steps) > 1 else "single model"
    probs = dict(zip(final["labels"], final["probs"])); top = max(probs, key=probs.get); conf = probs[top]
    multi = main.multi_from({"answer": top, "confidence": conf, "probabilities": probs}, final["nouls"],
                            float(q.get("label_min", 0.25)), float(q.get("tag_min", 0.5)))
    review = conf < float(q.get("review", 0.7))
    return {"image": q["image"], "steps": steps, "answer": top, "confidence": round(conf, 4), "review": review, "used": used,
            "multi": multi["labels"] + multi["tags"], "tags": multi["tag_probabilities"],
            "infer_ms": round(sum(s["infer_ms"] for s in steps), 1), "pre_ms": round(sum(s["pre_ms"] for s in steps), 1)}


def compare(q: dict) -> dict:
    """Same photo through every model, each with its own labels: answer, confidence, time, size."""
    set_threads(int(q.get("threads", 0))); path = (BASE / q["image"]).resolve(); rows = []
    for m in catalog()["models"]:
        pack = q["pack"] if q["pack"] in m["packs"] else m["packs"][0]
        try:
            mod = get_model(m["pte"], pack); run_one(mod, path, 1, q); r = run_one(mod, path, float(q.get("sharp", 1)), q)
            i = int(np.argmax(r["probs"]))
            rows.append({"model": m["pte"], "labels": pack, "answer": r["labels"][i], "confidence": r["probs"][i], "infer_ms": r["infer_ms"], "mb": m["mb"], "res": m["res"]})
        except Exception as e:  # noqa: BLE001
            rows.append({"model": m["pte"], "labels": pack, "answer": f"ERROR {type(e).__name__}", "confidence": 0, "infer_ms": 0, "mb": m["mb"], "res": m["res"]})
    return {"rows": rows}


def bench(q: dict) -> dict:
    """Time the chosen model on one photo: N runs after 3 warm-ups."""
    set_threads(int(q.get("threads", 0))); m = get_model(q["pte"], q["pack"]); x = prep(m, (BASE / q["image"]).resolve(), q)
    for _ in range(3):
        m.embed(x)
    t = []
    for _ in range(int(q.get("runs", 30))):
        a = time.perf_counter(); m.embed(x); t.append((time.perf_counter() - a) * 1000)
    t.sort()
    return {"runs": len(t), "median": round(t[len(t) // 2], 2), "p90": round(t[int(len(t) * .9)], 2), "min": round(t[0], 2), "max": round(t[-1], 2),
            "threads": THREADS["n"] or "default", "per_second": round(1000 / t[len(t) // 2], 1), "model": q["pte"]}


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
            "throttled": vc("get_throttled"), "memory_mb": memory_mb(), "can_edit_labels": can_text(), "cpus": __import__("os").cpu_count(),
            "backend": os.environ.get("BINSENSE_BACKEND", "executorch")}


START = time.time()


def telemetry() -> dict:
    """Live hardware numbers for the machine this server runs on."""
    def vc(*a):
        try:
            return subprocess.run(["vcgencmd", *a], capture_output=True, text=True, timeout=2).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    d = {"host": socket.gethostname(), "board": "not a Raspberry Pi", "cpu": [], "uptime_s": round(time.time() - START)}
    try:
        d["board"] = Path("/proc/device-tree/model").read_text().strip("\0")
    except OSError:
        pass
    try:
        import psutil
        d["cpu"] = psutil.cpu_percent(interval=0.25, percpu=True)
        m = psutil.virtual_memory(); d["mem_used_mb"] = round((m.total - m.available) / 1e6); d["mem_total_mb"] = round(m.total / 1e6)
        d["server_mb"] = round(psutil.Process().memory_info().rss / 1e6)
        busy = sorted(psutil.process_iter(["name", "cpu_percent"]), key=lambda q: -(q.info["cpu_percent"] or 0))[:3]
        d["busiest"] = [f"{q.info['name']} {q.info['cpu_percent']:.0f}%" for q in busy if (q.info["cpu_percent"] or 0) > 1]
    except Exception:  # noqa: BLE001
        pass
    try:
        d["load"] = [round(v, 2) for v in os.getloadavg()]
    except (AttributeError, OSError):
        pass
    t = vc("measure_temp"); c = vc("measure_clock", "arm"); th = vc("get_throttled")
    if "=" in t:
        d["temp_c"] = float(t.split("=")[1].split("'")[0])
    if "=" in c:
        d["clock_mhz"] = round(int(c.split("=")[1]) / 1e6)
    if "=" in th:
        v = int(th.split("=")[1], 16); d["throttled"] = th.split("=")[1]
        d["throttle_text"] = "none" if v == 0 else ", ".join(n for b, n in ((0x1, "under-voltage now"), (0x2, "frequency capped now"), (0x4, "throttled now"),
                                                                            (0x10000, "under-voltage earlier"), (0x20000, "capped earlier"), (0x40000, "throttled earlier")) if v & b)
    try:
        d["governor"] = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        pass
    try:
        from importlib.metadata import version
        d["executorch"] = version("executorch"); d["torch"] = version("torch")
    except Exception:  # noqa: BLE001
        pass
    home = Path.home()
    d["runtime_ready"] = not os.environ.get("BINSENSE_CAMERA_ONLY")
    d["runtime_built_here"] = (home / "ET_BUILD_DONE").exists()
    log = home / "et_build.log"
    if log.exists() and not d["runtime_built_here"]:
        try:
            tail = log.read_text(errors="ignore")[-4000:]
            d["build"] = "failed" if "BUILD FAILED" in tail else "compiling ExecuTorch on this device"
        except OSError:
            pass
    d["threads"] = THREADS["n"] or (os.cpu_count() or 0); d["backend"] = os.environ.get("BINSENSE_BACKEND", "executorch")
    d["models_loaded"] = sorted({k[0] for k in MODELS})
    return d


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


CAM = {"frame": None, "t": 0.0, "err": "", "started": False, "lock": threading.Lock(), "size": ""}


def cam_loop(index: int):
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        CAM["err"] = f"opencv not installed ({type(e).__name__})"; return
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        CAM["err"] = f"camera {index} did not open"; CAM["started"] = False; return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ok, f = cap.read()
        if not ok:
            time.sleep(0.2); continue
        ok2, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok2:
            with CAM["lock"]:
                CAM["frame"] = buf.tobytes(); CAM["t"] = time.time(); CAM["size"] = f"{f.shape[1]}x{f.shape[0]}"
        time.sleep(0.03)


def cam_frame() -> bytes | None:
    url = os.environ.get("BINSENSE_CAMERA_URL")          # take frames from another BinSense server (e.g. the Pi's camera)
    if url:
        try:
            from urllib.request import urlopen
            f = urlopen(url, timeout=4).read()
            with CAM["lock"]:
                CAM["frame"] = f; CAM["t"] = time.time(); CAM["size"] = "remote camera"; CAM["err"] = ""
            return f
        except Exception as e:  # noqa: BLE001
            CAM["err"] = f"remote camera not reachable ({type(e).__name__})"; return None
    if not CAM["started"]:
        CAM["started"] = True; CAM["err"] = ""
        threading.Thread(target=cam_loop, args=(int(os.environ.get("BINSENSE_CAMERA", "0")),), daemon=True).start()
        for _ in range(40):
            if CAM["frame"] or CAM["err"]:
                break
            time.sleep(0.1)
    with CAM["lock"]:
        return CAM["frame"]


def cam_classify(q: dict) -> dict:
    f = cam_frame()
    if f is None:
        raise ValueError(CAM["err"] or "no camera frame yet")
    out = BASE / "results"; out.mkdir(exist_ok=True); (out / "_live.jpg").write_bytes(f)
    r = classify({**q, "image": "results/_live.jpg"}); r["frame_age_ms"] = round((time.time() - CAM["t"]) * 1000); r["frame_size"] = CAM["size"]
    return r


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
            if u.path == "/live":
                return self.send((BASE / "live.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/cam.jpg":
                f = cam_frame()
                return self.send(f, "image/jpeg") if f else self.send({"error": CAM["err"] or "no frame"}, code=503)
            if u.path == "/api/state":
                return self.send({**catalog(), "datasets": datasets(), "system": system()})
            if u.path == "/api/telemetry":
                return self.send(telemetry())
            if u.path == "/api/pi/telemetry":            # the other device (e.g. the Pi), fetched server-side
                from urllib.request import urlopen
                base = os.environ.get("BINSENSE_PI_URL")
                if not base:
                    return self.send({"error": "no second device configured"}, code=404)
                return self.send(urlopen(base + "/api/telemetry", timeout=5).read())
            if u.path == "/api/system":
                return self.send(system())
            if u.path == "/img":
                p = (BASE / unquote(parse_qs(u.query)["p"][0])).resolve()
                if allowed(p):
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
            if u.path == "/api/cam/classify":
                return self.send(cam_classify(q))
            if u.path == "/api/classify":
                return self.send(classify(q))
            if u.path == "/api/compare":
                return self.send(compare(q))
            if u.path == "/api/bench":
                return self.send(bench(q))
            if u.path == "/api/run_all":
                return self.send(run_all(q))
            if u.path == "/api/build_pack":
                return self.send(build_pack(q))
            if u.path == "/api/truth":
                dn = "".join(c for c in q["dataset"] if c.isalnum() or c in "-_")
                d = BASE / dn if (BASE / dn).is_dir() else EXTRA / dn; f = d / "labels.csv"
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
