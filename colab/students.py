# Runs inside the live Colab notebook (exec'd in its globals): distil the fine-tuned MobileCLIP teacher
# into small, low-resolution students, export each to ExecuTorch (FP32 + INT8), and record a speed/accuracy table.
import timm, time, shutil
from google.colab import files

teacher = nets["mobileclip2_s0"].to(dev).eval()
SOUT = Path("binsense_students"); SOUT.mkdir(exist_ok=True)
IMN_MEAN, IMN_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class Body(nn.Module):
    def __init__(s, m):
        super().__init__(); s.body = m
    def forward(s, x):
        return s.body(x)


def student_run(arch, res, epochs=18, lr=2e-3, bs=64, T_kd=3.0, alpha=0.7):
    m = timm.create_model(arch, pretrained=True, num_classes=0)
    net = Net(Body(m), m.num_features, len(LABELS)).to(dev)
    mean = torch.tensor(IMN_MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(IMN_STD, device=dev).view(1, 3, 1, 1)
    aug = T.Compose([T.RandomResizedCrop(256, scale=(0.5, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                     T.RandomRotation(20), T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.ToTensor()])
    plain = T.Compose([T.Resize(res), T.CenterCrop(res), T.ToTensor(), T.Normalize(IMN_MEAN, IMN_STD)])
    counts = np.bincount([y for _, y in train], minlength=len(LABELS))
    sampler = torch.utils.data.WeightedRandomSampler([1.0 / counts[y] for _, y in train], len(train))
    dl = torch.utils.data.DataLoader(DS(train, aug), bs, sampler=sampler, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    @torch.no_grad()
    def predict(items):
        net.eval(); P, Y = [], []
        for x, y in torch.utils.data.DataLoader(DS(items, plain), 128, num_workers=2):
            P.append(net(x.to(dev)).argmax(1).cpu()); Y.append(y)
        return torch.cat(P), torch.cat(Y)

    best, state, t0 = 0, None, time.time()
    for ep in range(epochs):
        net.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)                      # x: 256 px in [0,1] = the teacher's own input format
            with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
                t_logits = teacher(x).float()
            xs = (F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False) - mean) / std
            with torch.autocast(dev, enabled=dev == "cuda"):
                s_logits = net(xs).float()
                kd = F.kl_div(F.log_softmax(s_logits / T_kd, 1), F.softmax(t_logits / T_kd, 1), reduction="batchmean") * T_kd * T_kd
                loss = alpha * kd + (1 - alpha) * F.cross_entropy(s_logits, y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        p, y = predict(val); a = (p == y).float().mean().item()
        if a >= best: best, state = a, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    p, y = predict(test); correct = int((p == y).sum())
    net = net.cpu().eval()
    r = dict(arch=arch, res=res, mean=IMN_MEAN, std=IMN_STD, val_acc=best, test_correct=correct, test_acc=correct / len(y),
             params_m=round(sum(q.numel() for q in m.parameters()) / 1e6, 2), train_seconds=round(time.time() - t0))
    tag = f"stu_{arch.split('.')[0]}_{res}"
    bb = net.backbone.eval(); x = torch.rand(1, 3, res, res)
    W = (net.fc.weight.detach() * net.scale).numpy().astype(np.float32); b = net.fc.bias.detach().numpy().astype(np.float32)
    try:
        fp = SOUT / f"{tag}_fp32.pte"; r["fp32_mb"] = lower(torch.export.export(bb, (x,)), fp)
        mod = _load_for_executorch(str(fp)); r["fp32_pte_correct"] = count_correct(lambda t: mod.forward((t,))[0], r, W, b)
        ts = []
        for _ in range(60):
            s0 = time.perf_counter(); mod.forward((x,)); ts.append((time.perf_counter() - s0) * 1000)
        r["colab_cpu_fp32_ms"] = round(sorted(ts)[30], 2)
        q = XNNPACKQuantizer(); q.set_global(get_symmetric_quantization_config(is_per_channel=True))
        prep = prepare_pt2e(torch.export.export(bb, (x,)).module(), q); move_exported_model_to_eval(prep)
        with torch.no_grad():
            for im, _ in train[:200]: prep(plain(im)[None])
        conv = convert_pt2e(prep)
        with torch.no_grad(): r["int8_converted_correct"] = count_correct(lambda t: conv(t), r, W, b)
        ip = SOUT / f"{tag}_int8.pte"; r["int8_mb"] = lower(torch.export.export(conv, (x,)), ip)
        mi = _load_for_executorch(str(ip)); ts = []
        for _ in range(60):
            s0 = time.perf_counter(); mi.forward((x,)); ts.append((time.perf_counter() - s0) * 1000)
        r["colab_cpu_int8_ms"] = round(sorted(ts)[30], 2)
    except Exception as e:
        r["export_error"] = repr(e)[:300]
    meta = {"model": f"timm {arch} distilled from fine-tuned MobileCLIP2-S0 (TrashNet)", "pretrained": "distilled", "logit_scale": 1.0,
            "labels": LABELS, "preprocess": {"size": res, "mean": IMN_MEAN, "std": IMN_STD}}
    np.savez(SOUT / f"{tag}.npz", meta=np.array(json.dumps(meta)), head_w=W, head_b=b)
    print({k: v for k, v in r.items() if k not in ("mean", "std")}, flush=True)
    return r


GRID = [("mobilenetv2_100.ra_in1k", 160), ("mobilenetv2_100.ra_in1k", 128), ("mobilenetv2_100.ra_in1k", 96),
        ("mobilenetv2_050.lamb_in1k", 160), ("mobilenetv2_050.lamb_in1k", 128), ("mobilenetv2_050.lamb_in1k", 96),
        ("mobilenetv4_conv_small.e2400_r224_in1k", 160), ("mobilenetv4_conv_small.e2400_r224_in1k", 128),
        ("mobilenetv3_small_100.lamb_in1k", 160), ("mobilenetv3_small_100.lamb_in1k", 128)]
student_results = []
for arch, res in GRID:
    try:
        student_results.append(student_run(arch, res))
    except Exception as e:
        student_results.append({"arch": arch, "res": res, "error": repr(e)[:300]}); print("FAILED", arch, res, repr(e)[:200], flush=True)
    json.dump(student_results, open(SOUT / "students.json", "w"), indent=1)
print("STUDENTS DONE")
shutil.make_archive("binsense_students", "zip", "binsense_students")
files.download("binsense_students.zip")
