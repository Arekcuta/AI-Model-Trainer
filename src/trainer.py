import os, json, sys
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ModuleNotFoundError:
    print("PyTorch is required. Install it from https://pytorch.org", file=sys.stderr)
    sys.exit(1)
import numpy as np
from pathlib import Path

torch.backends.cudnn.benchmark = True
device = "cuda"

# ── data prep with persistent cache ──────────────────────────────────
def load_or_build_npz(json_path: str, max_t: int, max_l: int):
    cache = Path(json_path).with_suffix(".npz")
    if cache.exists():
        arr = np.load(cache, allow_pickle=False)
        X, y = arr["X"], arr["y"]
    else:
        rows = json.load(open(json_path))
        def vec_text(t: str):
            a = [ord(c) / 255 for c in t[:max_t]]
            return a + [0.0] * (max_t - len(a))
        def vec_lab(s: str):
            b = [int(x) for x in s[:max_l]]
            return b + [0] * (max_l - len(b))
        X = np.array([vec_text(r["text"])  for r in rows], dtype=np.float32)
        y = np.array([vec_lab (r["label"]) for r in rows], dtype=np.float32)
        np.savez_compressed(cache, X=X, y=y)
    return X, y

# ── new make_loader: preload entire dataset onto GPU ────────────────
def make_loader(X_np: np.ndarray, y_np: np.ndarray, batch: int):
    """
    Takes numpy arrays, moves them once to GPU, and yields CUDA
    slices.  No CPU collation / PCIe transfer in the training loop.
    """
    X_cuda = torch.from_numpy(X_np).to(device, non_blocking=True)
    y_cuda = torch.from_numpy(y_np).to(device, non_blocking=True)

    N = X_cuda.size(0)
    indices = torch.arange(N, device=device)

    def batch_iter():
        perm = torch.randperm(N, device=device)   # shuffle every epoch
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            yield X_cuda[idx], y_cuda[idx]

    return batch_iter


# ── simple MLP ───────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, layers, act):
        super().__init__()
        act = dict(relu=nn.ReLU, gelu=nn.GELU)[act]
        seq = []
        for i, o in zip(layers, layers[1:]):
            seq += [nn.Linear(i, o), act()]
        self.core = nn.Sequential(*seq[:-1])  # last act removed

    def forward(self, x):
        return self.core(x)


class BiLSTM(nn.Module):
    def __init__(self, hidden, out_size):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, out_size)

    def forward(self, x):
        x = x.unsqueeze(-1)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h)


class TransformerNet(nn.Module):
    def __init__(self, dim, depth, heads, out_size):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.fc = nn.Linear(dim, out_size)

    def forward(self, x):
        x = self.proj(x.unsqueeze(-1))
        x = self.enc(x)
        x = x.mean(dim=1)
        return self.fc(x)

# ── optional compile helper (skip Triton on Windows) ────────────────
def maybe_compile(model):
    """
    Try torch.compile() → falls back to eager if Triton / Inductor
    isn't available (current Windows wheels).
    """
    try:
        import triton  # noqa: F401
        return torch.compile(model)
    except Exception:
        print("Triton not found – running eager", flush=True)
        return model


# ── main training routine ────────────────────────────────────────────
def main(cfg_path: str):
    cfg       = json.load(open(cfg_path))
    workers   = min(8, os.cpu_count())
    batch   = cfg.get("batch", 4096)          # keep the same
    X, y    = load_or_build_npz(cfg["data"], cfg["max_text_len"], cfg["max_lab_len"])
    loader  = make_loader(X, y, batch)        # ← new call returns an iterator factory


    model_type = cfg.get("model", "mlp")
    if model_type == "bilstm":
        hidden = cfg.get("hidden", 128)
        model = BiLSTM(hidden, cfg["max_lab_len"]) 
    elif model_type == "transformer":
        dim = cfg.get("d_model", 128)
        depth = cfg.get("depth", 2)
        heads = cfg.get("heads", 4)
        model = TransformerNet(dim, depth, heads, cfg["max_lab_len"])
    else:
        model = MLP(cfg["layers"], cfg.get("activ", "relu"))
    model = maybe_compile(model.to(device))

    resume = cfg.get("resume")
    if resume and os.path.exists(resume):
        print(f"Loading weights from {resume}", flush=True)
        model.load_state_dict(torch.load(resume, map_location=device))
    opt       = optim.AdamW(model.parameters(), lr=cfg["lr"])
    lossf     = nn.BCEWithLogitsLoss()
    scaler    = torch.amp.GradScaler("cuda")

    for epoch in range(cfg["epochs"]):
        for xb, yb in loader():      # parentheses!
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                loss   = lossf(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        print(json.dumps({"epoch": epoch, "loss": loss.item()}), flush=True)

    model_dir_base = Path("./../models")
    i = 0
    while (model_dir_base / f"model{i}").exists():
        i += 1
    model_dir = model_dir_base / f"model{i}"
    model_dir.mkdir(parents=True)

    # ── save model and config ────────────────────────────────────────
    torch.save(model.state_dict(), model_dir / "model_weights.pt")
    torch.jit.script(model).save(model_dir / "model_scripted.pt")
    json.dump(cfg, open(model_dir / "cfg.json", "w"), indent=2)

    try:
        import onnx  # noqa: F401
        if model_type == "mlp":
            dummy = torch.zeros(1, cfg["layers"][0], device=device)
        else:
            dummy = torch.zeros(1, cfg["max_text_len"], device=device)
        torch.onnx.export(
            model,
            dummy,
            model_dir / "model.onnx",
            input_names=["input"],
            output_names=["logits"],
            opset_version=18,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
    except ImportError:
        print("ONNX not installed – skipped export", flush=True)

# ── multiprocess‑safe entry for Windows ──────────────────────────────
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main(sys.argv[1])
