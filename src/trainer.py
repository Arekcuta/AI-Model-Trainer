import os, json, sys, torch, torch.nn as nn, torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

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
class Net(nn.Module):
    def __init__(self, layers, act):
        super().__init__()
        act = dict(relu=nn.ReLU, gelu=nn.GELU)[act]
        seq = []
        for i, o in zip(layers, layers[1:]):
            seq += [nn.Linear(i, o), act()]
        self.core = nn.Sequential(*seq[:-1])  # last act removed
    def forward(self, x): return self.core(x)

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


    model = maybe_compile(Net(cfg["layers"], cfg["activ"]).to(device))
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

    torch.save(model.state_dict(), "./../models/model_weights.pt")
    torch.jit.script(model).save("./../models/model_scripted.pt")

    try:
        import onnx  # noqa: F401
        dummy = torch.zeros(1, cfg["layers"][0], device=device)
        torch.onnx.export(model, dummy, "./../models/model.onnx",
                          input_names=["input"], output_names=["logits"],
                          opset_version=18)
    except ImportError:
        print("ONNX not installed – skipped export", flush=True)

# ── multiprocess‑safe entry for Windows ──────────────────────────────
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main(sys.argv[1])
