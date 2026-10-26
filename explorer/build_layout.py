"""Seriated sheet layouts: places each layer's neurons on its 96x96 grid so that neurons with similar weights sit
together. Embeds each neuron by its down_proj column ("writes": the direction it adds to the residual stream) or its
gate and up rows ("reads"), cosine-normalised, PCA to 50 dims, UMAP to 2D, then snaps to the grid: sort by y into
rows of 96, then by x within each row.

    uv run python explorer/build_layout.py --kind writes

Writes explorer/cache/layout_<kind>.npz: cell[L, N] (int16, row * 96 + col for neuron n of layer L).
"""
import argparse
import json
import os
import struct

import numpy as np
import umap
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--kind", default="writes", choices=["writes", "reads"])
ap.add_argument("--pca", type=int, default=50)
ap.add_argument("--neighbors", type=int, default=30)
a = ap.parse_args()

root = snapshot_download(a.model, allow_patterns=["*.safetensors", "*.json"])
index_path = os.path.join(root, "model.safetensors.index.json")
index = json.load(open(index_path))["weight_map"] if os.path.exists(index_path) else None  # None: one model.safetensors
cfg = json.load(open(os.path.join(root, "config.json")))["text_config"]
L, N = cfg["num_hidden_layers"], cfg["intermediate_size"]
side = int(round(N ** 0.5))
assert side * side == N
headers = {}


def tensor(name):
    """A checkpoint tensor as float32, read straight from its safetensors file (an 8-byte little-endian header length,
    a JSON header with each tensor's dtype, shape and byte range, then the data), so torch isn't needed; safetensors'
    numpy loader can't read bf16. A bf16 is the top half of an fp32, so shifting its bits up 16 widens it exactly."""
    path = os.path.join(root, index[name] if index else "model.safetensors")
    if path not in headers:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            headers[path] = (8 + n, json.loads(f.read(n)))
    base, h = headers[path]
    t = h[name]
    begin, end = t["data_offsets"]
    dt = np.dtype({"BF16": "<u2", "F16": "<f2", "F32": "<f4", "F64": "<f8"}[t["dtype"]])
    x = np.fromfile(path, dt, (end - begin) // dt.itemsize, offset=base + begin).reshape(t["shape"])
    return (x.astype(np.uint32) << 16).view(np.float32) if t["dtype"] == "BF16" else x.astype(np.float32)


cells = np.zeros((L, N), np.int16)
for l in range(L):
    pre = f"model.language_model.layers.{l}.mlp."
    if a.kind == "writes":
        X = tensor(pre + "down_proj.weight").T  # [N, hidden]
    else:
        X = np.concatenate([tensor(pre + "gate_proj.weight"), tensor(pre + "up_proj.weight")], 1)  # [N, 2*hidden]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = X - X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    Z = U[:, : a.pca] * S[: a.pca]
    xy = umap.UMAP(n_neighbors=a.neighbors, min_dist=0.05, metric="cosine", random_state=0).fit_transform(Z)
    by_y = np.argsort(xy[:, 1], kind="stable")
    for r in range(side):
        band = by_y[r * side:(r + 1) * side]
        band = band[np.argsort(xy[band, 0], kind="stable")]
        cells[l, band] = r * side + np.arange(side)
    print(f"L{l} done", flush=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", f"layout_{a.kind}.npz")
np.savez(out, cell=cells)
print("->", out)
