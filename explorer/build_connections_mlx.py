"""Direct-path connections for every neuron in every layer, for the explorer's details tab, on MLX (Apple silicon).
build_connections.py on sweep/qwen_mlx.py: the same math and precision (bf16 weights, fp32 products), flags and output.

For layers i < j, P[B, A] = (W_gate_j[B] . v_A) * (W_up_j[B] . v_A) with v_A = (1 + w_j) * W_down_i[:, A], w_j the
pre-MLP norm weight of layer j. Neuron A's downstream links in layer j are column A of P, |P| z-scored over B; neuron
B's upstream links in layer i are row B of P, |P| z-scored over A. Each neuron keeps its k strongest links per side
across all layers, with sign(P).

    uv run python explorer/build_connections_mlx.py

Writes explorer/cache/connections.npz, the file build_connections.py writes: {down,up}_layer [L, N, k] int8,
{down,up}_n int16, {down,up}_z float16, {down,up}_sign int8. Each layer pair is evaluated on its own, so besides the
weights (about 9 GB for the 4B in bf16) only one pair's N x N products, z-scores and sort buffers are live: about 4 GB
at N = 9216.
"""
import argparse
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweep"))
from qwen_mlx import Qwen, topk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--k", type=int, default=24)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "connections.npz"))
a = ap.parse_args()

# Weights come through qwen_mlx, so a hub id or local dir, sharded or single-file, or an mlx-lm conversion (quantized
# ones dequantized) all load. norm_scale is the fp32 1 + w of the stored w; like the original, everything is bf16 here.
q = Qwen(a.model)
L, N, K = q.n_layers, q.inter, a.k
bf16 = mx.bfloat16
gate, up, down = zip(*[[w.astype(bf16) for w in q.mlp_weights(l)] for l in range(L)])
norm = [q.norm_scale(l).astype(bf16) for l in range(L)]
mx.eval(gate, up, down, norm)

# running top-k per neuron and side, per layer: score, source layer, source neuron, sign
best = {s: [(mx.full((N, K), -1e9), mx.zeros((N, K), mx.int16), mx.zeros((N, K), mx.int16), mx.zeros((N, K), mx.int8))
            for _ in range(L)] for s in ("down", "up")}


def merge(side, l, z, idx, other_layer, p):
    bz, bl, bn, bs = best[side][l]
    cz = mx.concatenate([bz, z], 1)
    cl = mx.concatenate([bl, mx.full(idx.shape, other_layer, mx.int16)], 1)
    cn = mx.concatenate([bn, idx.astype(mx.int16)], 1)
    cs = mx.concatenate([bs, mx.where(p > 0, 1, -1).astype(mx.int8)], 1)
    vals, top = topk(cz, K)
    best[side][l] = (vals, *(mx.take_along_axis(c, top, 1) for c in (cl, cn, cs)))


def zscore(M, axis):
    """M z-scored along an axis with the unbiased std, as torch's M.std computes it."""
    return (M - M.mean(axis, keepdims=True)) / (mx.std(M, axis=axis, keepdims=True, ddof=1) + 1e-8)


t0 = time.time()
for j in range(1, L):
    for i in range(j):
        V = down[i] * norm[j][:, None]  # [hidden, N_i]
        P = (gate[j] @ V).astype(mx.float32) * (up[j] @ V).astype(mx.float32)  # [N_j (B), N_i (A)]
        M = mx.abs(P)
        # downstream of A (layer i): column A over B, on the transpose since topk runs along the last axis
        z, idx = topk(zscore(M, 0).T, K)  # [N_i, K]
        merge("down", i, z, idx, j, mx.take_along_axis(P.T, idx, 1))
        # upstream of B (layer j): row B over A
        z, idx = topk(zscore(M, 1), K)  # [N_j, K]
        merge("up", j, z, idx, i, mx.take_along_axis(P, idx, 1))
        mx.eval(best["down"][i], best["up"][j])  # one pair per graph, so only one P and its z-scores are ever live
    mx.clear_cache()
    print(f"L{j} done ({time.time() - t0:.0f}s)", flush=True)

out = {}
for side, per_layer in best.items():
    bz, bl, bn, bs = (mx.stack(c) for c in zip(*per_layer))  # [L, N, K] each
    out[f"{side}_z"] = np.array(mx.maximum(bz, -60000)).astype(np.float16)
    out[f"{side}_layer"] = np.array(bl).astype(np.int8)
    out[f"{side}_n"] = np.array(bn)
    out[f"{side}_sign"] = np.array(bs)
np.savez_compressed(a.out, **out, k=K)
print("->", a.out, f"{os.path.getsize(a.out) / 1e6:.0f} MB")
