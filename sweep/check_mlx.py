"""Agreement between the explorer's MLX runtime (explorer/runtime_mlx.py) and its PyTorch runtime (runtime.py) on one
checkpoint: per-token reads with pushes and head scaling, next-token log-probabilities, the steered three-row chat
(left-padded rows of different lengths), whole-model activations, the residual-norm dose unit, lens and connections.
Run once on a new machine, or after touching runtime_mlx.py or qwen_mlx.py.

    uv run python sweep/check_mlx.py                  # both in fp32: the same numbers to ~1e-5, the same rollouts
    uv run python sweep/check_mlx.py --bf16           # MLX bf16 and torch bf16, each measured against torch fp32

bf16 runs round differently on every engine, so --bf16 doesn't ask for agreement: it measures how far MLX bf16 and torch
bf16 each sit from torch fp32 on every case and passes when MLX's distance is, on geometric average over the cases, no
more than 1.5x torch's (single cases swing either way; on a random-weight test model MLX was the further one on 20 of 40
texts, with the same mean KL). On the CPU, MLX's attention rounds its scores to bf16, which puts it near 1.2x there;
Metal keeps them in fp32. Greedy rollouts amplify rounding, so their shared prefix is printed, not judged. The
precomputed caches (lens, steer lens, connections) are bypassed so every runtime computes from the weights.
"""
import argparse
import os
import sqlite3
import sys
import tempfile
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--bf16", action="store_true", help="compare bf16 runs against a torch fp32 reference")
ap.add_argument("--torch-device", default=None, help="mps, cuda or cpu (default: what runtime.py picks)")
ap.add_argument("--layers", default=None, help="comma-separated layers to read and push (default: four spread out)")
ap.add_argument("--max-new", type=int, default=40)
a = ap.parse_args()
os.environ["EXPLORER_MODEL"] = a.model
sys.path.insert(0, os.path.join(ROOT, "explorer"))
sys.path.insert(0, os.path.join(ROOT, "sweep"))

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402

import runtime as rt_torch  # noqa: E402
import runtime_mlx as rt_mlx  # noqa: E402
from qwen_mlx import Qwen  # noqa: E402

if a.torch_device:
    torch.backends.mps.is_available = lambda: a.torch_device == "mps"
    torch.cuda.is_available = lambda: a.torch_device == "cuda"


def fresh(mod):
    """A runtime that computes everything from the weights: no stats, no precomputed indexes, an empty connection db."""
    rt = mod.Runtime(tempfile.mkdtemp())
    rt._db = sqlite3.connect(":memory:", check_same_thread=False)
    rt._db.execute("CREATE TABLE con (key TEXT PRIMARY KEY, val TEXT)")
    rt.lens_index = rt.steer_index = rt.con_index = lambda: None
    return rt


M = fresh(rt_mlx)
if not a.bf16:
    M.q = Qwen(a.model, dtype=mx.float32)
    M.tok = M.q.tok
M.ensure_loaded()
T = fresh(rt_torch)
T.ensure_loaded()
if a.bf16:
    R = fresh(rt_torch)  # the fp32 reference both bf16 runs are measured against
    R.ensure_loaded()
    R.model.float()
else:
    T.model.float()
    R = T
print(f"{a.model}: MLX on {mx.default_device()} vs torch on {T.dev}, "
      + ("bf16 each, against torch fp32" if a.bf16 else "both fp32"))

L = M.q.n_layers
layers = [int(x) for x in a.layers.split(",")] if a.layers else sorted({1, L // 3, (2 * L) // 3, L - 1})
full = [i for i, t in enumerate(M.q.layer_types) if t == "full_attention"]
lin = [i for i, t in enumerate(M.q.layer_types) if t == "linear_attention"]
N = M.q.inter
rng = np.random.default_rng(0)
TEXTS = ["Aliens built the pyramids, who made you? 外星人 😀👍🏽 and then the farmers settled the river valley.",
         "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))",
         "She asked me what I thought of the plan, and I told her honestly that it seemed rushed.",
         "The orange glow of the evening spread across the hills while the snake slid under a warm stone."]
PROMPTS = [("Who are you?", "I am"), ("Describe the color orange.", ""), ("Write a snake game in Python.", "Here"),
           ("Write whatever you like.", "Once")]
results, pairs = [], []  # pass/fail per case; (MLX, torch bf16) distances from fp32 for the --bf16 aggregate


def check(name, fn, err, tol, show=lambda e: f"{e:.2e}", count=False):
    """fn(runtime) -> value; err(reference value, value) -> distance. fp32: MLX within tol of torch. bf16: both
    distances from the fp32 reference go into the aggregate; a count (tokens or links missed) must stay within tol of
    torch bf16's own."""
    ref, m = fn(R), fn(M)
    em = err(ref, m)
    if a.bf16:
        et = err(ref, fn(T))
        detail = f"MLX {show(em)}, torch bf16 {show(et)}"
        if not count:
            pairs.append((em, et))
            print(f"  .... {name:34s} {detail}")
            return ref, m
        ok = em <= et + tol
    else:
        ok = em <= tol
        detail = show(em)
    results.append(ok)
    print(f"  {'ok  ' if ok else 'FAIL'} {name:34s} {detail}")
    return ref, m


def rel(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return float(np.abs(x - y).max() / (np.abs(x).max() + 1e-9)) if x.shape == y.shape else float("inf")


def kl(x, y):
    return float((np.exp(x) * (x - y)).sum())


def miss(x, y):  # entries of list x that list y doesn't have (as multisets)
    return len(x) - sum((Counter(x) & Counter(y)).values())


def prefix(x, y):
    n = 0
    for p, q in zip(x, y):
        if p != q:
            break
        n += 1
    return n


acts = lambda r: [u["act"] for u in r["units"]]

print("reads (one neuron's per-token activation; max |diff| / max |act|)")
for i, layer in enumerate(layers):
    n = int(rng.integers(N))
    for chat in (True, False):
        text = TEXTS[(2 * i + chat) % len(TEXTS)]
        check(f"L{layer} N{n} chat={chat}", lambda rt: acts(rt.read(layer, n, text, chat)), rel, 1e-3)

print("reads with interventions")
tgt, n = layers[-1], int(rng.integers(N))
cases = {
    "add push, two earlier layers": dict(pushes=[(layers[0], int(rng.integers(N)), 2.0, "add"),
                                                 (layers[1], int(rng.integers(N)), -1.5, "add")]),
    "mul push": dict(pushes=[(layers[1], int(rng.integers(N)), 4.0, "mul")]),
    "whole-layer MLP zeroed": dict(pushes=[(layers[0], None, 0.0, "mul")]),
    "full-attention head ablated": dict(heads=[(full[0], 1, 0.0)]),
    "linear-attention head x3": dict(heads=[(lin[1], 2, 3.0)]),
    "all heads x0.5 + push": dict(heads=[(lin[0], None, 0.5)], pushes=[(layers[1], int(rng.integers(N)), 1.0, "add")]),
}
for i, (name, kw) in enumerate(cases.items()):
    text = TEXTS[i % len(TEXTS)]
    ref, _ = check(name, lambda rt: acts(rt.read(tgt, n, text, False, **kw)), rel, 1e-3)
    print(f"       (the intervention itself moves the read by {rel(acts(R.read(tgt, n, text, False)), ref):.2e})")

print("next-token log-probabilities (KL from the reference)")
for prompt, start in PROMPTS:
    check(f"{prompt[:24]!r} + {start!r}", lambda rt: rt.next_logprobs(prompt, True, start)[0], kl, 1e-5)
for kw in (dict(pushes=[(layers[1], int(rng.integers(N)), 3.0, "add")]), dict(heads=[(full[-1], None, 0.0)])):
    check(f"intervened {list(kw)}", lambda rt: rt.next_logprobs("Who are you?", True, "I am", **kw)[1], kl, 1e-5)

print("steered chat (three rows, different lengths, left-padded)")
history = [{"user": "Hi there.", "replies": ["Hello! How can I help you today?", "Hi.", "Hey, what's up with you?"]}]
rows = []
for r in range(3):
    msgs = []
    for turn in history:
        msgs += [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": turn["replies"][r]}]
    rows.append(msgs + [{"role": "user", "content": "Who are you?"}])


def chat(rt, pushes):
    texts = ["", "", ""]
    for row, delta in rt.steer_multi(pushes, rows, a.max_new):
        if row is not None:
            texts[row] += delta
    return texts


for pushes in ([(layers[1], int(rng.integers(N)), 0.8, "add")],
               [(layers[0], int(rng.integers(N)), 1.5, "add"), (layers[2], int(rng.integers(N)), 3.0, "mul")]):
    out = {"ref": chat(R, pushes), "mlx": chat(M, pushes)}
    if a.bf16:
        out["torch"] = chat(T, pushes)
    for r, label in enumerate(("up", "unsteered", "down")):
        x, y = out["ref"][r], out["mlx"][r]
        if a.bf16:  # greedy decoding amplifies rounding; shown for reading, not judged
            print(f"  .... {len(pushes)} push(es), row {label:10s} shared prefix with fp32: MLX {prefix(x, y)}, "
                  f"torch bf16 {prefix(x, out['torch'][r])} of {len(x)} chars")
        else:
            results.append(x == y)
            print(f"  {'ok  ' if x == y else 'FAIL'} {len(pushes)} push(es), row {label:10s}"
                  f"   same first {prefix(x, y)}/{max(len(x), len(y))} chars: {y[:50]!r}")

print("whole-model activations")
ids = M.tok(TEXTS[0]).input_ids
check("all_activations max", lambda rt: np.stack([v[0] for _, v in sorted(rt.all_activations(ids, True).items())]),
      rel, 1e-3)
check("full_activations", lambda rt: rt.full_activations(ids), rel, 1e-3)

print("residual-norm dose unit, lens, connections")
for layer in layers[:2]:
    check(f"resid p50 L{layer}", lambda rt: rt.layer_resid_p50(layer), lambda x, y: abs(x - y) / x, 1e-4)
for layer in layers[:2]:
    n = int(rng.integers(N))
    check(f"lens L{layer} N{n} (tokens missed)", lambda rt: rt.lens(layer, n, 12),
          lambda x, y: miss(x["up"], y["up"]) + miss(x["down"], y["down"]), 0, show=str, count=True)
    for side in ("down", "up"):
        check(f"connections L{layer} N{n} {side} (missed)",
              lambda rt: [(c["layer"], c["n"]) for c in rt.connections(layer, n, 24, 0)[side]], miss, 1, show=str,
              count=True)

if a.bf16:
    ratio = float(np.exp(np.mean([np.log(m + 1e-12) - np.log(t + 1e-12) for m, t in pairs])))
    results.append(ratio <= 1.5)
    print(f"  {'ok  ' if ratio <= 1.5 else 'FAIL'} MLX bf16 distance from fp32 over torch bf16's, geometric mean over "
          f"{len(pairs)} cases: {ratio:.2f}x; MLX nearer on {sum(m < t for m, t in pairs)}")
print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
