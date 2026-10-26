"""Steer lens on MLX (Apple silicon): what pushing each neuron does to the model's next-token prediction, through every
layer after it. steer_lens.py on sweep/qwen_mlx.py: the same contexts, pushes and output files.

For each neuron and sign, adds ±dose x (median residual norm / ||W_down[:, n]||) to the neuron at every position of a
few short contexts, runs one forward pass, and compares the last position's log-probabilities with the unsteered run.
Keeps, per neuron and sign, the 12 tokens whose log-probability rises most (averaged over contexts) and the KL
divergence from the unsteered prediction.

    uv run python sweep/steer_lens_mlx.py --layers 0-7 --out stats/steer_lens

Writes stats/steer_lens_L{l}.npz per layer: up_tok[N, 12], down_tok[N, 12] (token ids for the + push and - push),
up_kl[N], down_kl[N].

Layers before the pushed one see no push and are the same for every row, so each steered pass takes their output from
the unsteered pass instead of recomputing it (lazy evaluation then never runs them): about half the layer passes of a
full sweep, with the same results (bit-identical in fp32 on the test models).
"""
import argparse
import os
import time

import mlx.core as mx
import numpy as np

from qwen_mlx import Hook, Qwen, topk

CHAT = ["Who are you?", "Write whatever you like."]
PLAIN = ["I think the most important thing to remember is", "Today we are going to talk about"]


class Push(Hook):
    """Adds vals[r] to neuron cols[r] of row r at every position of `layer`'s activations. Records the residual leaving
    each layer into `seen` when that is a dict; `prefix` = (layer, residual of the contexts) stands in, tiled over the
    rows, for the residual leaving that layer."""

    def __init__(self):
        self.layer = self.cols = self.vals = None
        self.seen = None
        self.prefix = None

    def act(self, i, a, gate, up):
        if i != self.layer:
            return a
        on = mx.arange(a.shape[-1]) == self.cols[:, None, None]
        return mx.where(on, a + self.vals.astype(a.dtype)[:, None, None], a)

    def resid(self, i, h):
        if self.seen is not None:
            self.seen[i] = h
        if self.prefix is not None and self.prefix[0] == i:
            p = self.prefix[1]
            return mx.tile(p, (h.shape[0] // p.shape[0], 1, 1))
        return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layers", default="0-31")
    ap.add_argument("--dose", type=float, default=0.8)
    ap.add_argument("--neurons-per-batch", type=int, default=96)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--out", default="stats/steer_lens")
    a = ap.parse_args()
    lo, _, hi = a.layers.partition("-")
    layers = list(range(int(lo), int(hi or lo) + 1))

    q = Qwen(a.model)
    tok = q.tok
    ctxs = [tok.apply_chat_template([{"role": "user", "content": c}], add_generation_prompt=True, enable_thinking=False,
                                    tokenize=True) for c in CHAT] + [tok(p).input_ids for p in PLAIN]
    ctxs = [list(c["input_ids"] if hasattr(c, "keys") else c) for c in ctxs]
    C, T = len(ctxs), max(map(len, ctxs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    left = [T - len(c) for c in ctxs]
    ids = mx.array([[pad] * p + c for p, c in zip(left, ctxs)], dtype=mx.int32)
    real = np.array([[0] * p + [1] * len(c) for p, c in zip(left, ctxs)], bool)

    hook = Push()

    def last_logprobs(input_ids, left_padding):
        # a fresh cache per pass builds the padding masks and each row's positions (its real tokens start at 0)
        lp = q(input_ids, cache=q.make_cache(left_padding), hook=hook)[:, -1].astype(mx.float32)
        return lp - mx.logsumexp(lp, axis=-1, keepdims=True)

    hook.seen = {}
    base = last_logprobs(ids, left)  # [C, V]
    resid, hook.seen = hook.seen, None
    mx.eval(base, resid)
    # median residual norm leaving each layer over these contexts' real tokens (the lower middle value, as torch.median)
    flat = mx.array(np.flatnonzero(real))
    norms = {}
    for i in layers:
        r = mx.sort(mx.linalg.norm(resid[i].reshape(C * T, -1)[flat].astype(mx.float32), axis=-1))
        norms[i] = r[(r.size - 1) // 2].item()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    B = a.neurons_per_batch
    for l in layers:
        W = q.down(l)  # [hidden, inter]
        N = W.shape[1]
        unit = norms[l] / mx.linalg.norm(W.astype(mx.float32), axis=0)  # [N]
        hook.prefix = (l - 1, resid[l - 1]) if l > 0 else None
        out = {k: [] for k in ("up_tok", "down_tok", "up_kl", "down_kl")}
        t0 = time.time()
        for s in range(0, N, B):
            n = mx.arange(s, min(s + B, N))
            b = n.size
            for sign, key in ((1.0, "up"), (-1.0, "down")):
                # rows: neuron-major, contexts repeated
                cols = mx.repeat(n, C)
                vals = sign * a.dose * unit[cols]
                hook.layer, hook.cols, hook.vals = l, cols, vals
                lp = last_logprobs(mx.tile(ids, (b, 1)), left * b).reshape(b, C, -1)
                hook.layer = None
                delta = (lp - base[None]).mean(1)  # [b, V]
                kl = (mx.exp(lp) * (lp - base[None])).sum(-1).mean(1)  # KL(steered || base), averaged over contexts
                out[f"{key}_tok"].append(topk(delta, a.k)[1].astype(mx.int32))
                out[f"{key}_kl"].append(kl)
                mx.eval(out[f"{key}_tok"][-1], out[f"{key}_kl"][-1])
        hook.prefix = None
        arrs = {k: np.array(mx.concatenate(v)) for k, v in out.items()}
        np.savez(f"{a.out}_L{l}.npz", **arrs, dose=a.dose, contexts=np.array(CHAT + PLAIN))
        print(f"L{l}: {N} neurons in {time.time() - t0:.0f}s, KL+ median {np.median(arrs['up_kl']):.3f}", flush=True)


if __name__ == "__main__":
    main()
