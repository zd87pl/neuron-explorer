"""Per-neuron activation ranges for MLP layers of a Qwen3.5 model, plus their down-projection columns, on MLX (Apple
silicon). neuron_stats.py on sweep/qwen_mlx.py: the same corpus, sign regimes and output file.

A neuron's activation is silu(gate) * up, the input to down_proj. Positive activations come from two sign regimes
(gate>0, up>0) and (gate<0, up<0); negative ones from (gate>0, up<0) and (gate<0, up>0). This records the extreme of
each regime per neuron over a mixed web + chat corpus, and saves W_down[:, n] for every neuron, which is the direction
a steering push on neuron n adds to the residual stream.

    uv run python sweep/neuron_stats_mlx.py --model Qwen/Qwen3.5-4B --layers 22 --tokens 2000000 --out stats/q35-4b

Writes stats/q35-4b_L22.npz, the same file neuron_stats.py writes, with
    pp_max, nn_max   largest positive activation in the (+,+) and (-,-) regimes          [I]
    pn_min, np_min   most negative activation in the (gate+, up-) and (gate-, up+) regimes [I]
    abs_max          max |activation| over everything (the unit for steering doses)      [I]
    w_down           down_proj columns, one row per neuron                                [I, H] float16
    resid_norm_q10_50_90   quantiles of the per-token residual norm leaving the layer (scale for push sizes)
"""
import argparse
import itertools
import os
import time

import mlx.core as mx
import numpy as np

from qwen_mlx import Hook, Qwen


def text_stream(tok, seed):
    from datasets import load_dataset  # imported here so a test can swap in its own corpus without datasets installed
    web = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed)
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).shuffle(seed=seed)
    for w, c in zip(web, chat):
        yield w["text"]
        yield tok.apply_chat_template(c["messages"], tokenize=False, enable_thinking=False)


def windows(tok, seq_len, batch, seed):
    """Packs documents (EOS-separated) into [batch, seq_len] token windows."""
    buf = []
    eos = tok.eos_token_id
    for text in text_stream(tok, seed):
        buf.extend(tok(text, add_special_tokens=False)["input_ids"] + [eos])
        while len(buf) >= seq_len * batch:
            chunk, buf = buf[: seq_len * batch], buf[seq_len * batch :]
            yield mx.array(chunk, dtype=mx.int32).reshape(batch, seq_len)


class RegimeStats(Hook):
    """Running regime extremes and residual norms for the target layers, kept on the device."""

    def __init__(self, layers, n):
        z = lambda: mx.zeros((n,), mx.float32)
        self.acc = {l: {"pp_max": z(), "nn_max": z(), "pn_min": z(), "np_min": z()} for l in layers}
        self.resid_norms = {l: [] for l in layers}  # per-token norm leaving the layer, positions >= 1 (0 is a sink)

    def act(self, i, a, gate, up):
        s = self.acc.get(i)
        if s is None:
            return a
        # position 0 of each window is an attention sink with outsized activations; it would inflate abs_max
        g, u, x = (t[:, 1:].reshape(-1, t.shape[-1]) for t in (gate, up, a))
        gp, up_ = g > 0, u > 0
        # max and min only pick values, so reducing in the activation dtype and widening the result is exact
        f32 = lambda v: v.astype(mx.float32)
        s["pp_max"] = mx.maximum(s["pp_max"], f32(mx.where(gp & up_, x, -mx.inf).max(0)))
        s["nn_max"] = mx.maximum(s["nn_max"], f32(mx.where(~gp & ~up_, x, -mx.inf).max(0)))
        s["pn_min"] = mx.minimum(s["pn_min"], f32(mx.where(gp & ~up_, x, mx.inf).min(0)))
        s["np_min"] = mx.minimum(s["np_min"], f32(mx.where(~gp & up_, x, mx.inf).min(0)))
        return a

    def resid(self, i, h):
        if i in self.resid_norms:
            self.resid_norms[i].append(mx.linalg.norm(h[:, 1:].astype(mx.float32), axis=-1).reshape(-1))
        return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layers", default="22", help="comma-separated decoder layer indices")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="stats/q35-4b")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    q = Qwen(a.model)
    tok = q.tok
    stats = RegimeStats(layers, q.inter)

    n_steps = -(-a.tokens // (a.seq_len * a.batch))
    t0 = time.time()
    for step, ids in enumerate(itertools.islice(windows(tok, a.seq_len, a.batch, a.seed), n_steps)):
        q(ids, hook=stats, stop=max(layers))  # nothing past the deepest target layer is needed
        mx.eval(stats.acc, [v[-1] for v in stats.resid_norms.values()])  # keeps each step's graph from piling up
        if step % 10 == 0 or step == n_steps - 1:
            done = (step + 1) * a.seq_len * a.batch
            print(f"{step + 1}/{n_steps} steps, {done:,} tokens, {done / (time.time() - t0):,.0f} tok/s", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for l, s in stats.acc.items():
        pp, nn, pn, np_ = (np.array(s[k]) for k in ("pp_max", "nn_max", "pn_min", "np_min"))
        abs_max = np.maximum.reduce([pp, nn, -pn, -np_])
        w_down = np.array(q.down(l).astype(mx.float32)).T.astype(np.float16)
        resid_q = np.quantile(np.array(mx.concatenate(stats.resid_norms[l])), [0.1, 0.5, 0.9])
        path = f"{a.out}_L{l}.npz"
        np.savez(path, pp_max=pp, nn_max=nn, pn_min=pn, np_min=np_, abs_max=abs_max, w_down=w_down,
                 resid_norm_q10_50_90=resid_q, model=a.model, layer=l, tokens=n_steps * a.seq_len * a.batch)
        print(f"L{l}: residual norm leaving the layer p10/p50/p90 = {resid_q.round(1)}")
        dn_top = (nn > pp).mean()
        print(f"L{l}: abs_max median {np.median(abs_max):.3f} [{abs_max.min():.3f}, {abs_max.max():.3f}]; "
              f"(-,-) regime holds the positive max for {dn_top:.1%} of neurons -> {path}")


if __name__ == "__main__":
    main()
