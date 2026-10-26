"""Downstream state of every rollout in a sweep run, on MLX (Apple silicon): the MLP input (post-attention-norm hidden
state, the vector the gate and up projections read) at the last prompt token, at the run's capture layers, with the
rollout's steering push applied at every prompt position. capture_prefill.py on sweep/qwen_mlx.py: the same inputs,
control directions, push, batching (one prefill per rollout, batched by prompt) and output files.

    uv run python sweep/capture_prefill_mlx.py runs/q35-4b-L22-full.jsonl runs/q35-4b-L13-r384.jsonl ...

Writes captures/<run>.mlp_in.npy, float16 [rows, capture layers, hidden], row i = line i of the jsonl, and
captures/<run>.mlp_in.json with the capture layers. Neuron activations follow as silu(x @ W_gate.T) * (x @ W_up.T).
"""
import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np

from qwen_mlx import Hook, Qwen


class Taps(Hook):
    """One batch's push and captures. qwen_mlx calls a layer's mlp_in before its mlp_out, so a capture at the steered
    layer itself is taken before that layer's push, as the torch pre-hook / forward-hook pair does."""

    def __init__(self, layer, vec, cap_layers):
        self.layer, self.vec = layer, vec  # vec: fp32 [rows, hidden]
        self.caps = dict.fromkeys(cap_layers)

    def mlp_in(self, i, x):
        if i in self.caps:
            self.caps[i] = x[:, -1].astype(mx.float32)
        return x

    def mlp_out(self, i, y):
        if i == self.layer:  # cast first: an fp32 array would promote the rest of the pass to fp32
            return y + self.vec[:, None, :].astype(y.dtype)
        return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0, help="the sweep's --seed (random-control directions)")
    ap.add_argument("--out-dir", default="captures")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    q = Qwen(a.model, dtype=mx.bfloat16)  # bf16, as the torch original and the sweep run

    for run in a.runs:
        meta = json.load(open(run.replace(".jsonl", ".meta.json")))
        layer, cap_layers = meta["layer"], meta["capture_layers"]
        name = os.path.basename(run).replace(".jsonl", "")
        out_path = os.path.join(a.out_dir, f"{name}.mlp_in.npy")
        if os.path.exists(out_path):
            print(f"{name}: exists, skipping", flush=True)
            continue
        w_down = np.array(q.down(layer).T.astype(mx.float32))  # [N, hidden], the model's own down_proj columns
        rows = [json.loads(l) for l in open(run)]
        # the sweep drew --random-controls directions; a run cut short holds rows for fewer, and drawing only those
        # would give other directions (the ref_neuron check below would still pass), so take the count from the meta
        n_ctrl = meta.get("random_controls", max([r["neuron"] for r in rows if r["kind"] == "random"], default=-1) + 1)
        rng = np.random.default_rng(a.seed)
        ctrl_ref = rng.integers(len(w_down), size=n_ctrl)
        ctrl_vecs = rng.standard_normal((n_ctrl, w_down.shape[1])).astype(np.float32)
        ctrl_vecs *= (np.linalg.norm(w_down[ctrl_ref], axis=1) / np.linalg.norm(ctrl_vecs, axis=1))[:, None]
        for r in rows:  # the sweep's own record of which neuron a control borrowed its norm from
            if r["kind"] == "random":
                assert r["ref_neuron"] == int(ctrl_ref[r["neuron"]]), "control directions don't match the sweep's seed"

        def vec(r):
            if r["kind"] == "neuron":
                return w_down[r["neuron"]] * r["add"]
            if r["kind"] == "random":
                return ctrl_vecs[r["neuron"]] * r["add"]
            return np.zeros(w_down.shape[1], np.float32)

        out = np.lib.format.open_memmap(out_path + ".tmp", mode="w+", dtype=np.float16,
                                        shape=(len(rows), len(cap_layers), w_down.shape[1]))
        t0 = time.time()
        by_prompt = {}
        for i, r in enumerate(rows):
            by_prompt.setdefault(r["prompt"], []).append(i)
        for p, idx in by_prompt.items():
            for s in range(0, len(idx), a.batch):
                chunk = idx[s:s + a.batch]
                taps = Taps(layer, mx.array(np.stack([vec(rows[i]) for i in chunk])), cap_layers)
                ids = mx.array([meta["prompt_ids"][p]] * len(chunk), dtype=mx.int32)
                q(ids, hook=taps, stop=max(cap_layers))  # lazy: evaluating the captures computes only what they need
                out[chunk] = np.array(mx.stack([taps.caps[c] for c in cap_layers], 1)).astype(np.float16)
        out.flush()
        del out
        os.replace(out_path + ".tmp", out_path)
        json.dump({"run": name, "capture_layers": cap_layers, "hook": "mlp_in", "position": "last_prompt",
                   "rows": len(rows)}, open(out_path.replace(".npy", ".json"), "w"))
        mx.clear_cache()
        print(f"{name}: {len(rows)} rows, layers {cap_layers}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
