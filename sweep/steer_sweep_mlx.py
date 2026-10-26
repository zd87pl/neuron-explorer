"""steer_sweep.py on MLX (Apple silicon): steer every neuron of one MLP layer, one greedy rollout per (prompt, dose),
and keep everything a rollout produces. Same jobs, random-direction controls, resume rule and output files as the vLLM
sweep, so rank.py, the explorer and the capture readers take its runs unchanged; steer_sweep.py's docstring explains
the designs and dose units.

The push is the vLLM fork's `mlp_out` hook: add * direction joins the steered layer's MLP output at every position,
prompt and generated (direction = the stats file's W_down row for a neuron, the control vector for a random direction,
nothing for a baseline), cast to the model dtype. Decoding follows vLLM's output conventions: greedy on fp32 logits,
stop on the tokenizer's eos or any eos id in the checkpoint's generation_config.json, the stop token kept in token_ids
(with its logprob and top-k) but not in text, logprobs from a log-softmax of fp32 logits.

Rows that share a prompt are prefilled together (no padding; each row carries its own push), then all rows decode as one
batch. Finished rows leave the batch and new prefills join it once enough slots are free (mlx-lm's batch caches,
left-padded where prompts differ in length); each chunk drains before its records are written in job order.

Captures replace the fork's filesystem capture: the MLP input (post-attention-norm hidden state) at the last prompt
token at the capture layers, taken during each prefill, as captures/<run>.mlp_in.npy, float16 [rows, capture layers,
hidden], row i = line i of the jsonl, plus captures/<run>.mlp_in.json: the files sweep/capture_prefill.py writes. The
.npy grows in place a chunk at a time and is a valid array between chunks.

    python sweep/steer_sweep_mlx.py --stats stats/q35-4b_L22.npz --neurons 2742 --run smoke-2742
    python sweep/steer_sweep_mlx.py --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel --neurons all \
        --random-controls 256 --run q35-4b-L22

Writes runs/<run>.jsonl, one record per rollout (layer, kind, neuron, ref_neuron, prompt, dose, add, rel_push, text,
finish, n_gen, token_ids, logprob, top_ids, top_lps, capture_id), runs/<run>.meta.json and the captures above.
Rerunning the same command resumes: rollouts already in the jsonl are skipped, a line cut short by a kill is dropped,
and the captures are brought back in line with the jsonl (extra rows cut, missing rows recomputed by prefill).
"""
import argparse
import collections
import json
import os
import shutil
import struct
import time

import mlx.core as mx
import numpy as np
from transformers import AutoConfig, AutoTokenizer

from qwen_mlx import Hook, Qwen, topk
from steer_sweep import DESIGNS, PROMPTS, parse_neurons


class Steer(Hook):
    """Adds each batch row's push [B, hidden] to the steered layer's MLP output at every position; while `caps` is a
    dict, also keeps the MLP input at the last position of each capture layer."""

    def __init__(self, layer):
        self.layer, self.vecs, self.caps = layer, None, None

    def mlp_in(self, i, x):
        if self.caps is not None and i in self.caps:
            self.caps[i] = x[:, -1]
        return x

    def mlp_out(self, i, y):
        return y + self.vecs[:, None] if i == self.layer else y


class Captures:
    """captures/<run>.mlp_in.npy grown in place: float16 [rows, layers, hidden], row i = line i of the jsonl. The header
    is padded to a fixed size so the row count can be rewritten after each append (data first, then the count), so the
    file is always a valid .npy of the rows appended so far; the .json beside it is rewritten to match."""

    HEADER = 128

    def __init__(self, path, run, layers, hidden):
        self.path, self.run, self.layers, self.hidden = path, run, layers, hidden
        self.row_bytes = len(layers) * hidden * 2
        self.rows = 0
        if not os.path.exists(path):
            self._commit(0)
            return
        with open(path, "rb") as f:
            major, _ = np.lib.format.read_magic(f)
            read = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = read(f)
            offset = f.tell()
        js = path[:-len(".npy")] + ".json"
        old = json.load(open(js))["capture_layers"] if os.path.exists(js) else layers
        if dtype != np.float16 or fortran or tuple(shape[1:]) != (len(layers), hidden) or old != layers:
            raise SystemExit(f"{path} holds captures {tuple(shape)} at layers {old}; this run captures layers {layers} "
                             f"of width {hidden}. Pass the run's own --capture-layers, or move the file away.")
        if offset != self.HEADER:  # written by capture_prefill.py: copy under a header with room to grow
            with open(path, "rb") as src, open(path + ".tmp", "wb") as dst:
                dst.write(self._header(shape[0]))
                src.seek(offset)
                shutil.copyfileobj(src, dst)
            os.replace(path + ".tmp", path)
        self.truncate(shape[0])  # also drops bytes a kill left past the last committed count

    def _header(self, rows):
        h = "{'descr': '<f2', 'fortran_order': False, 'shape': (%d, %d, %d), }" % (rows, len(self.layers), self.hidden)
        return b"\x93NUMPY\x01\x00" + struct.pack("<H", self.HEADER - 10) + h.ljust(self.HEADER - 11).encode() + b"\n"

    def _commit(self, rows):
        with open(self.path, "r+b" if os.path.exists(self.path) else "wb") as f:
            f.write(self._header(rows))
        self.rows = rows
        with open(self.path[:-len(".npy")] + ".json", "w") as f:
            json.dump({"run": self.run, "capture_layers": self.layers, "hook": "mlp_in", "position": "last_prompt",
                       "rows": rows}, f)

    def truncate(self, rows):
        self._commit(rows)
        with open(self.path, "r+b") as f:
            f.truncate(self.HEADER + rows * self.row_bytes)

    def append(self, arr):
        with open(self.path, "r+b") as f:
            f.seek(self.HEADER + self.rows * self.row_bytes)
            f.write(np.ascontiguousarray(arr, np.float16).tobytes())
        self._commit(self.rows + len(arr))


def top_logprobs(lp, k, block=256):
    """topk(lp, k) over the vocabulary without sorting it: mx.argpartition is a full sort on the GPU, which for a
    batch of 248k-entry rows costs about as much as the forward pass. The k largest entries sit in the k blocks with the
    largest maxima, so only those k * block candidates are ranked."""
    B, V = lp.shape
    nb = -(-V // block)
    if nb <= k:
        return topk(lp, k)
    if nb * block != V:
        lp = mx.pad(lp, [(0, 0), (0, nb * block - V)], constant_values=-float("inf"))
    blocks = lp.reshape(B, nb, block)
    bidx = topk(blocks.max(-1), k)[1]
    vals, j = topk(mx.take_along_axis(blocks, bidx[..., None], axis=1).reshape(B, k * block), k)
    return vals, mx.take_along_axis(bidx, j // block, axis=-1) * block + j % block


def cache_arrays(cache):
    """The arrays a batch cache carries from step to step (evaluated with each step, so graphs don't grow)."""
    return [x for c in cache for x in ((c.keys, c.values) if hasattr(c, "keys") else c.cache) if x is not None]


class Engine:
    """Steered greedy rollouts and capture prefills on one Qwen, rows batched as the module docstring describes."""

    def __init__(self, q, dtype, layer, prompt_ids, vec, cap_layers, batch, max_tokens, top_k, eos):
        self.q, self.dtype, self.prompt_ids, self.vec, self.cap_layers = q, dtype, prompt_ids, vec, cap_layers
        self.batch, self.max_tokens, self.top_k, self.eos = batch, max_tokens, top_k, eos
        self.steer = Steer(layer)

    def prefill(self, group, cache=None, stop=None):
        """One pass over the prompt `group` shares (jobs or jsonl records: kind, neuron, add, prompt), each row with its
        own push. Returns (last-position logits [n, vocab], or None with stop; captures float16 [n, layers, hidden]
        or None; the rows' pushes)."""
        ids = mx.array([self.prompt_ids[group[0]["prompt"]]] * len(group), dtype=mx.int32)
        vecs = mx.array(np.stack([self.vec(j) for j in group])).astype(self.dtype)
        self.steer.vecs = vecs
        self.steer.caps = dict.fromkeys(self.cap_layers) if self.cap_layers else None
        logits = self.q(ids, cache=cache, hook=self.steer, stop=stop)
        caps = None
        if self.cap_layers:
            caps = mx.stack([self.steer.caps[l] for l in self.cap_layers], 1).astype(mx.float16)
        self.steer.caps = None
        return (None if logits is None else logits[:, -1]), caps, vecs

    def generate(self, jobs):
        """Greedy rollouts for `jobs`: per job a dict (ids, lp, top_ids, top_lps, finish) and its captures (float16
        [layers, hidden] or None), in job order."""
        queue = collections.OrderedDict()  # prompt -> job indices, in job order; prefills take from the front
        for i, j in enumerate(jobs):
            queue.setdefault(j["prompt"], collections.deque()).append(i)
        res = [{"ids": [], "lp": [], "top_ids": [], "top_lps": [], "finish": None} for _ in jobs]
        caps = []  # (job indices, captures) per prefill, read back once the chunk is done
        rows, cache, vecs, logits = [], None, None, None  # rows: job index per batch row, None once it has finished
        min_refill = max(1, self.batch // 8)  # each join copies the whole batch cache, so slots are refilled in groups
        while True:
            n_dead = rows.count(None)
            n_live = len(rows) - n_dead
            refill = bool(queue) and (self.batch - n_live >= min_refill or n_live == 0)
            if n_dead and (refill or n_dead >= max(1, len(rows) // 8)):  # drop finished rows (a copy, so not each one)
                keep = [k for k, r in enumerate(rows) if r is not None]
                rows = [rows[k] for k in keep]
                if keep:
                    self.q.filter_cache(cache, keep)
                    idx = mx.array(keep)
                    vecs, logits = vecs[idx], logits[idx]
                else:
                    cache = vecs = logits = None
            while refill and queue and len(rows) < self.batch:
                p = next(iter(queue))
                group = [queue[p].popleft() for _ in range(min(len(queue[p]), self.batch - len(rows)))]
                if not queue[p]:
                    del queue[p]
                new_cache = self.q.make_cache([0] * len(group))
                new_logits, new_caps, new_vecs = self.prefill([jobs[i] for i in group], new_cache)
                mx.async_eval(new_logits, new_caps, cache_arrays(new_cache))
                caps.append((group, new_caps))
                if cache is None:
                    cache, vecs, logits = new_cache, new_vecs, new_logits
                else:
                    for c, n in zip(cache, new_cache):
                        c.extend(n)
                    vecs, logits = mx.concatenate([vecs, new_vecs]), mx.concatenate([logits, new_logits])
                rows += group
            if not rows:
                break
            logits = logits.astype(mx.float32)
            nxt = mx.argmax(logits, axis=-1)  # vLLM's greedy pick, on fp32 logits
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            chosen = mx.take_along_axis(lp, nxt[:, None], axis=-1)[:, 0]
            top_lp, top_id = top_logprobs(lp, self.top_k) if self.top_k else (None, None)
            # this step's readbacks go in their own eval (one completion event per call), queued before the next step
            # is built and queued, so reading them back below doesn't wait for the next forward pass and the device
            # doesn't wait on the bookkeeping; rows that finish now ride along in that pass and are dropped afterwards
            mx.async_eval(nxt, chosen, top_lp, top_id)
            self.steer.vecs = vecs
            logits = self.q(nxt[:, None].astype(mx.int32), cache=cache, hook=self.steer)[:, -1]
            mx.async_eval(logits, cache_arrays(cache))
            nxt, chosen = nxt.tolist(), chosen.tolist()
            top_lp, top_id = (top_lp.tolist(), top_id.tolist()) if self.top_k else ([[]] * len(rows), [[]] * len(rows))
            for k, j in enumerate(rows):
                if j is None:
                    continue
                r = res[j]
                r["ids"].append(nxt[k])
                r["lp"].append(chosen[k])
                r["top_ids"].append(top_id[k])
                r["top_lps"].append(top_lp[k])
                if len(r["ids"]) >= self.max_tokens:  # vLLM checks the length cap before the stop tokens
                    r["finish"], rows[k] = "length", None
                elif nxt[k] in self.eos:
                    r["finish"], rows[k] = "stop", None
        out = [None] * len(jobs)
        for group, c in caps:
            if c is not None:
                for i, row in zip(group, np.array(c)):
                    out[i] = row
        return res, out


def drop_partial_line(path):
    """Cuts a record a kill left half-written at the end of a jsonl, reading only the file's tail."""
    with open(path, "r+b") as f:
        size = pos = f.seek(0, 2)
        while pos > 0:
            start = max(0, pos - (1 << 16))
            f.seek(start)
            k = f.read(pos - start).rfind(b"\n")
            if k >= 0:
                pos = start + k + 1
                break
            pos = start
        if pos < size:
            print(f"{path}: dropping a partial last line ({size - pos} bytes)", flush=True)
            f.truncate(pos)


def row_bytes(q, dtype, max_len):
    """Cache bytes one batch row holds: KV in the model dtype for the full-attention layers (allocated in 256-token
    steps), fp32 recurrent state plus the conv window for the linear-attention layers."""
    a, item = q.args, dtype.size
    n_full = q.layer_types.count("full_attention")
    kv = 2 * a.num_key_value_heads * a.head_dim * item * (-(-max_len // 256) * 256)
    conv_dim = 2 * a.linear_num_key_heads * a.linear_key_head_dim + a.linear_num_value_heads * a.linear_value_head_dim
    lin = (a.linear_num_value_heads * a.linear_value_head_dim * a.linear_key_head_dim * 4
           + (a.linear_conv_kernel_dim - 1) * conv_dim * item)
    return n_full * kv + (q.n_layers - n_full) * lin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--stats", required=True, help="npz from neuron_stats.py for the layer to steer")
    ap.add_argument("--neurons", default="all", help='"all", "random:128", "2742", "0-99", or "1,5,9-12"')
    ap.add_argument("--design", default="main", choices=sorted(DESIGNS))
    ap.add_argument("--dose-unit", default="absmax", choices=["absmax", "rel"], help="absmax: dose x the neuron's "
                    "abs_max; rel: dose x the median residual norm, the same push size for every neuron")
    ap.add_argument("--run", required=True, help="run name: output file and capture file name")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--max-tokens", type=int, default=256, help="steered rollouts often loop; this bounds them")
    ap.add_argument("--top-k", type=int, default=5, help="alternative tokens kept per generated step")
    ap.add_argument("--capture-layers", default="+0,+1,+4,+8", help="mlp_in layers to capture, relative to the "
                    "steered layer; empty string turns capture off")
    ap.add_argument("--capture-root", default="captures", help="where <run>.mlp_in.npy and .json go")
    ap.add_argument("--random-controls", type=int, default=256, help="random-direction controls spread through the run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4000, help="rollouts per chunk (one jsonl flush each; the batch "
                    "drains at the end of a chunk)")
    ap.add_argument("--batch", type=int, default=256, help="rows decoded together. Each row keeps bf16 KV for the 8 "
                    "full-attention layers and fp32 recurrent state for the 24 linear-attention layers, about 70 MB a "
                    "row on Qwen3.5-4B at --max-tokens 256 (the figure for the loaded model is printed at start), so "
                    "256 rows hold ~18 GB beside ~9 GB of weights")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="bfloat16: the checkpoint as "
                    "stored (bf16 for Qwen3.5), as the vLLM sweep ran; float32 for checks against torch fp32")
    a = ap.parse_args()

    st = np.load(a.stats)
    layer, abs_max, w_down = int(st["layer"]), st["abs_max"], st["w_down"]
    neurons = parse_neurons(a.neurons, len(abs_max), a.seed)
    design = DESIGNS[a.design]
    resid_p50 = float(st["resid_norm_q10_50_90"][1])
    down_norm = np.linalg.norm(w_down.astype(np.float32), axis=1)
    text_cfg = AutoConfig.from_pretrained(a.model).get_text_config()
    assert not getattr(text_cfg, "layer_scale", False), "mlp_out sits after an FFN layer scale on this model, so " \
        "adding W_down[:, n] there is no longer a pure push on neuron n"
    n_layers = text_cfg.num_hidden_layers
    cap_layers = sorted({layer + int(x) for x in a.capture_layers.split(",") if x} & set(range(n_layers)))

    tok = AutoTokenizer.from_pretrained(a.model)
    prompt_ids = {
        k: tok.apply_chat_template([{"role": "user", "content": v}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=True)
        for k, v in PROMPTS.items()
    }
    prompt_ids = {k: list(v["input_ids"] if hasattr(v, "keys") else v) for k, v in prompt_ids.items()}

    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, f"{a.run}.jsonl")
    key = lambda j: (j["kind"], j["neuron"], j["prompt"], j["dose"])
    done, earlier = set(), []  # earlier: per jsonl line, what a capture rebuild needs (the rest stays on disk)
    if os.path.exists(out_path):
        drop_partial_line(out_path)
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done.add(key(r))
                earlier.append({k: r[k] for k in ("kind", "neuron", "ref_neuron", "prompt", "add")})
    cap = None
    if cap_layers:
        os.makedirs(a.capture_root, exist_ok=True)
        cap = Captures(os.path.join(a.capture_root, f"{a.run}.mlp_in.npy"), a.run, cap_layers, w_down.shape[1])
        if cap.rows > len(earlier):  # captures of a chunk whose records never reached the jsonl
            cap.truncate(len(earlier))

    with open(os.path.join(a.out_dir, f"{a.run}.meta.json"), "w") as f:
        json.dump({"model": a.model, "layer": layer, "stats": os.path.abspath(a.stats), "prompts": PROMPTS,
                   "design_name": a.design, "design": design, "resid_norm_p50": resid_p50, "dose_unit": a.dose_unit,
                   "hook": "mlp_out", "max_tokens": a.max_tokens, "top_k": a.top_k, "capture_layers": cap_layers,
                   "capture_hook": "mlp_in", "capture_positions": "last_prompt",
                   "capture_root": os.path.abspath(a.capture_root),
                   "capture_file": os.path.abspath(cap.path) if cap else None,
                   "random_controls": a.random_controls, "seed": a.seed,
                   "engine": "mlx", "dtype": a.dtype, "prompt_ids": prompt_ids}, f, indent=1)

    # Random-direction controls: neuron j's down column rotated to a random direction (same norm, same abs_max).
    # They answer "how often does a theme appear when the steered direction means nothing?"
    rng = np.random.default_rng(a.seed)
    ctrl_ref = rng.integers(len(abs_max), size=a.random_controls)
    ctrl_vecs = rng.standard_normal((a.random_controls, w_down.shape[1])).astype(np.float32)
    ctrl_vecs *= (np.linalg.norm(w_down[ctrl_ref].astype(np.float32), axis=1)
                  / np.linalg.norm(ctrl_vecs, axis=1))[:, None]

    units = [("neuron", n) for n in neurons]
    for k in range(a.random_controls):  # spread evenly so any prefix of the run has its share of controls
        units.insert(round((k + 1) * len(units) / (a.random_controls + 1)), ("random", k))

    jobs = [{"kind": "baseline", "neuron": -1, "prompt": p, "dose": 0.0} for p in PROMPTS]
    jobs += [{"kind": kind, "neuron": n, "prompt": p, "dose": d} for kind, n in units for p, d in design]
    jobs = [j for j in jobs if key(j) not in done]
    print(f"L{layer}: {len(neurons)} neurons + {a.random_controls} random controls, {len(jobs)} rollouts to run "
          f"({len(done)} already done)", flush=True)
    missing = earlier[cap.rows:] if cap else []
    del earlier
    if not jobs and not missing:
        return

    # bf16 checkpoints load as stored, the way the explorer's runtime_mlx loads them (checked against transformers)
    q = Qwen(a.model, dtype=mx.float32 if a.dtype == "float32" else None)
    dtype = q.dtype
    if dtype != getattr(mx, a.dtype):
        print(f"note: this checkpoint computes in {dtype}, not {a.dtype}", flush=True)
    assert (q.n_layers, q.inter, q.hidden) == (n_layers, *w_down.shape), "the stats file is for another model"
    eos = {tok.eos_token_id}  # vLLM stops on the tokenizer's eos and on the generation config's eos ids
    gen_cfg = q.path / "generation_config.json"
    if gen_cfg.exists():
        e = json.load(open(gen_cfg)).get("eos_token_id")
        eos |= set(e if isinstance(e, list) else [e])
    eos.discard(None)
    max_len = max(map(len, prompt_ids.values())) + a.max_tokens
    mb = row_bytes(q, dtype, max_len) / 1e6
    print(f"{a.model} on MLX ({mx.default_device()}, {dtype}): {mb:.0f} MB of cache a row, about "
          f"{mb * a.batch / 1e3:.1f} GB at --batch {a.batch}; stop ids {sorted(eos)}", flush=True)

    def vec(job):  # the push: add * direction, fp32 like the vector sent to the fork
        if job["kind"] == "neuron":
            return w_down[job["neuron"]].astype(np.float32) * job["add"]
        if job["kind"] == "random":
            return ctrl_vecs[job["neuron"]] * job["add"]
        return np.zeros(w_down.shape[1], np.float32)

    def params(job):
        kind, n, p, d = job["kind"], job["neuron"], job["prompt"], job["dose"]
        ref = n if kind == "neuron" else int(ctrl_ref[n]) if kind == "random" else None
        job["ref_neuron"] = ref
        if ref is None:
            job["add"] = 0.0
        elif a.dose_unit == "rel":
            job["add"] = float(d * resid_p50 / down_norm[ref])
        else:
            job["add"] = float(d * abs_max[ref])
        job["rel_push"] = abs(job["add"]) * float(down_norm[ref]) / resid_p50 if ref is not None else 0.0
        job["capture_id"] = f"L{layer}_{kind}{n}_{p}_{d:+g}".replace("+", "p").replace("-", "m")

    eng = Engine(q, dtype, layer, prompt_ids, vec, cap_layers, a.batch, a.max_tokens, a.top_k, eos)

    if missing:  # jsonl rows without captures (capture was off, or the file was lost): rebuild them by prefill
        t0 = time.time()
        for r in missing:
            assert r["kind"] != "random" or r["ref_neuron"] == int(ctrl_ref[r["neuron"]]), \
                "control directions don't match this --seed / --random-controls"
        for s in range(0, len(missing), a.chunk):
            part = missing[s:s + a.chunk]
            out = np.zeros((len(part), len(cap_layers), w_down.shape[1]), np.float16)
            by_prompt = collections.defaultdict(list)
            for i, r in enumerate(part):
                by_prompt[r["prompt"]].append(i)
            for idx in by_prompt.values():
                for b in range(0, len(idx), a.batch):
                    sub = idx[b:b + a.batch]
                    out[sub] = np.array(eng.prefill([part[i] for i in sub], stop=max(cap_layers))[1])
            cap.append(out)
        print(f"captures for {len(missing)} earlier rollouts rebuilt by prefill, {time.time() - t0:.0f}s", flush=True)

    with open(out_path, "a") as f:
        for c in range(0, len(jobs), a.chunk):
            chunk = jobs[c : c + a.chunk]
            for job in chunk:
                params(job)
            t0 = time.time()
            res, caps = eng.generate(chunk)
            if cap:  # captures before records: a kill in between leaves extra capture rows, cut on resume
                cap.append(np.stack(caps))
            n_tok = 0
            for job, r in zip(chunk, res):
                ids = r["ids"]
                text = tok.decode(ids[:-1] if r["finish"] == "stop" else ids, skip_special_tokens=True)
                rec = {
                    "layer": layer, "kind": job["kind"], "neuron": job["neuron"], "ref_neuron": job["ref_neuron"],
                    "prompt": job["prompt"], "dose": job["dose"], "add": job["add"], "rel_push": job["rel_push"],
                    # vLLM's incremental detokenizer holds back a character still incomplete at the end
                    "text": text.rstrip("\ufffd"), "finish": r["finish"], "n_gen": len(ids), "token_ids": ids,
                    "logprob": [round(x, 4) for x in r["lp"]], "top_ids": r["top_ids"],
                    "top_lps": [[round(x, 4) for x in row] for row in r["top_lps"]], "capture_id": job["capture_id"],
                }
                n_tok += rec["n_gen"]
                f.write(json.dumps(rec) + "\n")
            f.flush()
            mx.clear_cache()
            dt = time.time() - t0
            print(f"{min(c + a.chunk, len(jobs))}/{len(jobs)} rollouts, {n_tok / dt:,.0f} gen tok/s, "
                  f"{len(chunk) / dt:.1f} rollouts/s, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
