"""Live Qwen3.5 model for the explorer on MLX (Apple silicon): steered generation (add or multiply one neuron, per batch
row), per-token neuron activations on arbitrary text, and logit-lens tokens for a neuron's down column.

The hook sits on the input of a layer's down_proj, where the neuron activations silu(gate) * up live, so add,
multiply, and reading are all exact. Add doses are in units of the median residual norm leaving the layer, the same
unit as the sweep's `--dose-unit rel`. The model and its hooks are sweep/qwen_mlx.py; runtime.py is the same runtime on
PyTorch, which the server uses elsewhere or with EXPLORER_BACKEND=torch.
"""
import json
import os
import sqlite3
import sys
import threading

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweep"))
from qwen_mlx import Hook, Qwen, topk  # noqa: E402

MODEL = os.environ.get("EXPLORER_MODEL", "Qwen/Qwen3.5-4B")
NO_MODEL = bool(os.environ.get("EXPLORER_NO_MODEL"))  # cache-only: never download or load weights
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


class _Taps(Hook):
    """The explorer's interventions and reads for one pass, on the activations (input of down_proj) and the residual."""

    def __init__(self, rt):
        self.rt = rt

    def act(self, i, h, gate, up):
        rt = self.rt
        if rt._capture_full is not None:
            rt._capture_full[i] = h[0].astype(mx.float32)  # [tokens, inter]
        if rt._capture_all is not None:
            a = h[:, rt._capture_from:].astype(mx.float32).reshape(-1, h.shape[-1])  # [tokens, inter]
            rt._capture_all[i] = (a.max(0), a.argmax(0), a.min(0), a.argmin(0))
        if rt._read is not None and rt._read[0] == i:
            rt._seen["act"] = h[0, :, rt._read[1]].astype(mx.float32)
        for _, n, mode, vals in [p for p in rt._push if p[0] == i] if rt._push else []:
            v = vals.astype(h.dtype)[:, None, None]
            if n is None:  # the whole layer's MLP (multiply only)
                h = h * v
            else:
                on = mx.arange(h.shape[-1]) == n
                h = mx.where(on, h * v if mode == "mul" else h + v, h)
        return h

    def resid(self, i, h):
        if self.rt._seen.get("want_resid") == i:
            self.rt._seen["resid"] = mx.linalg.norm(h[0, 1:].astype(mx.float32), axis=-1)
        return h


class Runtime:
    def __init__(self, stats_dir):
        self.stats_dir = stats_dir
        os.makedirs(CACHE, exist_ok=True)
        self._db = sqlite3.connect(os.path.join(CACHE, "connections.sqlite"), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS con (key TEXT PRIMARY KEY, val TEXT)")
        self._db_lock = threading.Lock()
        self._lens_idx = None
        self._con_idx = None
        self.lock = threading.RLock()  # re-entrant: a push's dose unit may need a forward pass of its own
        self.q = self.tok = None
        self.status = "idle"
        self.resid_p50 = {}
        self._push = None  # [(layer, neuron, mode, per-row values array)], any number of neurons at once (neuron None = all)
        self._read = None  # (layer, neuron) while reading
        self._seen = {}
        self._con_cache = {}
        self._capture_all = None
        self._capture_from = 0
        self._capture_full = None
        self._lens_cache = {}
        self._taps = _Taps(self)

    # ---------- loading ----------
    def ensure_loaded(self):
        if self.q is not None:
            return
        if NO_MODEL:
            raise RuntimeError("live model disabled (EXPLORER_NO_MODEL)")
        with self.lock:
            if self.q is not None:
                return
            self.status = "loading"
            q = Qwen(MODEL)
            self.tok = q.tok
            self.dev = str(mx.default_device())
            self.q = q
            self.status = "ready"

    def n_heads(self, layer):
        self.ensure_loaded()
        return self.q.n_heads(layer)

    def _forward(self, ids, cache=None, stop=None, heads=None):
        """One pass with this runtime's taps; heads = [(layer, head, scale)] scaled for the pass."""
        self.q.heads = {}
        for layer, head, scale in heads or []:
            self.q.heads.setdefault(layer, []).append((head, scale))
        try:
            return self.q(ids, cache=cache, hook=self._taps, stop=stop)
        finally:
            self.q.heads = {}

    def layer_resid_p50(self, layer):
        if layer in self.resid_p50:
            return self.resid_p50[layer]
        path = os.path.join(self.stats_dir, f"q35-4b_L{layer}.npz")
        if os.path.exists(path):
            v = float(np.load(path)["resid_norm_q10_50_90"][1])
        else:
            with self.lock:
                ids = self.tok(REFERENCE_TEXT).input_ids
                self._seen["want_resid"] = layer
                try:
                    self._forward([ids], stop=layer)
                    r = np.sort(np.array(self._seen.pop("resid")))
                    v = float(r[(len(r) - 1) // 2])  # the lower middle value, as torch.median gives
                finally:
                    self._seen.pop("want_resid", None)
        self.resid_p50[layer] = v
        return v

    def down_col(self, layer, n):
        return self.q.down(layer)[:, n].astype(mx.float32)

    # ---------- steering ----------
    def chat_ids(self, prompt):
        return self.messages_ids([{"role": "user", "content": prompt}])

    def messages_ids(self, messages):
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        return list(enc["input_ids"] if hasattr(enc, "keys") else enc)

    def steer_stream(self, layer, n, rows_messages, dose, mode, max_new=200, cancel=None):
        """Greedy-decodes three conversations in one batch: row 0 pushed up, row 1 unsteered, row 2 pushed down.
        rows_messages holds each row's own message list (the rows can differ in length; they are left-padded).
        Yields (row, text_delta), then (None, None)."""
        yield from self.steer_multi([(layer, n, dose, mode)], rows_messages, max_new, cancel)

    def steer_multi(self, pushes, rows_messages, max_new=200, cancel=None):
        """steer_stream for any number of neurons at once: pushes = [(layer, n, dose, mode)], all applied together.
        Each is reversed in row 2 the same way a single neuron is (add: -dose; mul: x -dose)."""
        self.ensure_loaded()
        with self.lock:
            if cancel is not None and cancel.is_set():
                yield None, None
                return
            try:
                push = [self._push_entry(layer, n, dose, mode, (1, 0, -1)) for layer, n, dose, mode in pushes]
                seqs = [self.messages_ids(m) for m in rows_messages]
                T = max(map(len, seqs))
                pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
                ids = mx.array([[pad] * (T - len(q)) + q for q in seqs], dtype=mx.int32)
                cache = self.q.make_cache([T - len(q) for q in seqs])
                eos = self.tok.convert_tokens_to_ids("<|im_end|>")
                done, out_ids, sent = [False] * 3, [[] for _ in range(3)], [""] * 3
                self._push = push
                logits = self._forward(ids, cache)
                for _ in range(max_new):
                    if cancel is not None and cancel.is_set():
                        break
                    nxt = logits[:, -1].argmax(-1)
                    toks = nxt.tolist()
                    for r in range(3):
                        if done[r]:
                            continue
                        t = toks[r]
                        if t == eos:
                            done[r] = True
                            continue
                        out_ids[r].append(t)
                        text = self.tok.decode(out_ids[r], skip_special_tokens=True)
                        if not text.endswith("\ufffd"):
                            yield r, text[len(sent[r]):]
                            sent[r] = text
                    if all(done):
                        break
                    logits = self._forward(nxt[:, None], cache)
            finally:
                self._push = None
        yield None, None

    def token_units(self, ids):
        """Groups token ids into displayable units: byte-level pieces of one multi-byte character (emoji, CJK split
        across tokens) join the token that completes it. Returns [(text, [token indices])]."""
        units, pending, prev = [], [], ""
        for i in range(len(ids)):
            full = self.tok.decode(ids[: i + 1])
            pending.append(i)
            if full.endswith("\ufffd") and i < len(ids) - 1:
                continue
            units.append((full[len(prev):] if full.startswith(prev) else self.tok.decode([ids[j] for j in pending]), pending))
            prev, pending = full, []
        # keep emoji sequences whole: joiners, variation selectors and skin tones attach to the unit before them
        glue = {"\u200d", "\ufe0f", "\ufe0e", "\u20e3"} | {chr(c) for c in range(0x1F3FB, 0x1F400)}
        merged = []
        for text, idx in units:
            if merged and text and (text[0] in glue or merged[-1][0].endswith("\u200d")):
                merged[-1] = (merged[-1][0] + text, merged[-1][1] + idx)
            else:
                merged.append((text, idx))
        return merged

    # ---------- reading ----------
    def _push_entry(self, layer, n, dose, mode, rows):
        """One neuron's push for the down_proj hook, per batch row: rows holds +1 (pushed), 0 (left alone) or -1
        (pushed the other way). add doses are in units of the layer's median residual norm along the neuron's down
        column; mul doses multiply the activation (left alone = x1, the other way = x -dose)."""
        if n is None and mode != "mul":
            raise ValueError("a whole-layer MLP push must be mode 'mul' (e.g. dose 0 to ablate)")
        if mode == "mul":
            vals = [dose if r > 0 else 1.0 if r == 0 else -dose for r in rows]
        else:
            unit = self.layer_resid_p50(layer) / float(mx.linalg.norm(self.down_col(layer, n)))
            vals = [r * dose * unit for r in rows]
        return (layer, n, mode, mx.array(vals, dtype=mx.float32))

    def read(self, layer, n, text, chat=True, pushes=None, heads=None):
        """Per-token activation of neuron n at `layer` on `text` (as a user turn when chat=True), optionally with other
        neurons pushed at the same time: pushes = [(layer, n, dose, mode)], the same units as steering (n None = the
        layer's whole MLP, mul only), and attention heads scaled: heads = [(layer, head, scale)], head None = all heads,
        scale 0 = ablated. Only earlier layers can move the read (a same-layer MLP push lands after the capture; a
        same-layer head change lands before it)."""
        self.ensure_loaded()
        with self.lock:
            if chat:
                ids = self.chat_ids(text)
            else:
                ids = self.tok(text).input_ids
            try:
                self._push = [self._push_entry(pl, pn, d, m, (1,)) for pl, pn, d, m in pushes or []] or None
                self._read = (layer, n)
                self._forward([ids], stop=layer, heads=heads)
                act = np.array(self._seen.pop("act")).tolist()
            finally:
                self._read = None
                self._push = None
        units = []
        for text, idx in self.token_units(ids):
            vals = [act[i] for i in idx]
            units.append({"text": text, "act": max(vals, key=abs), "pieces": vals})
        return {"units": units}

    def next_logprobs(self, text, chat=True, prefix="", pushes=None, heads=None):
        """Log-probabilities of the next token after `text` (as a user turn when chat=True, so the next token opens the
        reply) followed by `prefix`, as a float32 numpy array [vocab]. With pushes / heads (same units as read()) it
        also returns the same distribution under that intervention: (base, intervened or None)."""
        self.ensure_loaded()
        ids = self.chat_ids(text) if chat else self.tok(text).input_ids
        if prefix:
            ids = ids + self.tok(prefix, add_special_tokens=False).input_ids

        def run(entries, hd):
            with self.lock:
                try:
                    self._push = [self._push_entry(pl, pn, d, m, (1,)) for pl, pn, d, m in entries] or None
                    logits = self._forward([ids], heads=hd)
                    lp = logits[0, -1].astype(mx.float32)
                    return np.array(lp - mx.logsumexp(lp))
                finally:
                    self._push = None

        base = run([], None)
        if not pushes and not heads:
            return base, None
        return base, run(pushes or [], list(heads) if heads else None)

    def con_index(self):
        """Precomputed connections (build_connections.py), loaded once."""
        if self._con_idx is None:
            path = os.path.join(CACHE, "connections.npz")
            if os.path.exists(path):
                z = np.load(path)
                self._con_idx = {key: z[key] for key in z.files}
                self._con_idx["k"] = int(z["k"])
            else:
                self._con_idx = False
        return self._con_idx or None

    # ---------- logit lens ----------
    def lens_index(self):
        if self._lens_idx is None:
            path = os.path.join(CACHE, "lens_index.npz")
            if os.path.exists(path):
                z = np.load(path)
                self._lens_idx = {"up": z["up"], "down": z["down"]}
        return self._lens_idx

    def steer_index(self):
        if getattr(self, "_steer_idx", None) is None:
            path = os.path.join(CACHE, "steer_index.npz")
            self._steer_idx = dict(np.load(path)) if os.path.exists(path) else False
        return self._steer_idx or None

    def _lens_logits(self, D):
        """Logits the final norm and lm_head give each row of D [rows, hidden] (model dtype), float32 [rows, vocab]."""
        return self.q.logits(self.q.final_norm(D[:, None]))[:, 0].astype(mx.float32)

    def lens(self, layer, n, k=16):
        idx = self.lens_index()
        if idx is not None and k <= idx["up"].shape[-1]:
            v = self.vocab_strings()
            out = {"up": [v[t] for t in idx["up"][layer, n, :k]], "down": [v[t] for t in idx["down"][layer, n, :k]]}
            st = self.steer_index()
            if st is not None:
                out.update(steer_up=[v[t] for t in st["up"][layer, n, :k]], steer_down=[v[t] for t in st["down"][layer, n, :k]],
                           kl_up=round(float(st["up_kl"][layer, n]), 3), kl_down=round(float(st["down_kl"][layer, n]), 3))
            return out
        self.ensure_loaded()
        if (layer, n, k) in self._lens_cache:
            return self._lens_cache[(layer, n, k)]
        with self.lock:
            d = self.q.down(layer)[:, n]
            logits = self._lens_logits(mx.stack([d, -d]))
            out = {name: [self.tok.decode([t]) for t in topk(logits[i], k)[1].tolist()]
                   for i, name in enumerate(("up", "down"))}
        self._lens_cache[(layer, n, k)] = out
        return out

    # ---------- connections ----------
    def connections(self, layer, n, k=24, lens_k=5):
        """Direct-path weights between neuron (layer, n) and neurons in other layers, ignoring everything between.
        Downstream B in layer j > layer: g = W_gate_j[B] . ((1 + w_j) * d_n), u = W_up_j[B] . (...), where d_n is this
        neuron's down column and w_j the pre-MLP norm weight. Upstream A in layer i < layer: the same with A's down
        column against this neuron's gate and up rows. Score = |g * u| z-scored within each layer; sign = sign(g * u).
        Each returned neuron carries its own logit-lens tokens."""
        key = (layer, n, k, lens_k)
        if key in self._con_cache:
            return self._con_cache[key]
        ci = self.con_index()
        if ci is not None and k <= ci["k"]:
            out = {}
            for side in ("down", "up"):
                z = ci[f"{side}_z"][layer, n]
                out[side] = [{"layer": int(ci[f"{side}_layer"][layer, n, i]), "n": int(ci[f"{side}_n"][layer, n, i]),
                              "z": round(float(z[i]), 2), "sign": int(ci[f"{side}_sign"][layer, n, i])}
                             for i in range(k) if z[i] > -1e4]
            idx = self.steer_index() or self.lens_index()  # linked neurons show their steer lens when there is one
            if lens_k and idx is not None:
                v = self.vocab_strings()
                lg = self.lens_index()
                for c in out["down"] + out["up"]:
                    c["lens"] = [v[t] for t in idx["up"][c["layer"], c["n"], :lens_k]]
                    c["lens_neg"] = [v[t] for t in idx["down"][c["layer"], c["n"], :lens_k]]
                    if lg is not None and lg is not idx:  # the logit lens too, shown faintly under the steer lens
                        c["logit"] = [v[t] for t in lg["up"][c["layer"], c["n"], :lens_k]]
                        c["logit_neg"] = [v[t] for t in lg["down"][c["layer"], c["n"], :lens_k]]
            return out
        dkey = f"{layer}:{n}:{k}:{lens_k}"
        with self._db_lock:
            row = self._db.execute("SELECT val FROM con WHERE key = ?", (dkey,)).fetchone()
        if row:
            out = json.loads(row[0])
            self._con_cache[key] = out
            return out
        self.ensure_loaded()
        with self.lock:
            q = self.q
            dt = q.dtype
            out = {"down": [], "up": []}
            g_me, u_me, d_me = q.mlp_weights(layer)
            d = d_me[:, n]
            scale_me = q.norm_scale(layer).astype(dt)  # (1 + w) of this layer's pre-MLP norm, in the weights' dtype
            r_g = g_me[n] * scale_me
            r_u = u_me[n] * scale_me
            for j in range(q.n_layers):
                if j == layer:
                    continue
                G, U, W = q.mlp_weights(j)
                if j > layer:
                    v = d * q.norm_scale(j).astype(dt)
                    g, u = (G @ v).astype(mx.float32), (U @ v).astype(mx.float32)
                    side = "down"
                else:
                    g, u = (r_g @ W).astype(mx.float32), (r_u @ W).astype(mx.float32)
                    side = "up"
                prod = g * u
                mag = mx.abs(prod)
                z = (mag - mag.mean()) / (mx.sqrt(mx.var(mag, ddof=1)) + 1e-8)
                vals, idx = topk(z, min(k, z.size))
                sgn = mx.take(prod, idx) > 0
                out[side] += [{"layer": j, "n": int(i), "z": round(float(sc), 2), "sign": 1 if s else -1}
                              for sc, i, s in zip(vals.tolist(), idx.tolist(), sgn.tolist())]
            for side in out:
                out[side] = sorted(out[side], key=lambda c: -c["z"])[:k]
            every = out["down"] + out["up"]
            idx = self.steer_index() or self.lens_index()
            if every and lens_k and idx is not None and lens_k <= idx["up"].shape[-1]:
                v = self.vocab_strings()
                for c in every:
                    c["lens"] = [v[t] for t in idx["up"][c["layer"], c["n"], :lens_k]]
                    c["lens_neg"] = [v[t] for t in idx["down"][c["layer"], c["n"], :lens_k]]
            elif every and lens_k:
                D = mx.stack([q.down(c["layer"])[:, c["n"]] for c in every])
                logits = self._lens_logits(D)
                ids, ids_neg = topk(logits, lens_k)[1].tolist(), topk(-logits, lens_k)[1].tolist()
                for c, row, row_neg in zip(every, ids, ids_neg):
                    c["lens"] = [self.tok.decode([t]) for t in row]
                    c["lens_neg"] = [self.tok.decode([t]) for t in row_neg]
        if len(self._con_cache) > 2000:
            self._con_cache.clear()
        self._con_cache[key] = out
        with self._db_lock:
            self._db.execute("INSERT OR REPLACE INTO con VALUES (?, ?)", (dkey, json.dumps(out)))
            self._db.commit()
        return out

    # ---------- logit-lens index over every neuron ----------
    def build_lens_index(self, path, k=12, chunk=1024):
        """Top-k tokens each neuron's down column pushes up and down, for every layer. Saved as int32 arrays
        up[L, N, k], down[L, N, k]."""
        self.ensure_loaded()
        L, N = self.q.n_layers, self.q.inter
        up = np.zeros((L, N, k), np.int32)
        down = np.zeros((L, N, k), np.int32)
        for li in range(L):
            self.status = f"indexing lens {li}/{L}"
            with self.lock:
                W = self.q.down(li)  # [hidden, inter]
                for s in range(0, N, chunk):
                    logits = self._lens_logits(W[:, s:s + chunk].T)
                    up[li, s:s + chunk] = np.array(topk(logits, k)[1])
                    down[li, s:s + chunk] = np.array(topk(-logits, k)[1])
            mx.clear_cache()
        np.savez(path, up=up, down=down)
        self.status = "ready"
        return path

    def vocab_strings(self):
        if getattr(self, "_vocab", None) is None:
            path = os.path.join(CACHE, "vocab.json")
            if os.path.exists(path):
                self._vocab = json.load(open(path))
            else:
                if self.tok is None:
                    from transformers import AutoTokenizer
                    self.tok = AutoTokenizer.from_pretrained(MODEL)
                self._vocab = [self.tok.decode([i]) for i in range(len(self.tok))]
                json.dump(self._vocab, open(path, "w"))
        return self._vocab

    # ---------- which neurons fire on a text ----------
    def all_activations(self, ids, skip_first=False):
        """Per layer: (max, argmax, min, argmin) over tokens for every neuron, as numpy arrays, on token ids [T] or
        [B, T]. With skip_first, position 0 (the attention sink) is left out and argmax/argmin count from position 1."""
        self.ensure_loaded()
        with self.lock:
            self._capture_all, self._capture_from = {}, int(skip_first)
            try:
                x = mx.array(ids, dtype=mx.int32)
                self._forward(x if x.ndim == 2 else x[None], stop=self.q.n_layers - 1)
                mx.eval(self._capture_all)
                return {l: tuple(np.array(t) for t in v) for l, v in self._capture_all.items()}
            finally:
                self._capture_all, self._capture_from = None, 0

    def build_act_scale(self, path, tokens=10000, seq=512, seed=0):
        """Each neuron's largest positive and most negative activation over web text (positions >= 1)."""
        from datasets import load_dataset
        self.ensure_loaded()
        L, N = self.q.n_layers, self.q.inter
        pos, neg = np.zeros((L, N), np.float32), np.zeros((L, N), np.float32)
        buf, seen = [], 0
        for d in load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed):
            buf += self.tok(d["text"], add_special_tokens=False).input_ids + [self.tok.eos_token_id]
            while len(buf) >= seq and seen < tokens:
                chunk, buf = buf[:seq], buf[seq:]
                self.status = f"measuring neuron ranges {seen // 1000}k/{tokens // 1000}k tokens"
                acts = self.all_activations(chunk, skip_first=True)
                for l, (mx_, _, mn, _) in acts.items():
                    pos[l] = np.maximum(pos[l], mx_); neg[l] = np.minimum(neg[l], mn)
                seen += seq
            if seen >= tokens:
                break
        np.savez(path, pos=pos, neg=neg, tokens=seen)
        self.status = "ready"
        return path

    def full_activations(self, ids):
        """Every neuron's activation at every token: float32 array [layers, tokens, inter]."""
        self.ensure_loaded()
        with self.lock:
            self._capture_full = {}
            try:
                self._forward([ids], stop=self.q.n_layers - 1)
                out = np.array(mx.stack([self._capture_full[l] for l in sorted(self._capture_full)]))
            finally:
                self._capture_full = None
            mx.clear_cache()
            return out
