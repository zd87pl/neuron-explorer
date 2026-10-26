"""Qwen3.5 on MLX with neuron-level access, shared by the explorer's live model and the MLX sweep scripts.

The forward pass is written out here around mlx-lm's attention modules and caches, so one pass can read or change a
layer's MLP input (the post-attention-norm hidden state the gate and up projections read), its neuron activations
silu(gate) * up (the input of down_proj), its MLP output and the residual leaving it, scale attention heads, and stop
after any layer. The zero-centred RMSNorms, the activation and the linear-attention recurrence are computed the way
transformers computes them (fp32 norm with an fp32 1 + w, silu in fp32 rounded once, fp32 q, k, v and decay into the
delta rule), so bf16 runs track the HF and vLLM runs the sweep data came from; mlx-lm alone folds 1 + w into a bf16
weight at load, which moves an fp32 forward pass as far from transformers' as bf16 rounding does.

    q = Qwen("Qwen/Qwen3.5-4B")              # HF checkpoint or an mlx-lm conversion; bf16 as stored
    logits = q(ids, hook=MyHook())           # ids [B, T]; logits of the last position, [B, 1, vocab]
"""
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops, normalize_qk
from mlx_lm.utils import _download, _get_classes, load_model

# the norms transformers computes as rms_norm(x) * (1 + w); mlx-lm adds the 1 to these at load
ZERO_CENTRED = ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm")


class Hook:
    """Per-pass callbacks, called for every layer in order. Each returns its (possibly replaced) input."""

    def mlp_in(self, i, x):  # post-attention-norm hidden state [B, T, hidden]
        return x

    def act(self, i, a, gate, up):  # neuron activations silu(gate) * up [B, T, inter], before down_proj
        return a

    def mlp_out(self, i, y):  # down_proj output [B, T, hidden], before it joins the residual
        return y

    def resid(self, i, h):  # residual leaving the layer [B, T, hidden]
        return h


class _Norm(nn.Module):
    """Zero-centred RMSNorm as transformers computes it: normalise in fp32, scale by fp32 (1 + w), round once."""

    def __init__(self, weight, eps):
        super().__init__()
        self.weight = weight  # fp32, 1 + w already added
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps).astype(x.dtype)


class _GatedNorm(nn.Module):
    """The linear-attention output norm as transformers computes it: normalise in fp32, round, scale by w, then gate
    by silu(z) in fp32."""

    def __init__(self, weight, eps):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def __call__(self, x, gate):
        h = mx.fast.rms_norm(x.astype(mx.float32), None, self.eps).astype(x.dtype)
        h = self.weight.astype(x.dtype) * h
        return (h.astype(mx.float32) * nn.silu(gate.astype(mx.float32))).astype(x.dtype)


class _ScaledHeads(nn.Module):
    """An attention output projection whose input heads (laid side by side) are scaled first when the model's
    `heads` dict names this layer: [(head or None for all, scale)], applied in order, 0 = ablated."""

    def __init__(self, proj, n_heads, owner, layer):
        super().__init__()
        self.proj = proj
        self.n_heads, self.layer = n_heads, layer
        self._owner = owner  # plain attribute, not a parameter

    def __call__(self, x):
        todo = self._owner.heads.get(self.layer)
        if todo:
            B, T, D = x.shape
            v = x.reshape(B, T, self.n_heads, D // self.n_heads)
            for head, scale in todo:  # each in fp32, rounded once, like a bf16 tensor times a Python float in torch
                if head is None:
                    v = (v.astype(mx.float32) * scale).astype(x.dtype)
                else:
                    on = (mx.arange(self.n_heads) == head)[:, None]
                    v = mx.where(on, (v.astype(mx.float32) * scale).astype(x.dtype), v)
            x = v.reshape(B, T, D)
        return self.proj(x)


def linear_attention(m, x, mask=None, cache=None):
    """mlx-lm's Qwen3.5 GatedDeltaNet forward (m is the module) with transformers' precision: q, k, v, beta and the
    decay enter the delta-rule recurrence in fp32, q and k are L2-normalised there, and the conv's silu rounds once."""
    B, S, _ = x.shape
    qkv = m.in_proj_qkv(x)
    z = m.in_proj_z(x).reshape(B, S, m.num_v_heads, m.head_v_dim)
    b, a = m.in_proj_b(x), m.in_proj_a(x)
    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, m.conv_kernel_size - 1, m.conv_dim), dtype=x.dtype)
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache is not None:
        n_keep = m.conv_kernel_size - 1
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, S)
            cache[0] = mx.take_along_axis(conv_input, (ends[:, None] + mx.arange(n_keep))[..., None], axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(m.conv1d(conv_input).astype(mx.float32))
    q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(
        mx.split(conv_out.astype(x.dtype).astype(mx.float32), [m.key_dim, 2 * m.key_dim], -1),
        [m.num_k_heads, m.num_k_heads, m.num_v_heads], [m.head_k_dim, m.head_k_dim, m.head_v_dim])]
    q, k = normalize_qk(q, k, inv_scale=m.head_k_dim ** -0.5, eps=1e-6)
    beta = mx.sigmoid(b).astype(mx.float32)
    g = mx.exp(-mx.exp(m.A_log.astype(mx.float32)) * nn.softplus(a.astype(mx.float32) + m.dt_bias.astype(mx.float32)))
    state = cache[1] if cache is not None else None
    if state is None:
        state = mx.zeros((B, m.num_v_heads, m.head_v_dim, m.head_k_dim), dtype=mx.float32)
    if mx.default_device() == mx.gpu and mx.metal.is_available() and m.head_k_dim % 32 == 0:
        out, state = gated_delta_kernel(q, k, v, g, beta, state, mask)
    else:
        out, state = gated_delta_ops(q, k, v, g, beta, state, mask)
    if cache is not None:
        cache[1] = state
        cache.advance(S)
    out = m.norm(out.astype(x.dtype), z)
    return m.out_proj(out.reshape(B, S, -1))


def silu_mul(gate, up):
    """silu(gate) * up as transformers computes it in bf16: silu in fp32 rounded once, then the product."""
    return nn.silu(gate.astype(mx.float32)).astype(gate.dtype) * up


def dense(linear):
    """A Linear's weight [out, in] as a dense array (dequantized when the checkpoint is quantized)."""
    if isinstance(linear, nn.QuantizedLinear):
        return mx.dequantize(linear.weight, linear.scales, linear.get("biases"), linear.group_size, linear.bits,
                             mode=getattr(linear, "mode", "affine"))
    return linear.weight


def topk(x, k):
    """The k largest entries along the last axis, largest first: (values, indices)."""
    idx = mx.argpartition(-x, kth=k - 1, axis=-1)[..., :k]
    vals = mx.take_along_axis(x, idx, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    return mx.take_along_axis(vals, order, axis=-1), mx.take_along_axis(idx, order, axis=-1)


class Qwen:
    def __init__(self, path, dtype=None):
        self.path = Path(_download(path))
        raw = {}

        def classes(config):
            Model, Args = _get_classes(config)

            class Recording(Model):  # keeps the norm weights as the checkpoint stores them, before mlx-lm adds 1
                def sanitize(self, weights):
                    for k, v in weights.items():
                        if k.endswith("norm.weight") and v.ndim == 1:
                            raw[k.replace("model.language_model", "language_model.model", 1)
                                if k.startswith("model.language_model") else
                                k if k.startswith("language_model.") else "language_model." + k] = v
                        if "conv1d.weight" in k and v.shape[-1] != 1:
                            raw["__hf_format__"] = True
                    return super().sanitize(weights)
            return Recording, Args

        self.model, self.config = load_model(self.path, get_model_classes=classes)
        if self.config.get("model_type") not in ("qwen3_5", "qwen3_5_text"):
            raise ValueError(f"expected a Qwen3.5 checkpoint, got model_type {self.config.get('model_type')!r}")
        if dtype is not None:
            self.model.set_dtype(dtype)
        lm = self.model.language_model
        self.args, self.inner = lm.args, lm.model
        self.layers = self.inner.layers
        self.lm_head = None if self.args.tie_word_embeddings else lm.lm_head
        self.heads = {}  # layer -> [(head or None, scale)], read by _ScaledHeads during a pass

        hf_format = raw.pop("__hf_format__", False)

        def precise(module, path):
            w = raw.get(path + ".weight")
            w = (1.0 + w.astype(mx.float32)) if hf_format and w is not None else module.weight.astype(mx.float32)
            return _Norm(w, module.eps)

        pre = "language_model.model"
        for i, L in enumerate(self.layers):
            p = f"{pre}.layers.{i}"
            L.input_layernorm = precise(L.input_layernorm, f"{p}.input_layernorm")
            L.post_attention_layernorm = precise(L.post_attention_layernorm, f"{p}.post_attention_layernorm")
            if L.is_linear:
                a = L.linear_attn
                a.norm = _GatedNorm(a.norm.weight, a.norm.eps)
                a.out_proj = _ScaledHeads(a.out_proj, a.num_v_heads, self, i)
            else:
                a = L.self_attn
                a.q_norm = precise(a.q_norm, f"{p}.self_attn.q_norm")
                a.k_norm = precise(a.k_norm, f"{p}.self_attn.k_norm")
                a.o_proj = _ScaledHeads(a.o_proj, a.num_attention_heads, self, i)
        self.inner.norm = precise(self.inner.norm, f"{pre}.norm")
        mx.eval(self.model.parameters())

        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.n_layers, self.hidden, self.inter = len(self.layers), self.args.hidden_size, self.args.intermediate_size
        self.layer_types = ["linear_attention" if L.is_linear else "full_attention" for L in self.layers]
        self.dtype = self.inner.embed_tokens.weight.dtype

    # ---------- shapes and weights ----------
    def n_heads(self, layer):
        L = self.layers[layer]
        return L.linear_attn.num_v_heads if L.is_linear else L.self_attn.num_attention_heads

    def mlp_weights(self, layer):
        """(gate [inter, hidden], up [inter, hidden], down [hidden, inter]) of a layer, dense."""
        m = self.layers[layer].mlp
        return dense(m.gate_proj), dense(m.up_proj), dense(m.down_proj)

    def down(self, layer):
        return dense(self.layers[layer].mlp.down_proj)

    def norm_scale(self, layer):
        """The full per-channel scale (1 + w) of the layer's pre-MLP norm, fp32."""
        return self.layers[layer].post_attention_layernorm.weight

    def logits(self, h):
        """lm_head on hidden states that already went through the final norm."""
        return self.inner.embed_tokens.as_linear(h) if self.lm_head is None else self.lm_head(h)

    def final_norm(self, h):
        return self.inner.norm(h)

    # ---------- caches ----------
    def make_cache(self, left_padding=None):
        """Per-layer caches for incremental decoding. left_padding: per row, how many pad tokens the row's prompt
        starts with (rows of a batch are left-padded to one length); None for a single unpadded row."""
        if left_padding is None:
            return [ArraysCache(size=2) if L.is_linear else KVCache() for L in self.layers]
        lp = [int(x) for x in left_padding]
        return [ArraysCache(size=2, left_padding=lp if any(lp) else None) if L.is_linear else BatchKVCache(lp)
                for L in self.layers]

    @staticmethod
    def filter_cache(cache, keep):
        """Keep only batch rows `keep` (a list of row indices) in caches from make_cache(left_padding)."""
        idx = mx.array(keep)
        for c in cache:
            c.filter(idx)

    # ---------- forward ----------
    def __call__(self, ids, cache=None, hook=None, stop=None, last_only=True):
        """One pass over token ids [B, T]. With `stop` = a layer index the pass ends after that layer (hooks run up to
        and including it) and returns None; otherwise it returns logits, of the last position when last_only
        ([B, 1, vocab], model dtype)."""
        if not isinstance(ids, mx.array):
            ids = mx.array(ids, dtype=mx.int32)
        if ids.ndim == 1:
            ids = ids[None]
        h = self.inner.embed_tokens(ids)
        if cache is None:
            cache = [None] * self.n_layers
        fa_mask = create_attention_mask(h, cache[self.inner.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.inner.ssm_idx])
        for i, (L, c) in enumerate(zip(self.layers, cache)):
            x = L.input_layernorm(h)
            h = h + (linear_attention(L.linear_attn, x, ssm_mask, c) if L.is_linear else L.self_attn(x, fa_mask, c))
            x = L.post_attention_layernorm(h)
            m = L.mlp
            if hook is None:
                h = h + m.down_proj(silu_mul(m.gate_proj(x), m.up_proj(x)))
            else:
                x = hook.mlp_in(i, x)
                g, u = m.gate_proj(x), m.up_proj(x)
                a = hook.act(i, silu_mul(g, u), g, u)
                h = hook.resid(i, h + hook.mlp_out(i, m.down_proj(a)))
            if stop is not None and i >= stop:
                return None
        if last_only:
            h = h[:, -1:]
        return self.logits(self.inner.norm(h))
