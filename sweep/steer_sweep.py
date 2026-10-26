"""Steer every neuron of one MLP layer, one rollout per (prompt, dose), and keep everything a rollout produces.

Steering neuron n by dose d adds d * abs_max[n] to its activation at every position (prompt and generated). That is
the same as adding d * abs_max[n] * W_down[:, n] to the layer's MLP output, which is what this sends to the steering
fork of vLLM at the `mlp_out` hook. abs_max is the neuron's largest |activation| over the stats corpus, so doses are in
units of the neuron's own range.

Design "main", per neuron, 10 rollouts: {who, snake, orange} x {-2, +2, +10} and {free} x {+10}. Design "calib" is a
dose-response ladder, {who, orange, free} x ±{1, 2, 4, 8, 16, 32}, for finding where effects begin and where text
breaks. Design "main_rel" (with --dose-unit rel) gives every neuron the same push sizes, as fractions of the median
residual norm: {who, snake, orange} x {-0.7, +0.5, +0.8} and {free} x {+3}. One unsteered baseline per prompt. Each
rollout also records rel_push, the size of its push relative to the
median residual norm leaving the layer.
Random-direction controls (a neuron's down column rotated to a random direction, same norm and dose scale) get the
same rollouts as the neurons and give the null for "how often does a theme appear by chance".

Each rollout keeps:
    text, generated token ids, the chosen token's logprob and the top-k alternatives at every step   (jsonl)
    mlp_in activations at the capture layers at the last prompt token, the state just before the model answers
    (vLLM filesystem capture under --capture-root/<run>/<capture_id>). Every neuron of a captured layer is
    recoverable from these as silu(W_gate x) * (W_up x). Activations at generated tokens come from replaying
    the saved token ids through the model with the same steering, one teacher-forced forward pass.

    uv run python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --neurons 2742 --run smoke-2742
    uv run python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --design calib --neurons random:128 --random-controls 64 --run calib-L22
    uv run python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --neurons all --run q35-4b-L22

Rerunning the same command resumes: rollouts already in the jsonl are skipped.
"""
import argparse
import functools
import json
import os
import time

import numpy as np
from transformers import AutoConfig, AutoTokenizer

PROMPTS = {
    "who": "Who are you?",
    "snake": "Write a snake game in Python.",
    "orange": "Describe the color orange.",
    "free": "Write whatever you like.",
}
LADDER = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DESIGNS = {
    "main": [(p, d) for p in ("who", "snake", "orange") for d in (-2.0, 2.0, 10.0)] + [("free", 10.0)],
    # dose-response calibration: where effects begin and where text breaks
    "calib": [(p, s * d) for p in ("who", "orange", "free") for d in LADDER for s in (-1.0, 1.0)],
    # doses as fractions of the median residual norm (use with --dose-unit rel): the coherent window is ~0.5-0.8x,
    # and +3x reads the direction out as wreckage tokens
    "main_rel": [(p, d) for p in ("who", "snake", "orange") for d in (-0.7, 0.5, 0.8)] + [("free", 3.0)],
}


def parse_neurons(spec, n, seed):
    if spec == "all":
        return list(range(n))
    if spec.startswith("random:"):
        return sorted(np.random.default_rng(seed + 1).choice(n, int(spec.split(":")[1]), replace=False).tolist())
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def rollout_record(out, job, layer, top_k):
    o = out.outputs[0]
    ids = list(o.token_ids)
    chosen, top_ids, top_lps = [], [], []
    for tid, lp in zip(ids, o.logprobs or []):
        chosen.append(round(lp[tid].logprob, 4))
        ranked = sorted((v.rank, k, v.logprob) for k, v in lp.items() if v.rank is not None and v.rank <= top_k)
        top_ids.append([k for _, k, _ in ranked])
        top_lps.append([round(x, 4) for _, _, x in ranked])
    return {
        "layer": layer, "kind": job["kind"], "neuron": job["neuron"], "ref_neuron": job["ref_neuron"],
        "prompt": job["prompt"], "dose": job["dose"], "add": job["add"], "rel_push": job["rel_push"],
        "text": o.text, "finish": o.finish_reason, "n_gen": len(ids), "token_ids": ids,
        "logprob": chosen, "top_ids": top_ids, "top_lps": top_lps, "capture_id": job["capture_id"],
    }


def main():
    # imported here, still before anything reads the config, so steer_sweep_mlx.py can import this module without vLLM
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--stats", required=True, help="npz from neuron_stats.py for the layer to steer")
    ap.add_argument("--neurons", default="all", help='"all", "random:128", "2742", "0-99", or "1,5,9-12"')
    ap.add_argument("--design", default="main", choices=sorted(DESIGNS))
    ap.add_argument("--dose-unit", default="absmax", choices=["absmax", "rel"], help="absmax: dose x the neuron's "
                    "abs_max; rel: dose x the median residual norm, the same push size for every neuron")
    ap.add_argument("--run", required=True, help="run name: output file and capture tag")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--max-tokens", type=int, default=256, help="steered rollouts often loop; this bounds them")
    ap.add_argument("--top-k", type=int, default=5, help="alternative tokens kept per generated step")
    ap.add_argument("--capture-layers", default="+0,+1,+4,+8", help="mlp_in layers to capture, relative to the "
                    "steered layer; empty string turns capture off")
    ap.add_argument("--capture-root", default="captures")
    ap.add_argument("--capture-layout", default="packed", choices=["packed", "sharded"], help="packed publishes each "
                    "rollout when it finishes; sharded publishes only when a shard seals")
    ap.add_argument("--random-controls", type=int, default=256, help="random-direction controls spread through the run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4000, help="requests per generate call (one jsonl flush each)")
    ap.add_argument("--max-configs", type=int, default=512, help="distinct steering configs resident at once")
    ap.add_argument("--max-num-seqs", type=int, default=256, help="concurrent sequences; capped by the per-sequence "
                    "linear-attention state memory (about 350 on a 24 GB card at --gpu-mem 0.85)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
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
    with open(os.path.join(a.out_dir, f"{a.run}.meta.json"), "w") as f:
        json.dump({"model": a.model, "layer": layer, "stats": os.path.abspath(a.stats), "prompts": PROMPTS,
                   "design_name": a.design, "design": design, "resid_norm_p50": resid_p50, "dose_unit": a.dose_unit,
                   "hook": "mlp_out", "max_tokens": a.max_tokens, "top_k": a.top_k, "capture_layers": cap_layers,
                   "capture_hook": "mlp_in", "capture_positions": "last_prompt",
                   "capture_root": os.path.abspath(a.capture_root), "capture_tag": a.run,
                   "random_controls": a.random_controls, "seed": a.seed, "prompt_ids": prompt_ids}, f, indent=1)

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

    key = lambda j: (j["kind"], j["neuron"], j["prompt"], j["dose"])
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {key(json.loads(line)) for line in f}

    jobs = [{"kind": "baseline", "neuron": -1, "prompt": p, "dose": 0.0} for p in PROMPTS]
    jobs += [{"kind": kind, "neuron": n, "prompt": p, "dose": d} for kind, n in units for p, d in design]
    jobs = [j for j in jobs if key(j) not in done]
    print(f"L{layer}: {len(neurons)} neurons + {a.random_controls} random controls, {len(jobs)} rollouts to run "
          f"({len(done)} already done)", flush=True)
    if not jobs:
        return

    llm = LLM(
        model=a.model, dtype="bfloat16", enable_steering=True, max_steering_configs=a.max_configs,
        max_num_seqs=a.max_num_seqs, max_model_len=max(map(len, prompt_ids.values())) + a.max_tokens + 16,
        gpu_memory_utilization=a.gpu_mem, limit_mm_per_prompt={"image": 0, "video": 0}, shutdown_timeout=120,
        capture_consumers=[{"name": "filesystem", "params": {"root": a.capture_root}}] if cap_layers else None,
    )

    @functools.lru_cache(maxsize=256)
    def direction(kind, n):
        return (w_down[n] if kind == "neuron" else ctrl_vecs[n]).astype(np.float32).tolist()

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
        kw = {"temperature": 0.0, "max_tokens": a.max_tokens, "logprobs": a.top_k}
        if ref is not None:
            kw["steering_vectors"] = {"mlp_out": {layer: {"vector": direction(kind, n), "scale": job["add"]}}}
        if cap_layers:
            kw["capture"] = {"filesystem": {
                "request_id": job["capture_id"], "tag": a.run, "layout": a.capture_layout,
                "hooks": {"mlp_in": cap_layers},
                "positions": "last_prompt",
            }}
        return SamplingParams(**kw)

    with open(out_path, "a") as f:
        for c in range(0, len(jobs), a.chunk):
            chunk = jobs[c : c + a.chunk]
            t0 = time.time()
            outs = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids[j["prompt"]]) for j in chunk],
                                [params(j) for j in chunk], use_tqdm=False)
            n_tok = 0
            for job, out in zip(chunk, outs):
                rec = rollout_record(out, job, layer, a.top_k)
                n_tok += rec["n_gen"]
                f.write(json.dumps(rec) + "\n")
            f.flush()
            dt = time.time() - t0
            print(f"{min(c + a.chunk, len(jobs))}/{len(jobs)} rollouts, {n_tok / dt:,.0f} gen tok/s, "
                  f"{len(chunk) / dt:.1f} rollouts/s", flush=True)

    llm.llm_engine.engine_core.shutdown(timeout=300)  # a clean engine shutdown seals the sharded captures


if __name__ == "__main__":
    main()
