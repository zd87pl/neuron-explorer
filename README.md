# neuron explorer

Steering-first neuron explorer for Qwen3.5-4B. Every MLP neuron in a layer is pushed up and down on a few prompts, every
rollout is kept, and the explorer lays the results out over a 3D view of the model.

![neuron explorer](explorer.png)

## Explorer

    uv sync
    uv run python explorer/server.py        # http://localhost:8791

Reads sweep runs from `runs/` and per-layer stats from `stats/`, both produced by the sweep below; the caches in
`explorer/cache/` ship with the repo. Loads Qwen3.5-4B locally for live steered chat, lens, connections and live
inference over your own text: on Apple silicon with MLX (`explorer/runtime_mlx.py`), elsewhere with PyTorch on CUDA or
CPU (`explorer/runtime.py`). `EXPLORER_BACKEND=mlx` or `torch` overrides the choice, `EXPLORER_MODEL` points at another
copy of the weights (a hub id or a local directory; the original HF checkpoint, not an mlx-lm conversion, which rounds
the norms to bf16), and `EXPLORER_NO_MODEL=1` runs from the caches alone.

- **map**: the model's layers in perspective, attention and MLP per layer, one point per neuron. Hover shows a neuron's
  steer lens and logit lens; click opens it; click a layer's label to view its sheet face-on. `writes` places
  neurons with similar down-projection columns next to each other.
- **list**: units ranked by word-cluster mass against random-direction controls; `lens` searches the steer and logit
  lenses for a token, `fires` runs your text through the model and filters neurons by activation.
- **neuron**: steered rollouts per prompt and dose, a three-column multi-turn steered chat (−, 0, +), lens and
  connections, and a name.
- **embeddings**: UMAP of a layer's rollouts, coloured by neuron, prompt, dose or kind.
- **Claude** (fourth neuron-panel tab, or the right-edge handle): a chat running the `claude` CLI (installed and
  logged in) with the explorer as an MCP server (`explorer/agent_mcp.py`). It reads neurons, cones and lenses, runs the
  live model (steering, activations, next-token logprobs, ablations), moves the map, list and highlights, and has file
  tools and a shell in your home folder. Its prompt is `explorer/agent_prompt.md` plus `explorer/agent_methodology.md`, and
  `explorer/agent_prompt.local.md` when present (untracked, for your own additions);
  `EXPLORER_CLAUDE_MODEL` picks the model. The server answers only pages and tools on this machine.

## Sweep

Runs on a CUDA 13 GPU pod with the steering fork of vLLM ([RhizoNymph/vllm](https://github.com/RhizoNymph/vllm)).

    bash sweep/setup_pod.sh                                   # fork, deps, model; writes /workspace/env.sh
    python sweep/neuron_stats.py --layers 22 --tokens 2000000 --out stats/q35-4b
    python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel \
        --neurons all --random-controls 256 --run q35-4b-L22
    sweep/run_depth.sh "4 9 13 17" 384 64                     # random-subset sweep across several layers

`main_rel` doses are fractions of the median residual norm leaving the layer: {who, snake, orange} × {−0.7, +0.5, +0.8}
and {free} × {+3}, plus one unsteered baseline per prompt.

### On a Mac

The same pipeline runs on Apple silicon with MLX: the `_mlx` scripts take the torch scripts' flags and write the same
files, and `sweep/steer_sweep_mlx.py` stands in for the vLLM fork (same jobs, controls, records and resume rule; it
writes the captures itself, in `capture_prefill.py`'s format). All of them build on `sweep/qwen_mlx.py`, which computes
the norms, the activation and the linear-attention recurrence the way transformers does, so bf16 results should differ
from the CUDA runs by bf16 rounding only; `sweep/check_mlx.py` measures that on the machine at hand.

    uv run python sweep/check_mlx.py                         # MLX against PyTorch on this machine, fp32 (--bf16 too)
    uv run python sweep/neuron_stats_mlx.py --layers 22 --tokens 2000000 --out stats/q35-4b
    uv run python sweep/steer_sweep_mlx.py --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel \
        --neurons all --random-controls 256 --run q35-4b-L22        # --batch rows decode together (default 256)
    uv run python sweep/rank.py runs/q35-4b-L22.jsonl --out runs/q35-4b-L22.units.jsonl
    uv run python sweep/steer_lens_mlx.py --layers 0-31 --out stats/steer_lens

| script | output |
|---|---|
| `sweep/neuron_stats.py` | per-neuron activation extremes by sign regime, down columns, residual norms |
| `sweep/steer_sweep.py` | `runs/<run>.jsonl`: text, token ids, logprobs, top-k per step; optional activation captures |
| `sweep/rank.py` | `runs/<run>.units.jsonl`: distinctive words, cluster mass, percentile against controls |
| `sweep/steer_lens.py` | `stats/steer_lens_L{l}.npz`: next-token Δlogprob and KL for ±0.8 pushes on every neuron |
| `sweep/depth_report.py` | `runs/depth_report.json`: per-layer share of neurons above controls, coherence, top units |
| `sweep/shift.py` | word-shift score against controls, all words counted |
| `sweep/merge_runs.py` | joins sharded runs of one layer |
| `sweep/check_equivalence.py` | HF vs vLLM steering and capture agreement |
| `sweep/check_mlx.py` | the explorer's MLX runtime vs its PyTorch runtime: reads, pushes, head scaling, logprobs, chats, lens, connections |
| `sweep/neuron_stats_mlx.py`, `sweep/steer_sweep_mlx.py`, `sweep/steer_lens_mlx.py`, `sweep/capture_prefill_mlx.py` | the same outputs on MLX (Apple silicon) |
| `sweep/bench.py`, `sweep/peek.py` | throughput conditions; terminal views of a run |
| `explorer/build_embeddings.py` | `runs/<run>.umap.json` |
| `explorer/build_layout.py` | `explorer/cache/layout_writes.npz` |
| `explorer/build_connections.py` | `explorer/cache/connections.npz`: each neuron's 24 strongest direct-path links up and down |
| `explorer/build_connections_mlx.py` | the same file on MLX |
| `sweep/capture_prefill.py` | `captures/<run>.mlp_in.npy`: MLP input at the last prompt token at the run's capture layers, steered |

`research_log.md` holds the findings and reasoning; `NOTES.md` the source notes.
