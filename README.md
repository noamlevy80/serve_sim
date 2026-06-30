# serve_sim

**A datacenter-scale simulator for large language model serving.**

`serve_sim` takes a declarative description of *what hardware you have*, *what models you
run*, and *what workload arrives* — then simulates inference end to end (prefill and decode,
batching, parallelism, KV caching, weight loading, and data transfers) on a roofline model of
compute and memory bandwidth. It replays real multi-turn agentic traces and produces a detailed
set of reports so you can find bottlenecks and explore architectural trade-offs before touching
any silicon.

It's built for the kind of question you can't easily answer on a spreadsheet or afford to answer
on a real cluster: *What happens to time-to-first-token if I split prefill and decode across
pools? How big a MoE model can I stream from a second memory tier? Where does my heterogeneous
B200 + Cerebras cluster actually spend its time?*

---

## Why you might want it

- **Architectural exploration.** Mix and match compute devices, memory tiers, network fabrics,
  parallelism schemes, and orchestration policies — all as JSON, no code required.
- **Realistic workloads.** Runs on real agentic conversation traces (multi-turn, with tool calls)
  rather than synthetic token streams, so prefix reuse and KV-cache behaviour reflect actual usage.
- **Modern model support.** Dense and MoE FFNs; MHA / GQA / MLA / sparse (DSA) attention; Mamba-2
  linear attention; latent MoE; multi-token prediction. Models ship as parameter files you can edit.
- **Honest about hardware limits.** Memory is a hard constraint — a run that would oversubscribe a
  device aborts with a clear out-of-memory report instead of inventing impossible occupancy.
- **Rich, inspectable output.** Per-request latency / TTFT / TPOT, throughput, per-device
  utilization and memory occupancy over time, a fine-grained execution-state breakdown, every
  orchestration decision, and the raw event log — plus a web UI to explore it all.

## How it works, briefly

Everything that drives a run is **data**. You describe your cluster in a few JSON files —
[Compute_devices/](Compute_devices), [Memory_devices/](Memory_devices) and
[Systems/](Systems) — pick [Models/](Models) and a workload [Suites/](Suites), and tie them
together with a single run config in [Configs/](Configs). The Python engine in
[Src/serve_sim/](Src/serve_sim) interprets that data and runs a strictly event-driven, fluid
(processor-sharing) simulation: batches are formed in a concurrency window, placed on engine
slots by an eager orchestrator, sharded across pipeline / expert / tensor parallelism, and
charged for compute, bandwidth, network collectives, weight streaming and KV transfers as
contended events on shared resources.

For the full behavioural model, see [Project.md](Project.md).

## Getting started

You'll need Python 3.10+.

```powershell
# 1. Set up an environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. Cache the workload dataset once (offline, reproducible runs)
python cache_dataset.py               # --max-rows N for a smaller, faster cache

# 3. Run a simulation from a config
python run_sim.py Configs/nv_kimi26.json

# 4. Explore the results in the web UI (defaults to the latest run)
python run_viz.py
```

Outputs land in `Outputs/<run_id>/`. Paths inside a config are resolved relative to the config
file, and `--output-root`, `--run-id` and `--tokenizer` override config values on the command line.
Pass `--quiet` to suppress the live progress display.

## Writing your own scenario

A run config names the pieces to load and the orchestration knobs to use. The most direct way to
start is to copy an existing file from [Configs/](Configs) and adjust it:

- Swap the `system` to a different topology from [Systems/](Systems) (or write your own).
- Point the suite at different `models` from [Models/](Models).
- Tune the strategy — `max_batch_size`, `max_concurrency`, parallelism degrees, `allow_pdd`,
  `global_kv_cache`, and more.

Adding a new device, memory, model or system is usually just a new JSON file; the engine picks it
up by filename. The full list of config parameters and their defaults is documented in
[Project.md](Project.md), and [ENGINEERING.md](ENGINEERING.md) maps each data directory to its
schema.

### Adding a new model

A model is a single JSON file in [Models/](Models), named by its stem (e.g. `kimi-k2.6.json` →
referenced as `"kimi-k2.6"` in a suite). The easiest way to add one is to **point Copilot at the
model's HuggingFace root** in agent mode — e.g. *"add https://huggingface.co/deepseek-ai/DeepSeek-V3
to the models"*. Using the bundled **add-model** skill
([.github/skills/add-model/SKILL.md](.github/skills/add-model/SKILL.md)), it fetches the HF
`config.json`, maps it onto the `global` / `blocks` / `layer_pattern` schema, picks the closest
existing model as a template, and validates it loads — the same process every existing model was
added with. To do it by hand, copy the closest file in [Models/](Models) and edit the layer counts,
attention type and FFN.

## What you get out

Each run writes a report aggregated over the suite plus per-request and per-device detail:

- `run_report.json` / `run_report.txt` — throughput, makespan, totals.
- `requests.csv` — per-request latency, TTFT and TPOT.
- `device_summary.csv` / `device_timeline.csv` — utilization, memory occupancy, and an
  execution-state breakdown (compute-bound, bandwidth-bound, waiting for KV / weights / experts,
  kernel-launch, idle) both aggregated and bucketed over time.
- `orchestration_decisions.csv` — every weight load/eviction, prefill, KV reuse/transfer/eviction
  and decode, with the device(s), model, sequence and the window in which it actually ran.
- `events_before_rescaling.csv` / `events_after_rescaling.csv` — the raw event log.

Run `python run_viz.py` to browse any of this in the bundled web visualizer.

## Documentation

- [Project.md](Project.md) — the behavioural spec: the simulation model, every config parameter,
  and the output format, in detail.
- [ENGINEERING.md](ENGINEERING.md) — a map of the codebase and the data directories.
- [Tests.md](Tests.md) — the test strategy.

## License

See [LICENSE](LICENSE).
```