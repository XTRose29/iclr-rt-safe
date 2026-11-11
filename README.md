# RT-SAFE

Core code for real-time embodied safety evaluation: agent control, prompts, five map configurations, 36 tasks, and evaluation tools. This September-derived release includes execution and accounting fixes; the original paper tables use historical campaign records.

## Setup

Requires Linux, Python 3.12, and a GPU for simulation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Live runs require a compatible RT-SAFE Unreal executable with UnrealCV, native swept movement, and maps RT10, RT12, RT15, RT18, and RT20. The executable, Blueprint assets, and cooked maps are not included. Keep rendering enabled.

## Run

Preview a run without starting UE or calling a model:

```bash
python scripts/run_paper.py --model Gemini --study rq2 --task-ids 0
```

Set the model credentials in your environment, then execute:

```bash
export SIMWORLD_UE_LAUNCHER=/path/to/runtime/SimWorld.sh
# Gemini requires OPENROUTER_API_KEY.
python scripts/run_paper.py --model Gemini --study rq2 --task-ids 0 --execute
```

Omit `--task-ids` for all tasks. Studies: `rq1` compares difficulties, `rq2` compares Static/Real-time, and `rq3` compares reasoning efforts. Use `--gpu`, `--ue-port`, and `--output` to select resources; independent runs need separate ports and output directories. Model settings are in `configs/`. CLI-backed models require their authenticated CLI; the historical Claude Code transport is not bundled.

## Measurements

Moves use UE sweeps and stop when blocked. Human, object, and building contacts use UE counters; vehicle impacts also use swept-distance detection. Contact windows reset on movement or rotation. Active-contact recovery lasts 6 s, including zero-intensity contacts. Recovery-period contacts remain active without recursively extending that delay. Building and stuck recovery can restore checkpoints; recovery teleports are excluded from traveled distance. Road contact uses UE state with map-based sidewalk/crosswalk exclusions. The default policy is `visual_only`.

SR, SafeSR, SPL, and safety score `−(C + H + 10V)` are computed from recorded events. Exposure rates divide total collisions by total traveled distance or simulation time.

```bash
python analysis/summarize.py --manifest examples/manifest.json --output results/example_summary
```

The included examples are scripted UE checks, not model benchmark results. For benchmark analysis, select one finalized record per task and condition, including failures. `analysis/paired_uncertainty.py` accepts matched Static/Real-time episode CSVs at one seed and difficulty.

## Checks and scope

```bash
python scripts/verify_source.py
python -m unittest discover -s tests -v
```

Full experiment records, rebuttal-specific runners, and RL training code are not included. See [examples](examples/README.md) and [license notices](THIRD_PARTY_NOTICES.md).
