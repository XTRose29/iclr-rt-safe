# RT-SAFE

Core implementation for the RT-SAFE benchmark. See the paper for methodology and results.

## Setup

Requires Linux, Python 3.12, and a GPU for simulation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The executable, packaged Blueprints and all five maps are available as a separate pinned Linux download. Follow the [environment setup and model-free check](docs/ENVIRONMENT.md) before running a model. An Unreal editor is not required. Keep rendering enabled.

## Run

Preview a run without launching the simulator or calling a model:

```bash
python scripts/run_paper.py --model Gemini --study rq2 --task-ids 0
```

Set `OPENROUTER_API_KEY` for Gemini, then run:

```bash
export SIMWORLD_UE_LAUNCHER="$PWD/runtime/Linux/SimWorld.sh"
export SIMWORLD_TIME_ADVANCE_MODE=resume_pause
python scripts/run_paper.py --model Gemini --study rq2 --task-ids 0 --execute
```

Omit `--task-ids` for all tasks. Choose `rq1`, `rq2`, or `rq3` with `--study`; see `configs/` for model and task settings. The historical Claude Code transport is not included.

## Analysis and checks

```bash
python analysis/summarize.py --manifest examples/manifest.json --output results/example_summary
python scripts/verify_source.py
python -m unittest discover -s tests -v
```

[Examples](examples/README.md) are scripted runtime checks. Full experiment records and RL training code are not included; paper results use historical campaign records.

See [third-party notices](THIRD_PARTY_NOTICES.md).
