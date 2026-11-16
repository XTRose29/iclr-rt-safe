# Running the environment

Use native Linux with an NVIDIA GPU and a working Vulkan driver, Python 3.12,
and about 60 GB free disk space. Ubuntu 24.04 with an RTX A5000 (24 GB) is the
validation configuration, not a minimum hardware specification.

## Download

The pinned SimWorld Linux package includes the executable, UnrealCV, packaged
Blueprints and all five RT-SAFE maps. Download it separately; an Unreal editor
is not required. Keep the upstream asset terms. Editable Unreal source assets
are not part of this release.

From the repository root:

```bash
mkdir -p runtime
curl -L --fail --retry 3 -o runtime/Linux.zip \
  'https://huggingface.co/datasets/SimWorld-AI/SimWorld/resolve/31bc15758b66cfd2effb93e2183a1d9ee76363c4/Base20260313/Linux.zip?download=true'
echo '3a0ca8d98ba1de099541549826dc974007243ad4f5d04d0694960cbfe523287f  runtime/Linux.zip' | sha256sum -c -
python3.12 -m zipfile -e runtime/Linux.zip runtime
chmod +x runtime/Linux/SimWorld.sh runtime/Linux/SimWorld/Binaries/Linux/SimWorld
```

The download is approximately 26 GB. Use this exact package, not a newer generic
SimWorld build. Windows and WSL are not validated.

## Install and check

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export SIMWORLD_UE_LAUNCHER="$PWD/runtime/Linux/SimWorld.sh"
export SIMWORLD_TIME_ADVANCE_MODE=resume_pause
python scripts/check_environment.py --launcher "$SIMWORLD_UE_LAUNCHER" \
  --gpu 0 --port 19091 --output results/environment-check
```

The check requires no model credentials. It starts and stops its own simulator
for each map, checking rendering, asset spawning, state interfaces, paused and
running movement, native swept movement and collision-counter resets. It saves
`report.json`, images and logs, and exits nonzero on failure. Use a new output
directory each time; `--levels RT10` runs a shorter check.

All five maps passed this check on the configuration above with a fresh Python
environment and checksum-verified runtime files (16 November 2026).

## Run a task

Once the checks pass, set your model credentials and use the existing runner:

```bash
export OPENROUTER_API_KEY='your-key'
python scripts/run_paper.py --model Gemini --study rq2 --task-ids 0 \
  --gpu 0 --ue-port 19091 --execute
```

This runs the selected task in Static and Real-time modes and makes model API
calls. Keep rendering enabled. Environment checks establish runtime usability;
they do not reproduce historical paper scores.
