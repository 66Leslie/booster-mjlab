[中文](README.zh.md) | **English**

# Booster MJLab — Humanoid Locomotion & Motion Tracking

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900.svg?logo=nvidia&logoColor=white)](pyproject.toml)
[![tests](https://github.com/66Leslie/booster-mjlab/actions/workflows/ci.yml/badge.svg?branch=feature/booster-locomotion-tasks)](https://github.com/66Leslie/booster-mjlab/actions/workflows/ci.yml)
[![built on mjlab_playground](https://img.shields.io/badge/built_on-mjlab__playground-lightgrey.svg)](https://github.com/mujocolab/mjlab_playground)

Reinforcement learning environments for **Booster T1** and **Booster K1** humanoids, covering velocity control, reference-motion tracking, and target-location locomotion with adversarial motion priors (AMP).

Based on [mujocolab/mjlab_playground](https://github.com/mujocolab/mjlab_playground) at [`b036472`](https://github.com/mujocolab/mjlab_playground/commit/b036472) — see [What's new](#whats-new-over-mjlab-playground) for the full delta.

## Contents

- [Demos](#demos)
- [What's new over MJLab Playground](#whats-new-over-mjlab-playground)
- [Environments](#environments)
- [Installation](#installation)
- [Training and evaluation](#training-and-evaluation)
- [Motion data](#motion-data)
- [Project structure](#project-structure)
- [Documentation](#documentation)
- [Checkpoints](#checkpoints)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Demos

**Booster T1 · Velocity control** — track planar velocity commands on flat ground.

<img src="docs/media/t1_velocity_tracking.gif" alt="Booster T1 velocity control" width="400">

**Booster T1 · Reference-motion tracking** — follow a reference motion.

<img src="docs/media/t1_motion_tracking.gif" alt="Booster T1 reference-motion tracking" width="400">

**Booster K1 · Target-location locomotion with AMP** — reach a commanded target position.

<img src="docs/media/k1_target_location_amp.gif" alt="Booster K1 target-location locomotion" width="400">

**Booster T1 · Target-location locomotion with AMP** — the same task family on the T1.

<img src="docs/media/t1_target_location_amp.gif" alt="Booster T1 target-location locomotion" width="400">

## What's new over MJLab Playground

| Capability | Implementation |
|---|---|
| **Target-location locomotion with AMP** | K1 and T1 environments with target commands, task rewards, termination conditions, and robot-specific training configurations. |
| **T1 reference-motion tracking** | A tracking environment with reference-motion commands and configurable local motion data. |
| **T1 velocity control** | A velocity environment adapted from [BoosterT1mjlab](https://github.com/KaydenKnapik/BoosterT1mjlab), integrated with this project's task configuration and training interfaces. |
| **K1 robot support** | Robot assets and configurations, a get-up environment, and target-location AMP support. |
| **Shared AMP components** | AMP algorithm and runner code, observations, and motion loaders in a task-independent `amp/` package. |

The upstream Go1 and T1 get-up tasks are retained. Task families are organized under `tasks/`, with tests for get-up, velocity, tracking, and T1 target-location AMP. See [THIRD_PARTY.md](THIRD_PARTY.md) for the origins of adapted code and assets.

## Environments

| Environment ID | Robot | Task |
|---|---|---|
| `Mjlab-Getup-Flat-Unitree-Go1` | Unitree Go1 | Get up on flat ground |
| `Mjlab-Getup-Flat-Booster-T1` | Booster T1 | Get up on flat ground |
| `Mjlab-Getup-Flat-Booster-K1` | Booster K1 | Get up on flat ground |
| `Mjlab-Velocity-Flat-Booster-T1` | Booster T1 | Track planar velocity commands |
| `Mjlab-Tracking-Flat-Booster-T1` | Booster T1 | Track a reference motion |
| `Mjlab-TargetLocationAmp-Flat-Booster-K1` | Booster K1 | Reach a target position using AMP |
| `Mjlab-TargetLocationAmp-Flat-Booster-T1` | Booster T1 | Reach a target position using AMP |

The T1 target-location task also provides `CompetitionFoot` and `CompetitionCollision` variants for deployment-specific contact and collision settings. Run `uv run list-envs` to list every registered environment.

## Installation

**Prerequisites**

| Requirement | Notes |
|---|---|
| Python | `>=3.13,<3.14` (pinned by `.python-version`) |
| [uv](https://docs.astral.sh/uv/) | `>=0.8.18,<0.9.0`; all commands below run through it |
| NVIDIA GPU + CUDA 12.8 | Required for training. On non-macOS platforms the dependency set pulls `mjlab[cu128]` and CUDA 12.8 PyTorch wheels. The test suite also runs on macOS. |

```bash
git clone https://github.com/66Leslie/booster-mjlab.git
cd booster-mjlab
uv sync
```

## Training and evaluation

Train a policy:

```bash
uv run train Mjlab-Velocity-Flat-Booster-T1 \
  --env.scene.num-envs 4096
```

Run a trained policy:

```bash
uv run play Mjlab-Velocity-Flat-Booster-T1 \
  --checkpoint-file /path/to/model.pt \
  --num-envs 1
```

Record a rollout with MJLab's video recorder:

```bash
uv run play Mjlab-Velocity-Flat-Booster-T1 \
  --checkpoint-file /path/to/model.pt \
  --num-envs 1 \
  --video True \
  --video-length 250
```

Replace the environment ID in these commands to train or evaluate another task.

Common development tasks are also exposed through the `Makefile`:

```bash
make sync     # uv sync
make format   # ruff format + ruff check --fix
make type     # pyright
make test     # pytest tests/ -v
make check    # format + type
```

## Motion data

Reference tracking and AMP training require locally prepared motion data:

- `Mjlab-Tracking-Flat-Booster-T1` reads an MJLab `.motion.npz` file from `MJLAB_TRACKING_MOTION_FILE`.
- The AMP environments load GMR-style `.pkl` or `.pickle` files from `MJLAB_PLAYGROUND_MOTION_ROOT`. The T1 locomotion directory can be overridden with `MJLAB_PLAYGROUND_T1_LOCOMOTION_MOTION_DIR`.

```bash
export MJLAB_TRACKING_MOTION_FILE=/absolute/path/to/reference.motion.npz
export MJLAB_PLAYGROUND_MOTION_ROOT=/absolute/path/to/mjlab_motions
```

Retargeted motion files are not distributed because their redistribution terms depend on the source dataset. See [docs/motion_data.md](docs/motion_data.md) for the expected directory layout, file schemas, and data-provenance requirements.

## Project structure

```text
src/mjlab_playground/
├── amp/                         # Shared AMP components and runners
├── asset_zoo/robots/            # Robot models and constants
│   ├── booster_k1/
│   └── booster_t1/
└── tasks/
    ├── common/                  # Utilities shared across task families
    ├── getup/
    ├── tracking/
    ├── velocity/
    └── target_location_amp/
```

The task-independent AMP implementation lives in `amp/`. Commands, rewards, termination conditions, and robot-specific configurations remain under their corresponding task packages. The Python package remains `mjlab_playground` for compatibility with existing imports and MJLab task discovery.

## Documentation

| Document | Contents |
|---|---|
| [docs/motion_data.md](docs/motion_data.md) | Motion data directory layout, file schemas, and provenance requirements |
| [docs/checkpoints.md](docs/checkpoints.md) | Checkpoint release checklist and required metadata |
| [THIRD_PARTY.md](THIRD_PARTY.md) | Origins of adapted code and assets |
| [NOTICE](NOTICE) | Attribution notices |

## Checkpoints

**No model weights are published with this repository.** Weights are not tracked in Git and no release assets exist yet.

Reviewed checkpoints can be published as versioned GitHub Release assets together with their training configuration, source commit, motion-data provenance, runtime contract, license, and checksums. See [docs/checkpoints.md](docs/checkpoints.md) for the release checklist.

## Acknowledgements

- [mujocolab/mjlab_playground](https://github.com/mujocolab/mjlab_playground) provides the base framework and task organization.
- The T1 velocity environment is adapted from [KaydenKnapik/BoosterT1mjlab](https://github.com/KaydenKnapik/BoosterT1mjlab).
- K1 assets and configurations reference [NishB17/MJLabModified](https://github.com/NishB17/MJLabModified) and [RCSSServerMJ](https://github.com/YiranWang2004/RCSSServerMJ).
- Motion retargeting is compatible with [YanjieZe/GMR](https://github.com/YanjieZe/GMR).

See [NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md) for detailed source and license information.

## Citation

If you use this repository in your research, please cite MJLab:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

Repository code is licensed under [Apache License 2.0](LICENSE), except for components explicitly identified under another license. Motion datasets and locally generated retargeted files are not covered by this repository license.
