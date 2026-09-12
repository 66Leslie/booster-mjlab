# Booster MJLab — Humanoid Locomotion and Motion Tracking

Reinforcement learning environments for Booster T1 and K1 humanoids, featuring velocity control, reference-motion tracking, and target-location locomotion with adversarial motion priors (AMP).

Developed and maintained by [66Leslie](https://github.com/66Leslie), Booster MJLab extends [MJLab Playground](https://github.com/mujocolab/mjlab_playground) with Booster locomotion tasks, shared AMP training components, and robot-specific configurations.

## Demos

| T1 velocity control | T1 reference-motion tracking |
|---|---|
| <img src="docs/media/t1_velocity_tracking.gif" alt="Booster T1 velocity control" width="400" /> | <img src="docs/media/t1_motion_tracking.gif" alt="Booster T1 reference-motion tracking" width="400" /> |

### Target-location locomotion with AMP

| Booster K1 | Booster T1 |
|---|---|
| <img src="docs/media/k1_target_location_amp.gif" alt="Booster K1 target-location locomotion" width="400" /> | <img src="docs/media/t1_target_location_amp.gif" alt="Booster T1 target-location locomotion" width="400" /> |

## Added capabilities

Building on MJLab Playground at [`b036472`](https://github.com/mujocolab/mjlab_playground/commit/b036472), this project adds:

| Capability | Implementation |
|---|---|
| **Target-location locomotion with AMP** | K1 and T1 environments with target commands, task rewards, termination conditions, and robot-specific training configurations. |
| **T1 reference-motion tracking** | A tracking environment with reference-motion commands and configurable local motion data. |
| **T1 velocity control** | A velocity environment adapted from [BoosterT1mjlab](https://github.com/KaydenKnapik/BoosterT1mjlab) and integrated with this project's task configuration and training interfaces. |
| **K1 robot support** | Robot assets and configurations, a get-up environment, and target-location AMP support. |
| **Shared AMP components** | AMP algorithm and runner code, observations, and motion loaders in a task-independent `amp/` package. |

The project retains the upstream Go1 and T1 get-up tasks. Task families are organized under `tasks/`, with tests for get-up, velocity, tracking, and T1 target-location AMP. See [THIRD_PARTY.md](THIRD_PARTY.md) for the origins of adapted code and assets.

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

## Checkpoints

Model weights are not stored in the Git repository. Reviewed checkpoints can be published as versioned GitHub Release assets together with their training configuration, source commit, motion-data provenance, runtime contract, license, and checksums. See [docs/checkpoints.md](docs/checkpoints.md) for the release checklist.

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
