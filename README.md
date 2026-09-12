# MJLab Playground: Booster Locomotion Tasks

This repository extends [MJLab Playground](https://github.com/mujocolab/mjlab_playground) with reinforcement-learning environments for Booster T1 and K1 humanoid robots. It includes velocity control, reference-motion tracking, target-location locomotion with adversarial motion priors (AMP), and get-up tasks.

## Demos

| T1 velocity control | T1 reference-motion tracking |
|---|---|
| <img src="docs/media/t1_velocity_tracking.gif" alt="Booster T1 velocity control" width="400" /> | <img src="docs/media/t1_motion_tracking.gif" alt="Booster T1 reference-motion tracking" width="400" /> |

### Target-location locomotion with AMP

| Booster K1 | Booster T1 |
|---|---|
| <img src="docs/media/k1_target_location_amp.gif" alt="Booster K1 target-location locomotion" width="400" /> | <img src="docs/media/t1_target_location_amp.gif" alt="Booster T1 target-location locomotion" width="400" /> |

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
git clone --branch feature/booster-locomotion-tasks \
  https://github.com/66Leslie/mjlab_playground.git
cd mjlab_playground
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

The task-independent AMP implementation lives in `amp/`. Commands, rewards, termination conditions, and robot-specific configurations remain under their corresponding task packages.

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
