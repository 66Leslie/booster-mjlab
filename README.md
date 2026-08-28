# mjlab playground: Booster locomotion tasks

This fork extends [mujocolab/mjlab_playground](https://github.com/mujocolab/mjlab_playground) with Booster T1/K1 locomotion tasks while keeping the upstream task-registration and configuration style.

## Demo

### Booster T1 velocity tracking

![Booster T1 velocity tracking](docs/media/t1_velocity_tracking.gif)

The dark-blue arrow is the commanded linear velocity and the cyan arrow is the
measured linear velocity. Dark/light green show commanded/measured yaw motion.
This five-second clip was recorded with MJLab's built-in `play --video` support
using the compatible `2026-04-15_10-04-41/model_10100.pt` checkpoint. The later
`2026-04-27_10-05-04/model_13900.pt` run has a known low-response command region
and is intentionally not used for this demo.

## Repository layout

```text
src/mjlab_playground/
├── amp/                         # Shared AMP algorithm, runner, observations, loaders
├── asset_zoo/robots/            # Robot-specific MJCF and constants
│   ├── booster_t1/
│   └── booster_k1/
└── tasks/                       # Environment/task implementations
    ├── common/                  # Small helpers shared by task families
    ├── getup/
    │   └── config/{go1,k1,t1}/
    ├── velocity/
    │   └── config/t1/
    ├── tracking/
    │   └── config/t1/
    └── target_location_amp/
        └── config/{k1,t1}/
```

Task-independent AMP code lives in `amp/`; target-location commands, rewards, termination rules and robot-specific configuration stay in `tasks/target_location_amp/`. This avoids coupling the AMP implementation to a single task and leaves a clear place for future robots.

## Tasks

| Task ID | Robot | Description |
|---|---|---|
| `Mjlab-Getup-Flat-Unitree-Go1` | Unitree Go1 | Upstream flat-ground get-up |
| `Mjlab-Getup-Flat-Booster-T1` | Booster T1 | Upstream flat-ground get-up |
| `Mjlab-Getup-Flat-Booster-K1` | Booster K1 | Flat-ground get-up |
| `Mjlab-Velocity-Flat-Booster-T1` | Booster T1 | Flat-ground velocity tracking |
| `Mjlab-Tracking-Flat-Booster-T1` | Booster T1 | Reference-motion tracking |
| `Mjlab-TargetLocationAmp-Flat-Booster-K1` | Booster K1 | Target-location locomotion with AMP |
| `Mjlab-TargetLocationAmp-Flat-Booster-T1` | Booster T1 | Target-location locomotion with AMP |

There is one public Booster velocity task: `Mjlab-Velocity-Flat-Booster-T1`.
Rough-terrain, raw-observation, and yaw-only builders remain implementation
details rather than separately advertised environments. There is currently no
K1 tracking task in this fork. The two additional T1 target-location task IDs
ending in `CompetitionFoot` and `CompetitionCollision` are deployment-oriented
contact/collision variants.

## Getting started

```bash
git clone https://github.com/66Leslie/mjlab_playground.git
cd mjlab_playground
uv sync
```

List registered tasks:

```bash
uv run list-envs
```

Train and play a task:

```bash
uv run train Mjlab-Velocity-Flat-Booster-T1 --env.scene.num-envs 4096
uv run play Mjlab-Velocity-Flat-Booster-T1
```

Record a short clip from a local checkpoint:

```bash
uv run play Mjlab-Velocity-Flat-Booster-T1 \
  --checkpoint-file /path/to/model_10100.pt \
  --num-envs 1 --video True --video-length 250
```

The tracking and AMP tasks require separately prepared motion data. See [Motion data](docs/motion_data.md) before running them.

## Motion-data policy

Retargeted files are generated artifacts, but their redistribution rights still depend on the original motion dataset. Consequently, this repository does **not** include LAFAN1-, AMASS-, or BMLrub-derived `.npz`, `.pkl`, or `.pickle` files. Retarget them locally from data you are licensed to use, then point the task to the local output. The expected paths, environment variables and pickle schema are documented in [docs/motion_data.md](docs/motion_data.md).

## Sources and attribution

- Base repository: [mujocolab/mjlab_playground](https://github.com/mujocolab/mjlab_playground), Apache-2.0.
- The T1 velocity task is adapted from [KaydenKnapik/BoosterT1mjlab](https://github.com/KaydenKnapik/BoosterT1mjlab), Apache-2.0, and has been substantially modified for this repository layout and current MJLab APIs.
- K1 MJLab asset/configuration work references [NishB17/MJLabModified](https://github.com/NishB17/MJLabModified), Apache-2.0. The competition-compatible K1 MJCF lineage also references [RCSSServerMJ](https://github.com/YiranWang2004/RCSSServerMJ), MIT.
- The shared AMP implementation was reorganized for this fork. [ccrpRepo/AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab) was consulted for high-level directory organization only; no source code or motion files were copied from it because the repository does not currently publish a root license.
- Motion retargeting is compatible with [YanjieZe/GMR](https://github.com/YanjieZe/GMR), MIT. GMR's software license does not replace the license of the source motion dataset.

See [NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md) for details.

## Checkpoints

Checkpoint files are intentionally ignored by Git. Publish reviewed weights as versioned GitHub Release assets instead of committing them to repository history. Each released checkpoint should identify the task ID, repository commit, complete training configuration, motion-data provenance, training command, observation/action contract and weight license. See [docs/checkpoints.md](docs/checkpoints.md).

## Citation

If you use this repository in research, consider citing MJLab:

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

Code in this repository is released under the [Apache-2.0 License](LICENSE), except where a bundled third-party component is identified with its own license. Motion datasets and locally generated retargeted files are not covered by this repository's Apache-2.0 license.
