# Motion data

Motion data is deliberately kept outside Git. Both `.npz` tracking files and GMR-style `.pkl`/`.pickle` AMP files are ignored so a local licensed dataset cannot be committed accidentally.

## Why retargeted files are not committed

Retargeting changes a motion's robot representation; it does not automatically erase the source dataset's copyright or license conditions.

- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/LICENSE) is distributed under CC BY-NC-ND 4.0. Retargeting is reasonably treated as creating adapted material, which that license does not permit users to share.
- [AMASS](https://amass.is.tue.mpg.de/license.html) access is subject to its own terms, and AMASS aggregates datasets with additional source-specific conditions. BMLrub-derived clips should therefore not be published here without explicit redistribution permission.
- [GMR](https://github.com/YanjieZe/GMR) is MIT-licensed software, but its license covers the retargeting code—not the motion inputs or the resulting adapted dataset.

This is a conservative repository policy, not legal advice. A motion file may be published only when its complete provenance and redistribution rights are documented.

## Recommended local layout

Set one root for local retargeted data:

```bash
export MJLAB_PLAYGROUND_MOTION_ROOT=/absolute/path/to/mjlab_motions
```

Arrange locomotion files as follows:

```text
$MJLAB_PLAYGROUND_MOTION_ROOT/
└── lafan/
    ├── k1_lafan1_fk_pkl/
    │   ├── walk1_subject5.pkl
    │   ├── run1_subject5.pkl
    │   └── sprint1_subject4.pkl
    └── booster_t1_lafan1_locomotion/
        ├── walk....pkl
        ├── run....pkl
        └── sprint....pkl
```

The K1 AMP loader selects `walk*.pkl`, `run*.pkl`, and `sprint*.pkl`. The T1 loader also accepts names prefixed by `lafan1_`.

The T1 locomotion directory can be overridden independently:

```bash
export MJLAB_PLAYGROUND_T1_LOCOMOTION_MOTION_DIR=/absolute/path/to/t1_locomotion_pkls
```

## AMP pickle schema

Each `.pkl`/`.pickle` file must contain at least:

| Key | Shape/type | Meaning |
|---|---|---|
| `dof_pos` | `[T, num_dof]` float array | Joint positions in the robot's configured joint order |
| `root_pos` | `[T, 3]` float array | Root position |
| `root_rot` | `[T, 4]` float array | Root quaternion in `xyzw` order |
| `local_body_pos` | `[T, B, 3]` float array | GMR forward-kinematics body positions |
| `link_body_list` | sequence of `B` names | Names corresponding to `local_body_pos` |
| `fps` | number | Source frame rate; defaults to 30 if absent |

For target-location AMP, the configured `expected_dof` is 22 for K1 and 23 for T1. The required key-body names are defined in `src/mjlab_playground/amp/amp_schema.py`.

## T1 reference tracking

Reference tracking uses the MJLab `.motion.npz` format expected by `MotionCommandCfg`. Point directly to a locally generated file:

```bash
export MJLAB_TRACKING_MOTION_FILE=/absolute/path/to/reference.booster_t1_23dof.motion.npz
uv run play Mjlab-Tracking-Flat-Booster-T1
```

Without that variable, the code checks this intentionally untracked package-relative placeholder:

```text
src/mjlab_playground/asset_zoo/motions/retargeted/
└── tracking/booster_t1_23dof/
    └── 0026_kicking2_stageii.booster_t1_23dof.motion.npz
```

The example filename documents the expected location only; the file is not distributed.

## Before publishing a motion

Record all of the following in a data manifest before considering a motion artifact for release:

1. Original dataset and exact clip identifier.
2. Original dataset version and license URL.
3. Whether derivative/adapted material may be redistributed.
4. Retargeting tool and commit.
5. Robot model and joint order.
6. Any attribution, noncommercial, or share-alike requirements.

If item 3 is unclear or negative, keep the file local and publish only the conversion instructions.
