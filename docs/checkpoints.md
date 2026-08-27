# Checkpoint releases

Model checkpoints may be released separately from the source repository after a provenance and reproducibility review. The `.gitignore` excludes `*.pt` and `*.onnx`, so accidental weight commits—and another large-file push failure—are less likely.

## Release layout

Use a versioned GitHub Release rather than storing binary weights in Git history:

```text
<task-id>-v<version>/
├── model.pt
├── policy.onnx                 # optional deployment export
├── env_cfg.yaml
├── agent_cfg.yaml
├── model_card.md
└── SHA256SUMS
```

The release tag should point at the exact source commit used for training.

## Required model-card fields

Every checkpoint release should state:

1. Task ID and robot model/version.
2. Source commit and clean/dirty training-tree state.
3. Exact `uv run train ...` command and random seed.
4. Environment and agent configuration, including any command curriculum.
5. Observation names/order, action names/order, control frequency and action scaling.
6. Motion data: dataset, clip IDs, license URLs, retargeting tool/commit and whether the data itself is redistributed.
7. Training duration, hardware and headline evaluation metrics.
8. Known limitations and the tested deployment/runtime contract.
9. License applied to the checkpoint and any attribution requirements.
10. SHA-256 hashes for every binary artifact.

## Motion-trained weights

A checkpoint does not contain the original `.npz` or `.pkl` file, but it is still an output of a motion-trained pipeline. Before releasing one, review the source dataset's terms for restrictions on trained models, derived outputs, commercial use and attribution. If the terms are unclear, release the source/configuration first and keep the checkpoint private until permission is confirmed.

Do not claim that the repository's Apache-2.0 license automatically covers a checkpoint trained from third-party data. State the checkpoint license explicitly in its release notes.
