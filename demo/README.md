# CXR AI Review & Attention Explorer

This is an isolated local Gradio MVP for presenting the P1_CXR research model. It does not modify or replace the project's training, evaluation, config, or artifact workflows.

## Scope

- Canonical E6 contract: `NORMAL`, `BACTERIAL_PNEUMONIA`, `VIRAL_PNEUMONIA`
- Anonymous in-memory cases only (`Case #001`, etc.)
- Worklist order: **AI Review Priority Score = `1 - P(NORMAL)`**
- Default display grouping: HIGH `>= 0.70`, REVIEW `0.30–<0.70`, LOW `<0.30`
- Optional class-specific E6 ConvNeXt-Tiny Grad-CAM, shown as **AI Attention Visualization**

The score and threshold are research-demo display controls, **not clinical urgency**.

## Install and run

From `P1_CXR/`, install the demo-only dependency into the intended local environment:

```zsh
../.venv/bin/python -m pip install -r demo/requirements-demo.txt
../.venv/bin/python demo/app.py
```

The app intentionally runs even when no model checkpoint is available. It displays **Model unavailable** and never invents predictions or attention maps.

`demo/assets/demo_cases/` is intentionally empty. Put only approved, de-identified presentation images there if desired; the app assigns anonymous case IDs and never shows source filenames.

## Connect a validated E6 checkpoint

The default expected path is:

```text
artifacts/experiments/e6/best.pt
```

A supplied checkpoint must be a project checkpoint with all of the following:

- `checkpoint_version == 2`
- `run_kind == "e6_full_epoch"`
- `config.experiment == "e6"`
- E6 three-class model contract

The demo explicitly rejects paths containing `smoke`, `dryrun`, or `e0_dryrun`; it never falls back to the existing E0 smoke/dry-run artifacts.

Override the location at launch time:

```zsh
P1_CXR_DEMO_CHECKPOINT="/absolute/path/to/e6/best.pt" ../.venv/bin/python demo/app.py
```

or:

```zsh
../.venv/bin/python demo/app.py --checkpoint /absolute/path/to/e6/best.pt
```

After receiving an AWS/JupyterLab checkpoint, run the existing E6 evaluation flow first to verify it against `configs/e6_training.yaml`. Once it meets the adapter contract, this demo loads it without UI changes.

## Architecture boundary

`demo_core/model_adapter.py` is the sole project-model integration point. A future validated architecture can replace or extend that adapter while preserving the case store, priority policy, presentation, and Gradio UI contract.

## Disclaimer

> This AI demo is intended for research and educational purposes only and does not replace clinical diagnosis.
>
> This prototype is limited to model predictions for Normal, Bacterial Pneumonia, and Viral Pneumonia.
