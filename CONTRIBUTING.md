# Contributing

1. Create a focused branch and keep changes limited to one concern.
2. Add or update tests for behavioural changes.
3. Run `python -m compileall -q floods scripts streamlit_app tests` and `python -m pytest -q`.
4. Do not commit datasets, processed rasters, checkpoints, credentials, local paths, or generated run outputs.
5. Preserve event-level split integrity and train-only normalisation in any reproducibility change.
6. Document new CLI flags in the relevant guide and add them to `--help` tests where applicable.

Bug reports should include the command, resolved configuration, log excerpt, package version, Python/PyTorch versions, GPU type, and whether the run was fresh or resumed.
