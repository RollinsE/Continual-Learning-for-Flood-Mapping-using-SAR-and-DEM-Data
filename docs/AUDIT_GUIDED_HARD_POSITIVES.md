# Audit-guided hard-positive regions

Mine false-negative flood regions on the training split with `floodmap mine-hard-positives`, then fine-tune from the strongest baseline using `--hard-positive-region-sampling`. Never mine validation/test regions for training.
