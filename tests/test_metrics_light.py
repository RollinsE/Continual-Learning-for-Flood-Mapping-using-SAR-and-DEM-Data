import torch

from floods.evaluation import BinaryThresholdSweep, BatchAverageMetrics


def test_threshold_sweep_honours_ignore_index():
    y_true = torch.tensor([[[1, 0], [255, 0]]])
    logits = torch.tensor([[[[3.0, -3.0], [3.0, -3.0]]]])
    sweep = BinaryThresholdSweep(thresholds=[0.5], device='cpu')
    sweep.update(y_true, logits)
    row = sweep.compute()[0]
    assert row.f1 == 1.0
    batch = BatchAverageMetrics(threshold=0.5, ignore_index=255)
    batch.update(y_true, logits)
    assert batch.compute()['f1'] == 1.0
