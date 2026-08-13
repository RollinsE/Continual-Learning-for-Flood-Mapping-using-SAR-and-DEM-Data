import sys
import types

import torch
from torch import nn

try:
    import accelerate  # noqa: F401
except ImportError:
    sys.modules["accelerate"] = types.SimpleNamespace(Accelerator=object)

from floods.trainer.base import Trainer


class _FakeAccelerator:
    mixed_precision = "fp16"
    device = torch.device("cpu")
    scaler = None

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def autocast(self):
        return torch.autocast(device_type="cpu", enabled=False)


class _FiniteForwardInfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value * 0.0 + 1.0

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full_like(grad_output, float("inf"))


class _RecoveryTrainer(Trainer):
    def train_batch(self, batch, *, full_precision=False, record_state=True):
        del batch, record_state
        parameter = next(self.model.parameters())
        if full_precision:
            loss = parameter.square().sum()
        else:
            loss = _FiniteForwardInfiniteBackward.apply(parameter).sum()
        return loss, {"loss": loss.detach()}


def test_amp_overflow_batch_is_retried_in_float32():
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _RecoveryTrainer(
        accelerator=_FakeAccelerator(),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        criterion=nn.MSELoss(),
        categories={},
        train_metrics={},
        val_metrics={},
        progress_bar=False,
        amp_full_precision_retry=True,
        skip_nonfinite_batches=True,
    )
    before = model.weight.detach().clone()
    losses, timings = trainer.train_epoch(0, [torch.tensor([0.0])])
    assert timings
    assert losses["loss"]
    assert trainer.amp_overflow_batches == 1
    assert trainer.fp32_recovery_successes == 1
    assert trainer.fp32_recovery_failures == 0
    assert trainer.skipped_batches == 0
    assert not torch.equal(before, model.weight.detach())
