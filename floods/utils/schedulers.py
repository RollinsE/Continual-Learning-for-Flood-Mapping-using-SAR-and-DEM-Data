from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning-rate schedule that never raises a parameter-group LR."""

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.0:
            raise ValueError("max_decay_steps must be greater than 1")
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def _decayed_lr(self, base_lr: float) -> float:
        final_lr = min(float(self.end_learning_rate), float(base_lr))
        if self.last_step > self.max_decay_steps:
            return final_lr
        progress = 1 - self.last_step / self.max_decay_steps
        return (base_lr - final_lr) * (progress ** self.power) + final_lr

    def get_lr(self):
        return [self._decayed_lr(base_lr) for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
