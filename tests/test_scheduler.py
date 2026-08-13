import torch

from floods.utils.schedulers import PolynomialLRDecay


def test_polynomial_scheduler_does_not_raise_low_learning_rate():
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam([{'params': model.parameters(), 'lr': 1e-5}], lr=1e-5)
    scheduler = PolynomialLRDecay(opt, max_decay_steps=99, end_learning_rate=1e-4, power=3.0)
    before = opt.param_groups[0]['lr']
    scheduler.step()
    after = opt.param_groups[0]['lr']
    assert after <= before
