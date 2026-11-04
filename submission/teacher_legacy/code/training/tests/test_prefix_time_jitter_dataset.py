from types import SimpleNamespace
import torch
from prefix_time_jitter_dataset import TimeJitterDataset


def test_prefix_boundary_pressure_and_isolated_reproducible_rng():
    trajectory = torch.arange(60, dtype=torch.float32).view(60, 1, 1, 1).expand(-1, 2, 3, 2)
    base = SimpleNamespace(trajectories=[trajectory], windows=[
        SimpleNamespace(trajectory=0, start=0), SimpleNamespace(trajectory=0, start=20)])
    a, b = TimeJitterDataset(base), TimeJitterDataset(base, full_time=True)
    before = torch.get_rng_state().clone()
    starts = []
    for _ in range(64):
        x, y = a[0]
        x2, y2 = b[0]
        assert torch.equal(x, x2) and torch.equal(y, y2)
        assert x.shape == y.shape == (20, 2, 3, 3)
        assert bool((x[..., 2] == 0).all()) and bool((y[..., 2] == 0).all())
        assert y[0, 0, 0, 0] == x[0, 0, 0, 0] + 20
        starts.append(float(x[0, 0, 0, 0]))
    last_x, last_y = a[1]
    assert last_x[0, 0, 0, 0] == 20 and last_y[-1, 0, 0, 0] == 59
    assert min(starts) >= 0 and max(starts) <= 19 and len(set(starts)) > 10
    assert torch.equal(before, torch.get_rng_state()) and a.report()["max_support_overrun"] == 0
    assert a.report()['support'] == 'prefix' and b.report()['support'] == 'full_fit_trajectories'
    real = TimeJitterDataset(base, full_time=True, max_shift=4)
    real_starts = {int(real[0][0][0, 0, 0, 0]) for _ in range(64)}
    assert real_starts == set(range(5)) and real.report()['max_support_overrun'] == 0
    assert torch.equal(before, torch.get_rng_state())
