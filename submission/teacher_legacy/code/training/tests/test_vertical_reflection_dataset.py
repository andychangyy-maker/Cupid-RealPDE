import torch
from vertical_reflection_dataset import VerticalReflectionDataset, reflect_uvp


def test_reflection_vector_law_mask_involution_and_no_source_mutation():
    x = torch.arange(2*4*5*3, dtype=torch.float32).reshape(2,4,5,3)
    x[:, 1, 2] = 0
    before = x.clone()
    out = reflect_uvp(x)
    assert torch.equal(x, before)
    assert torch.equal(out[..., 0], x.flip(-3)[..., 0])
    assert torch.equal(out[..., 1], -x.flip(-3)[..., 1])
    assert torch.equal(out[..., 2], x.flip(-3)[..., 2])
    assert torch.equal(out == 0, x.flip(-3) == 0)
    assert torch.equal(reflect_uvp(out), x)


def test_local_rng_does_not_change_training_rng_or_sample_order():
    class Base:
        windows = list(range(20))
        def __len__(self): return 20
        def __getitem__(self, index):
            x = torch.full((20,4,5,3), float(index+1))
            return x, x+1
    state = torch.get_rng_state().clone()
    a, b = VerticalReflectionDataset(Base()), VerticalReflectionDataset(Base())
    for i in range(100):
        xa, ya = a[i % 20]
        xb, yb = b[i % 20]
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
        assert torch.equal(xa[..., 0], torch.full_like(xa[...,0], float(i%20+1)))
    assert 0 < a.reflected == b.reflected < a.seen == 100
    assert torch.equal(state, torch.get_rng_state())
