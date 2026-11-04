"""Physical-space velocity reflection with its own reproducible CPU RNG."""
import torch


def reflect_uvp(value):
    # (..., H, W, [u,v,p]): spatial reflection also reverses the normal velocity.
    reflected = value.flip(-3)
    reflected[..., 1].neg_()
    return reflected


class VerticalReflectionDataset:
    def __init__(self, base, probability=.25):
        self.base = base
        self.probability = probability
        self.windows = base.windows
        self.generator = torch.Generator().manual_seed(29009)
        self.seen = self.reflected = 0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        history, target = self.base[index]
        self.seen += 1
        if torch.rand((), generator=self.generator).item() < self.probability:
            self.reflected += 1
            return reflect_uvp(history), reflect_uvp(target)
        return history, target
