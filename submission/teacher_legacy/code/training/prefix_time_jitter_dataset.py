"""Fixed allowed-trajectory time augmentation with an isolated random stream."""
import hashlib
import struct
import torch


class TimeJitterDataset:
    def __init__(self, base, *, full_time=False, max_shift=19):
        self.windows, self.trajectories = base.windows, base.trajectories
        self.support = 'full_fit_trajectories' if full_time else 'prefix'
        self.max_shift = max_shift
        self.generator = torch.Generator().manual_seed(29010)
        self.histogram = [0] * (max_shift + 1)
        self.digest = hashlib.sha256()
        self.max_support_overrun = 0

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        trajectory = self.trajectories[window.trajectory]
        maximum = min(self.max_shift, len(trajectory) - 40 - window.start)
        shift = int(torch.randint(maximum + 1, (), generator=self.generator))
        start = window.start + shift
        uv = trajectory[start:start + 40]
        pressure = torch.zeros((*uv.shape[:-1], 1), dtype=uv.dtype)
        uvp = torch.cat((uv, pressure), -1)
        self.histogram[shift] += 1
        self.digest.update(struct.pack("<qq", index, shift))
        self.max_support_overrun = max(self.max_support_overrun, start + 40 - len(trajectory))
        return uvp[:20], uvp[20:]

    def report(self):
        return {"kind": "anchor_uniform_shift", "support": self.support, "max_shift": self.max_shift, "cpu_rng_seed": 29010,
            "shift_histogram": self.histogram, "actual_sampling_sha256": self.digest.hexdigest(),
            "max_support_overrun": self.max_support_overrun}
