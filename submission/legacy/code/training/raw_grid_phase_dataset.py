"""Use released raw PIV sampling phases in training, with independent RNG."""
import hashlib
import torch


class RawGridPhaseDataset:
    def __init__(self, base):
        self.base = base
        self.windows = base.windows
        self.generator = torch.Generator().manual_seed(29011)
        self.phase_counts = [0, 0, 0, 0]
        self.phase_digest = hashlib.sha256()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        history, target = self.base[index]
        assert history.shape == target.shape == (20, 64, 128, 3)
        phase = 0
        if torch.rand((), generator=self.generator).item() < .25:
            phase = int(torch.randint(1, 4, (), generator=self.generator))
        self.phase_counts[phase] += 1
        self.phase_digest.update(bytes([phase]))
        dy, dx = divmod(phase, 2)
        return (history[:, dy::2, dx::2].contiguous(),
                target[:, dy::2, dx::2].contiguous())

    def report(self):
        return {'phase_counts': list(self.phase_counts),
                'phase_sha256': self.phase_digest.hexdigest(),
                'phase_order': [[0, 0], [0, 1], [1, 0], [1, 1]],
                'nonzero_probability': .25, 'cpu_seed': 29011,
                'input_target_same_phase': True, 'interpolation': False}
