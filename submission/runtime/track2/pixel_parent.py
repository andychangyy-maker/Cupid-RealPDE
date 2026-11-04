"""Causal empirical intervals in official coordinates, with sparse refreshes."""
import torch


def regions(value):
    return value.reshape(20, 2, 16, 4, 16, 2).permute(1, 3, 5, 0, 2, 4).reshape(16, -1)


def restore(value):
    return value.reshape(2, 4, 2, 20, 16, 16).permute(3, 0, 4, 1, 5, 2).reshape(20, 32, 64, 2)


class PeriodicCDF:
    def __init__(self, stats, sigma=.0563870259):
        device = stats['std_target'].device
        self.sigma = sigma
        self.input_zero = -stats['mean_input'][:2] / stats['std_input'][:2]
        self.target_zero = -stats['mean_target'][:2] / stats['std_target'][:2]
        self.physical_scale = stats['std_target'][:2].repeat(8)[:, None]
        self.counts = torch.zeros(64, 129, device=device)
        self.total = torch.zeros(64, 1, device=device)
        self.widths = torch.arange(129, device=device) * .0625
        self.edges = torch.linspace(-4, 4, 129, device=device)
        index = torch.arange(129, device=device)
        self.distances = index[None, :] - index[:, None]
        self.group_base = torch.arange(16, device=device)[:, None] * 4
        self.cutoffs = None
        self.lo_z = torch.full((64,), -1., device=device)
        self.hi_z = torch.ones(64, device=device)
        self.n = 0
        self.refreshes = 0

    def prepare(self, base, current):
        lower, upper = regions(base[1, ..., :2]), regions(base[2, ..., :2])
        midpoint, half = .5 * (lower + upper), .5 * (upper - lower)
        mask = regions(current[..., :2] != self.input_zero)
        return lower, upper, midpoint, half, mask

    def initialize_cutoffs(self, half, mask):
        selected = torch.where(mask, half, float('nan'))
        cutoffs = torch.nanquantile(selected, half.new_tensor([.25, .5, .75]), dim=1).T
        cutoffs.nan_to_num_(nan=0.)
        if self.cutoffs is None:
            self.cutoffs = cutoffs
        else:
            self.cutoffs.copy_(cutoffs)

    def classify(self, half):
        bins = (half[:, :, None] > self.cutoffs[:, None]).sum(-1)
        return bins, self.group_base + bins

    def refresh(self, half, mask, bins):
        exponents = torch.exp(-half[:, :, None] * self.physical_scale[:, :, None] * self.widths / self.sigma)
        sharpness = []
        for b in range(4):
            active = (mask & (bins == b)).float()
            sharpness.append((exponents * active[:, :, None]).sum(1) /
                             active.sum(1, keepdim=True).clamp_min(1))
        sharp = torch.stack(sharpness, 1).reshape(64, 129)
        cdf = torch.cat([torch.zeros_like(self.total), self.counts[:, :128].cumsum(1)], 1) / self.total.clamp_min(1)
        mass = cdf[:, None, :] - cdf[:, :, None]
        utility = mass * sharp[:, self.distances.clamp_min(0)]
        utility.masked_fill_(self.distances[None] <= 0, -1)
        best = utility.flatten(1).argmax(1)
        self.lo_z.copy_(torch.where(self.total[:, 0] > 0, self.edges[best // 129], -1.))
        self.hi_z.copy_(torch.where(self.total[:, 0] > 0, self.edges[best % 129], 1.))

    def apply(self, lower, upper, half, groups, alpha):
        lo = lower + alpha * (self.lo_z[groups] + 1) * half
        hi = upper + alpha * (self.hi_z[groups] - 1) * half
        return restore(lo), restore(hi)

    def predict(self, base, current):
        lower, upper, midpoint, half, mask = self.prepare(base, current)
        if self.cutoffs is None:
            self.initialize_cutoffs(half, mask)
        bins, groups = self.classify(half)
        candidate = base.clone()
        if self.n >= 3:
            if self.n == 3 or (self.n & (self.n - 1)) == 0:
                self.refresh(half, mask, bins)
                self.refreshes += 1
            lo, hi = self.apply(lower, upper, half, groups, self.n / (self.n + 5))
            candidate[1, ..., :2], candidate[2, ..., :2] = lo, hi
        return candidate, (midpoint, half, groups)

    def observe(self, pending, target):
        self.observe_tensor(pending, target, 1.)
        self.n += 1

    def observe_tensor(self, pending, target, revealed):
        midpoint, half, groups = pending
        y = regions(target[..., :2])
        measured = regions(target[..., :2] != self.target_zero)
        z = (y - midpoint) / half.clamp_min(1e-8)
        valid = measured & (z >= -4) & (z < 4)
        index = ((z + 4) / .0625).floor().long().clamp(0, 127)
        # The final bin retains measured residuals outside the scored CDF range.
        flat = groups * 129 + torch.where(valid, index, 128)
        self.counts.view(-1).scatter_add_(0, flat.flatten(), (measured.float() * revealed).flatten())
        self.total.copy_(self.counts.sum(1, keepdim=True))
