"""Causal pixel residual CDF with a fixed regional empirical prior."""
import torch
from pixel_parent import PeriodicCDF as RegionalCDF, regions, restore


class PeriodicCDF(RegionalCDF):
    def __init__(self, stats, sigma=.0563870259):
        super().__init__(stats, sigma)
        device = self.counts.device
        self.pixel = regions(torch.arange(4096, device=device).reshape(1,32,64,2).expand(20,-1,-1,-1))
        pixel = torch.arange(4096, device=device)
        parent = (((pixel // 128) // 16) * 4 + ((pixel // 2) % 64) // 16) * 2 + pixel % 2
        self.parent = (parent[:,None] * 4 + torch.arange(4,device=device)).flatten()
        self.local = torch.zeros(16384,65,device=device)
        self.local_total = torch.zeros(16384,1,device=device)
        self.local_lo = torch.full((16384,),-1.,device=device)
        self.local_hi = torch.ones(16384,device=device)
        self.local_grid = torch.linspace(-4,4,65,device=device)
        index = torch.arange(65,device=device)
        self.local_distance = index[None,:] - index[:,None]

    def observe_tensor(self, pending, target, revealed):
        super().observe_tensor(pending,target,revealed)
        midpoint,half,groups = pending
        local_groups = self.pixel * 4 + groups % 4
        y = regions(target[...,:2])
        measured = regions(target[...,:2] != self.target_zero)
        z = (y - midpoint) / half.clamp_min(1e-8)
        valid = measured & (z >= -4) & (z < 4)
        index = ((z + 4) / .125).floor().long().clamp(0,63)
        flat = local_groups * 65 + torch.where(valid,index,64)
        self.local.view(-1).scatter_add_(0,flat.flatten(),(measured.float()*revealed).flatten())
        self.local_total.copy_(self.local.sum(1,keepdim=True))

    def refresh(self, half, mask, bins):
        widths = torch.arange(65,device=half.device) * .125
        exponents = torch.exp(-half[:,:,None] * self.physical_scale[:,:,None] * widths / self.sigma)
        sharpness = []
        for b in range(4):
            active = (mask & (bins == b)).float()
            sharpness.append((exponents*active[:,:,None]).sum(1)/active.sum(1,keepdim=True).clamp_min(1))
        sharp = torch.stack(sharpness,1).reshape(64,65)
        parent_counts = self.counts[:,:128].reshape(64,64,2).sum(-1)
        parent_cdf = torch.cat((torch.zeros_like(self.total),parent_counts.cumsum(1)),1)/self.total.clamp_min(1)
        tiles = self.local.reshape(2,16,4,16,2,4,65).permute(0,2,4,5,6,1,3).reshape(8,520,16,16)
        pooled = torch.nn.functional.avg_pool2d(tiles,3,stride=1,padding=1,count_include_pad=False)
        local = pooled.reshape(2,4,2,4,65,16,16).permute(0,5,1,6,2,3,4).reshape(16384,65)
        local_total = local.sum(1,keepdim=True)
        empirical = torch.cat((torch.zeros_like(local_total),local[:,:64].cumsum(1)),1)
        cdf = (empirical + 100.*parent_cdf[self.parent])/(local_total+100.)
        if cdf.is_cuda:
            from cdf_search_kernel import choose
            lo,hi = choose(cdf,sharp,self.parent)
        else:
            lo,hi = torch.empty_like(self.local_lo),torch.empty_like(self.local_hi)
            for first in range(0,16384,512):
                stop = first+512
                value = cdf[first:stop]
                utility = (value[:,None,:]-value[:,:,None])*sharp[self.parent[first:stop]][:,self.local_distance.clamp_min(0)]
                utility.masked_fill_(self.local_distance[None]<=0,-1.)
                best = utility.flatten(1).argmax(1)
                lo[first:stop],hi[first:stop] = self.local_grid[best//65],self.local_grid[best%65]
        active = self.total[self.parent,0]>0
        self.local_lo.copy_(torch.where(active,lo,-1.))
        self.local_hi.copy_(torch.where(active,hi,1.))

    def apply(self, lower, upper, half, groups, alpha):
        local_groups = self.pixel*4 + groups%4
        lo = lower + alpha*(self.local_lo[local_groups]+1)*half
        hi = upper + alpha*(self.local_hi[local_groups]-1)*half
        return restore(lo),restore(hi)
