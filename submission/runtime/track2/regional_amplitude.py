"""Causal regional fluctuation scaling, preserving means and absolute bounds."""
import numpy as np
import torch

from sps_optimizer import solve


class RegionalAmplitude:
    def __init__(self, stats, target, output, count):
        self.stats = stats
        self.target = target
        self.output = output
        self.count = count
        self.previous = torch.zeros_like(output)
        self.target_zero = -stats['mean_target'][:2] / stats['std_target'][:2]
        self.coefficients = torch.zeros(193, device=target.device)
        self.scale = torch.ones(1, 4, 1, 8, 1, 1, device=target.device)
        self.graph = None
        self.apply_graph = None
        self.compiled = False
        if target.device.type == 'cuda':
            self._coefficients = torch.compile(self._coefficients, fullgraph=True, dynamic=False)
            self._apply_current = torch.compile(self._apply_current, fullgraph=True, dynamic=False)
            stream = torch.cuda.Stream(device=target.device)
            stream.wait_stream(torch.cuda.current_stream(target.device))
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(3):
                    self._coefficients()
                    self._apply_current()
            torch.cuda.current_stream(target.device).wait_stream(stream)
            torch.cuda.synchronize(target.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), torch.inference_mode():
                self._coefficients()
            self.apply_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.apply_graph), torch.inference_mode():
                self._apply_current()
            self.compiled = True
            print('[track2] compiled regional amplitude ready', flush=True)
        self.reset()

    def reset(self):
        self.previous.zero_()
        self.coefficients.zero_()
        self.scale.fill_(1.)
        self.history = []
        self.optimum = np.ones(32)
        self.solver_converged = True

    def _coefficients(self):
        std, mean = self.stats['std_target'][:2], self.stats['mean_target'][:2]
        p = self.previous[0, ..., :2] * std + mean
        y = self.target[..., :2] * std + mean
        average = p.mean(0, keepdim=True)
        q, delta = p-average, y-average
        fy = y-y.mean(0, keepdim=True)
        ynorm = y.square().sum().clamp_min(1e-16)
        kp = .5*q.square().mean(0).sum(-1)
        ky = .5*fy.square().mean(0).sum(-1)
        knorm = ky.square().sum().clamp_min(1e-16)
        uv = lambda value: value.reshape(20,4,8,8,8,2).sum((0,2,4,5)).reshape(32)/ynorm
        energy = lambda value: value.reshape(4,8,8,8).sum((1,3)).reshape(32)/knorm
        self.coefficients[:192].copy_(torch.stack((uv(q.square()), uv(q*delta), uv(delta.square()),
                                                 energy(kp.square()), energy(kp*ky), energy(ky.square())), -1).flatten())
        lo, hi = self.previous[1, ..., :2], self.previous[2, ..., :2]
        valid = self.target[..., :2] != self.target_zero
        inside = (self.target[..., :2] >= lo) & (self.target[..., :2] <= hi)
        reward = torch.exp(-(hi-lo)*std/.0563870259)
        self.coefficients[192].copy_((reward*(valid & inside)).sum())

    def _apply_current(self):
        self.previous.copy_(self.output)
        uv = self.output[0, ..., :2]
        fluctuation = uv-uv.mean(0, keepdim=True)
        shrink = self.count/(self.count+5)*(self.count >= 3)
        scale = 1+shrink*(self.scale-1)
        uv.add_((fluctuation.reshape(20,4,8,8,8,2)*(scale-1)).reshape(20,32,64,2))

    def apply(self, revealed):
        if revealed:
            if self.graph is None:
                self._coefficients()
            else:
                self.graph.replay()
            self.history.append(self.coefficients.clone())
            n = len(self.history)
            if n >= 3:
                if n == 3 or n & (n-1) == 0:
                    data = torch.stack(self.history).cpu().numpy().astype(np.float64)
                    rows, weights = data[:, :192].reshape(n, 32, 6), data[:, 192]
                    self.optimum, self.solver_converged = solve(rows, weights, self.optimum)
                    self.scale.copy_(torch.from_numpy(self.optimum.astype(np.float32).reshape(1,4,1,8,1,1)))
        # The fixed graph caches the unscaled forecast before modifying its point.
        if self.apply_graph is None:
            self._apply_current()
        else:
            self.apply_graph.replay()
