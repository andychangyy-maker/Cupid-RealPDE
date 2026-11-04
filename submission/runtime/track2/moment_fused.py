"""The same causal moment update, captured inside the existing shape graph."""
import torch


class HistoryMoment:
    def __init__(self, streaming):
        self.stats = streaming.official_stats
        self.target = streaming._target
        self.revealed = streaming._revealed
        self.count = streaming._count
        self.total = torch.zeros(32, 64, device=streaming.device)
        self.previous_k = torch.zeros_like(self.total)
        self.previous_d = torch.zeros_like(self.total)
        self.square = torch.zeros(32, device=streaming.device, dtype=torch.float64)
        self.cross = torch.zeros_like(self.square)

    def reset(self):
        for v in (self.total, self.previous_k, self.previous_d, self.square, self.cross):
            v.zero_()

    @staticmethod
    def regions(value):
        return value.reshape(4, 8, 8, 8).sum((1, 3)).reshape(32)

    def apply(self, point):
        std, mean = self.stats['std_target'][:2], self.stats['mean_target'][:2]
        y = self.target[..., :2] * std + mean
        ky = .5 * (y-y.mean(0, keepdim=True)).square().mean(0).sum(-1)
        den = ky.square().sum().clamp_min(1e-16)
        self.square.add_(self.regions(self.previous_d.square())/den*self.revealed)
        self.cross.add_(self.regions(self.previous_d*(ky-self.previous_k))/den*self.revealed)
        self.total.add_(ky*self.revealed)
        p = point[0, ..., :2] * std + mean
        q = p-p.mean(0, keepdim=True)
        kp = .5*q.square().mean(0).sum(-1)
        history = torch.where(self.count > 0, self.total/self.count.clamp_min(1), kp)
        d = history-kp
        self.previous_k.copy_(kp)
        self.previous_d.copy_(d)
        beta = (self.cross/self.square.clamp_min(1e-16)).clamp(0, 1)
        beta = beta*self.count/(self.count+5)*(self.count >= 3)
        k = kp+(beta.reshape(4, 1, 8, 1)*d.reshape(4, 8, 8, 8)).reshape(32, 64)
        gain = (k.clamp_min(1e-16)/kp.clamp_min(1e-16)).sqrt().clamp(.7, 1.3).float()
        uv = point[0, ..., :2]+q*(gain[None, ..., None]-1)/std
        return torch.cat((uv, point[0, ..., 2:]), -1).unsqueeze(0)
