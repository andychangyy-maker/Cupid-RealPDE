"""Causal regional linear readout; frozen network, original TKE and bounds."""
import torch


class FeatureReadout:
    def __init__(self, owner):
        self.owner = owner
        device = owner.device
        generator = torch.Generator(device='cpu').manual_seed(20260925)
        matrix = torch.randn(owner.model.config.base, 32, generator=generator)
        self.projection = torch.linalg.qr(matrix, mode='reduced').Q.T.to(device)
        self.a = torch.zeros(8, 33, 33, device=device)
        self.b = torch.zeros(8, 33, 40, device=device)
        self.w = torch.zeros_like(self.b)
        self.eye = torch.eye(33, device=device)
        self.previous_features = torch.zeros(8, 256, 33, device=device)
        self.previous_point = torch.zeros(20, 32, 64, 2, device=device)
        self.previous_scale = torch.ones(2, device=device)
        self.point = torch.zeros(20, 32, 64, 3, device=device)
        self.output = torch.zeros_like(self.point)
        self.input_zero = -owner.official_stats['mean_input'] / owner.official_stats['std_input']
        self.observe_graph = self.apply_graph = None
        self.compiled = False

    @staticmethod
    def group(value):
        return value.reshape(2, 16, 4, 16, -1).permute(0, 2, 1, 3, 4).reshape(8, 256, -1)

    @staticmethod
    def restore(value):
        return value.reshape(2, 4, 16, 16, 20, 2).permute(4, 0, 2, 1, 3, 5).reshape(20, 32, 64, 2)

    def reset(self):
        for value in (self.a, self.b, self.w, self.previous_features, self.previous_point, self.point, self.output):
            value.zero_()
        self.previous_scale.fill_(1.)

    def observe(self):
        stats = self.owner.official_stats
        truth = (self.owner._target * stats['std_target'] + stats['mean_target'])[..., :2]
        residual = (truth - self.previous_point) / self.previous_scale
        target = self.group(residual.permute(1, 2, 0, 3).reshape(32, 64, 40))
        features = self.previous_features
        self.a.add_((features.transpose(1, 2) @ features) * (self.owner._revealed / 256))
        self.b.add_((features.transpose(1, 2) @ target) * (self.owner._revealed / 256))

    def refresh(self):
        ridge = 5 / self.owner._count.clamp_min(1) * self.a.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-8)
        self.w.copy_(torch.linalg.solve_ex(self.a + ridge[:, None, None] * self.eye, self.b, check_errors=False).result)

    def apply(self):
        owner = self.owner
        stats = owner.official_stats
        feature = owner.model._conditional_feature[0].float()
        projected = torch.einsum('kc,chw->hwk', self.projection, feature)
        projected = projected / projected.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        projected = self.group(torch.cat((projected, torch.ones_like(projected[..., :1])), -1))
        observed = owner._input[..., :2] != self.input_zero[:2]
        uv = owner._input[..., :2] * stats['std_input'][:2] + stats['mean_input'][:2]
        uv = torch.where(observed, uv, torch.zeros_like(uv))
        scale = (uv - uv.mean(0, keepdim=True)).square().mean((0, 1, 2)).clamp_min(1e-8).sqrt()
        p = (self.point * stats['std_target'] + stats['mean_target'])[..., :2]
        mean = p.mean(0, keepdim=True)
        dynamic = p - mean
        correction = self.restore(projected @ self.w) * scale
        correction = correction - correction.mean(0, keepdim=True)
        alpha = owner._count / (owner._count + 5) * (owner._count >= 3)
        changed = dynamic + correction * alpha
        energy = dynamic.square().sum(-1).mean(0)
        new_energy = changed.square().sum(-1).mean(0)
        gain = (energy / new_energy.clamp_min(1e-20)).sqrt()
        changed = torch.where((new_energy > energy * 1e-10)[None, :, :, None], changed * gain[None, :, :, None], dynamic)
        mean_weights = self.w.reshape(8, 33, 20, 2).mean(2)
        mean_correction = (projected @ mean_weights).reshape(2, 4, 16, 16, 2)
        mean_correction = mean_correction.permute(0, 2, 1, 3, 4).reshape(32, 64, 2)
        value = mean + changed + mean_correction * scale * observed.any(0) * alpha
        self.output.copy_(self.point)
        normalized = (value - stats['mean_target'][:2]) / stats['std_target'][:2]
        self.output[..., :2].copy_(torch.where(owner._count >= 3, normalized, self.point[..., :2]))
        self.previous_features.copy_(projected)
        self.previous_point.copy_(p)
        self.previous_scale.copy_(scale)

    def capture(self):
        # Only the new readout graphs use TF32; preserve all base-model settings.
        previous = torch.backends.cuda.matmul.allow_tf32
        eager_observe, eager_apply = self.observe, self.apply
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            self._capture_impl()
        except Exception as error:
            import warnings
            self.observe, self.apply = eager_observe, eager_apply
            self.observe_graph = self.apply_graph = None
            self.compiled = False
            self.reset()
            warnings.warn('Conditional readout compiler unavailable: ' + repr(error))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    def _capture_impl(self):
        self.observe = torch.compile(self.observe, fullgraph=True, dynamic=False)
        self.apply = torch.compile(self.apply, fullgraph=True, dynamic=False)
        stream = torch.cuda.Stream(device=self.owner.device)
        stream.wait_stream(torch.cuda.current_stream(self.owner.device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                self.observe()
                self.apply()
        torch.cuda.current_stream(self.owner.device).wait_stream(stream)
        torch.cuda.synchronize(self.owner.device)
        self.observe_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.observe_graph), torch.inference_mode():
            self.observe()
        self.apply_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.apply_graph), torch.inference_mode():
            self.apply()
        self.compiled = True
        self.reset()
        print('[track2] compiled conditional readout ready', flush=True)

    def predict(self, point):
        self.point.copy_(point[0])
        if self.observe_graph is None:
            self.observe()
        else:
            self.observe_graph.replay()
        n = self.owner._n
        if n >= 3 and (n == 3 or n & (n - 1) == 0):
            self.refresh()
        if self.apply_graph is None:
            self.apply()
        else:
            self.apply_graph.replay()
        return self.output.unsqueeze(0).clone()
