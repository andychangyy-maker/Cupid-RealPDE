"""Frozen forecaster, causal regional means, and sparse empirical intervals."""
import torch

from base_submission import get_ttt_model as get_base_model
from periodic_cdf import PeriodicCDF
from regional_amplitude import RegionalAmplitude
from causal_shape import ShapeFusion
from feature_readout_runtime import FeatureReadout


class StreamingModel:
    def __init__(self, base):
        self.base = base
        self.model = base.model
        self.device = base.device
        self.dtype = base.dtype
        self.model_stats = base.model_stats
        self.official_stats = base.official_stats
        self.cdf = PeriodicCDF(self.official_stats)
        self.cdf.cutoffs = torch.zeros(16, 3, device=self.device)
        self._native = torch.zeros(3, 20, 32, 64, 3, device=self.device)
        self._input = torch.zeros_like(self._native[0])
        self._target = torch.zeros_like(self._input)
        self._output = torch.zeros_like(self._native)
        self._pending = (torch.zeros(16, 5120, device=self.device),
                         torch.zeros(16, 5120, device=self.device),
                         torch.zeros(16, 5120, device=self.device, dtype=torch.long))
        self._history_revealed = torch.zeros((), device=self.device)
        self._revealed = torch.zeros((), device=self.device)
        self._count = torch.zeros((), device=self.device)
        self.amplitude = RegionalAmplitude(self.official_stats, self._target, self._output, self._count)
        from amplitude_overlap import install as install_amplitude_overlap
        install_amplitude_overlap(self.amplitude)
        self._common_graph = None
        self._refresh_graph = None
        self.reset_ttt_state()

    def reset_ttt_state(self):
        with torch.inference_mode():
            self.base.reset_ttt_state()
            if hasattr(self, "shape_fusion"):
                self.shape_fusion.reset()
            self.amplitude.reset()
            if hasattr(self, "readout"):
                self.readout.reset()
            for value in (self.cdf.counts, self.cdf.total, self.cdf.cutoffs,
                          self.cdf.local, self.cdf.local_total,
                          self._count, self._revealed, *self._pending):
                value.zero_()
            self.cdf.lo_z.fill_(-1.)
            self.cdf.hi_z.fill_(1.)
            self.cdf.local_lo.fill_(-1.)
            self.cdf.local_hi.fill_(1.)
        self._n = 0
        self._has_previous = False
        self._cutoffs_ready = False

    def _apply(self):
        alpha = self._count / (self._count + 5) * (self._count >= 3)
        lo, hi = self.cdf.apply(self._lower, self._upper, self._half,
                                self._groups, alpha)
        self._output[1, ..., :2].copy_(lo)
        self._output[2, ..., :2].copy_(hi)

    def _common(self):
        value = self._input.unsqueeze(0)
        base, lower, upper = self.base._forward_base(value)
        shift = self.base._history_shift(value, None, base, self._history_revealed)
        prediction = base.clone()
        prediction[..., :2] += shift
        lower[..., :2] += shift
        upper[..., :2] += shift
        for destination, output in zip(self._native, (prediction, lower, upper)):
            destination.copy_(output[0])
        self.cdf.observe_tensor(self._pending, self._target, self._revealed)
        self._count.add_(self._revealed)
        self._lower, self._upper, midpoint, self._half, self._mask = (
            self.cdf.prepare(self._native, self._input))
        self._bins, self._groups = self.cdf.classify(self._half)
        for destination, value in zip(self._pending, (midpoint, self._half, self._groups)):
            destination.copy_(value)
        self._output.copy_(self._native)
        self._apply()

    def _refresh(self):
        self.cdf.refresh(self._half, self._mask, self._bins)
        self._apply()

    def capture(self):
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                self._common()
                self._refresh()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self._common_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._common_graph), torch.inference_mode():
            self._common()
        self._refresh_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._refresh_graph), torch.inference_mode():
            self._refresh()
        self.reset_ttt_state()

    def ttt_step(self, input_norm, prev_target_norm=None):
        with torch.inference_mode():
            value = torch.as_tensor(input_norm, dtype=torch.float32)
            if tuple(value.shape) != (1, 20, 32, 64, 3):
                raise ValueError('input_norm must have shape (1, 20, 32, 64, 3)')
            self._input.copy_(value[0], non_blocking=True)
            self._history_revealed.fill_(prev_target_norm is not None)
            revealed = self._has_previous and prev_target_norm is not None
            self._revealed.fill_(revealed)
            if revealed:
                target = torch.as_tensor(prev_target_norm, dtype=torch.float32)
                if tuple(target.shape) != (1, 20, 32, 64, 3):
                    raise ValueError('prev_target_norm must have shape (1, 20, 32, 64, 3)')
                self._target.copy_(target[0], non_blocking=True)
                self._n += 1
            if hasattr(self, "shape_fusion"):
                self.shape_fusion.start()
            self.amplitude.prepare(revealed)
            if self._common_graph is None:
                self._common()
            else:
                self._common_graph.replay()
            if not self._cutoffs_ready:
                self.cdf.initialize_cutoffs(self._half, self._mask)
                _, groups = self.cdf.classify(self._half)
                self._pending[2].copy_(groups)
                self._cutoffs_ready = True
            if self._n >= 3 and (self._n == 3 or (self._n & (self._n - 1)) == 0):
                if self._refresh_graph is None:
                    self._refresh()
                else:
                    self._refresh_graph.replay()
            self._has_previous = True
            self.amplitude.apply(revealed)
            lower, upper = (value.unsqueeze(0).clone() for value in self._output[1:])
            prediction = None
            if hasattr(self, "shape_fusion"):
                prediction = self.shape_fusion.predict()
            else:
                prediction = self._output[0].unsqueeze(0).clone()
            prediction = self.readout.predict(prediction)
        return prediction, {'adapt_loss': None, 'lower': lower, 'upper': upper}


def _build_ttt_model(submission_dir, device):
    base = get_base_model(submission_dir, device)
    if base.alpha != 0:
        raise ValueError("this fused package requires policy alpha=0")
    wrapped = StreamingModel(base)
    if wrapped.device.type == 'cuda':
        eager_common = wrapped._common
        compiled_forward = wrapped.model.forward_distribution
        wrapped.common_compiled = False
        try:
            # Establish the stable output buffers before Dynamo inspects attributes.
            with torch.inference_mode():
                eager_common()
                # Avoid Torch 2.2's bound-method guards on a nested compiled function.
                wrapped.model.forward_distribution = type(wrapped.model).forward_distribution.__get__(wrapped.model, type(wrapped.model))
                wrapped._common = torch.compile(eager_common, fullgraph=True, dynamic=False)
                wrapped._common()
            torch.cuda.synchronize(wrapped.device)
        except Exception as error:
            import warnings
            wrapped._common = eager_common
            wrapped.model.forward_distribution = compiled_forward
            warnings.warn('Streaming graph compiler unavailable: ' + repr(error))
        else:
            wrapped.common_compiled = True
            print('[track2] compiled streaming graph ready', flush=True)
        wrapped.capture()
    wrapped.shape_fusion = ShapeFusion(wrapped, submission_dir)
    wrapped.moment = wrapped.shape_fusion.moment
    wrapped.readout = FeatureReadout(wrapped)
    if wrapped.device.type == "cuda":
        wrapped.readout.capture()
    wrapped.reset_ttt_state()
    return wrapped


def get_ttt_model(submission_dir, device):
    import torch._inductor.config as inductor_config
    with inductor_config.patch(conv_1x1_as_mm=True):
        assert inductor_config.conv_1x1_as_mm
        model = _build_ttt_model(submission_dir, device)
        print('[track2] 1x1 convolutions use matrix multiplication', flush=True)
        return model
