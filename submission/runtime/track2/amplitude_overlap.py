"""Schedule the unchanged historical amplitude solve beside current inference.

Install after RegionalAmplitude construction and before the streaming factory
returns. Call amplitude.prepare(revealed) after target copy/teacher.start and
before the common graph; keep the existing amplitude.apply(revealed) call.
The coefficient graph reads only previous/target, never the current output.
No coefficients, update counts, warm starts, or mathematical formulas change.
"""
from types import MethodType

import numpy as np
import torch


def _prepare(self, revealed):
    if self._overlap_prepared is not None:
        raise RuntimeError('amplitude.prepare called twice without apply/reset')
    self._overlap_prepared = bool(revealed)
    if not revealed:
        return
    self.graph.replay()
    self.history.append(self.coefficients.clone())
    n = len(self.history)
    if n >= 3 and (n == 3 or n & (n - 1) == 0):
        stacked = torch.stack(self.history)
        if n not in self._overlap_host:
            self._overlap_host[n] = torch.empty(
                stacked.shape, dtype=stacked.dtype, device='cpu', pin_memory=True)
        host = self._overlap_host[n]
        host.copy_(stacked, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.target.device))
        # Keep both tensors alive until this particular D2H copy completes.
        self._overlap_pending = (host, stacked, event)


def _apply(self, revealed):
    if self._overlap_prepared is None or self._overlap_prepared != bool(revealed):
        raise RuntimeError('amplitude.apply requires one matching prepare')
    if self._overlap_pending is not None:
        host, stacked, event = self._overlap_pending
        # Wait only for history D2H, recorded before the current common/CDF
        # launches. Do not synchronize all of the device/default stream here.
        event.synchronize()
        n = len(self.history)
        data = host.numpy().astype(np.float64)
        rows, weights = data[:, :192].reshape(n, 32, 6), data[:, 192]
        self.optimum, self.solver_converged = self._overlap_solve(
            rows, weights, self.optimum)
        self.scale.copy_(torch.from_numpy(
            self.optimum.astype(np.float32).reshape(1, 4, 1, 8, 1, 1)))
        self._overlap_pending = None
    self.apply_graph.replay()
    self._overlap_prepared = None


def _reset(self):
    if self._overlap_pending is not None:
        # Also safe if a caller abandons a prepared step and resets directly.
        self._overlap_pending[2].synchronize()
    self._overlap_pending = None
    self._overlap_prepared = None
    self._overlap_original_reset()
    # Pinned buffers are reusable workspace, not adaptation state. Each is
    # overwritten completely and its copy event awaited before it is read.


def install(amplitude):
    """Patch one already-constructed CUDA amplitude object; never its weights."""
    if hasattr(amplitude, '_overlap_original_reset'):
        raise ValueError('amplitude overlap already installed')
    if (amplitude.target.device.type != 'cuda'
            or amplitude.graph is None or amplitude.apply_graph is None):
        raise ValueError('amplitude overlap requires the original CUDA graphs')
    amplitude._overlap_original_reset = amplitude.reset
    amplitude._overlap_solve = type(amplitude).apply.__globals__['solve']
    amplitude._overlap_host = {}
    amplitude._overlap_pending = None
    amplitude._overlap_prepared = None
    amplitude.prepare = MethodType(_prepare, amplitude)
    amplitude.apply = MethodType(_apply, amplitude)
    amplitude.reset = MethodType(_reset, amplitude)
    return amplitude
