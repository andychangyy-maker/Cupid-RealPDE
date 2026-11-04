"""Original 32-coefficient calibration recipe used by submission 924834."""
import numpy as np


def regions(a):
    n, t, h, w, c = a.shape
    assert (t, h, w, c) == (20, 32, 64, 2)
    return a.reshape(n, 20, 2, 16, 4, 16, 2).transpose(0, 2, 4, 6, 1, 3, 5).reshape(n, 16, -1)


def restore(a):
    return a.reshape(-1, 2, 4, 2, 20, 16, 16).transpose(0, 4, 1, 5, 2, 6, 3).reshape(-1, 20, 32, 64, 2)


def observed(history):
    return np.broadcast_to(np.any(history != 0, axis=1, keepdims=True), history.shape)


def log_scale(history):
    return .5 * np.log(np.mean(history.astype(np.float64) ** 2, axis=(1, 2, 3, 4)).clip(1e-12))


def fit(history, prediction, target, trajectory_ids):
    scale = log_scale(history)
    center, spread = scale.mean(), scale.std()
    assert spread > 0
    f = np.stack([np.ones(len(scale)), (scale - center) / spread], axis=1)
    direction = history.mean(1, keepdims=True) - prediction.mean(1, keepdims=True)
    error = target.mean(1, keepdims=True) - prediction.mean(1, keepdims=True)
    d = regions(np.broadcast_to(direction, history.shape)).astype(np.float64)
    e = regions(np.broadcast_to(error, history.shape)).astype(np.float64)
    mask = regions(observed(history))
    square = (d * d * mask).sum(2)
    cross = (d * e * mask).sum(2)
    gram = np.einsum('nr,ni,nj->rij', square, f, f)
    rhs = np.einsum('nr,ni->ri', cross, f)
    coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
    n = len(set(trajectory_ids))
    return {'center': center, 'spread': spread, 'coefficients': coefficients,
            'shrink': n / (n + 5) if n >= 3 else 0}


def apply(history, prediction, lower, upper, table):
    f = np.stack([np.ones(len(history)), (log_scale(history) - table['center']) / table['spread']], axis=1)
    alpha = np.clip(f @ table['coefficients'].T, 0, .5) * table['shrink']
    direction = history.mean(1, keepdims=True) - prediction.mean(1, keepdims=True)
    shift = restore(regions(np.broadcast_to(direction, history.shape)) * alpha[..., None])
    shift *= observed(history)
    return tuple((value + shift).astype(np.float32) for value in [prediction, lower, upper])
