"""Keep the original SPS value and permit its point and feature gradients."""
import hashlib
import inspect
import json

import torch


def coupled_objective(common, prediction, target, lower_extent, upper_extent,
                      normalizer, tau_fraction, metric_normalizer=None, *,
                      physical_target=None):
    prediction_physical = common.to_physical_target(prediction, normalizer)[..., :2]
    if (physical_target is not None and metric_normalizer is not None
            and metric_normalizer is not normalizer):
        raise ValueError('physical target override requires the same metric normalizer')
    target_physical = (common.to_physical_target(target, normalizer)[..., :2]
                       if physical_target is None else physical_target[..., :2])
    assert target_physical.shape == target[..., :2].shape
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    lower, upper = lower_extent*scale, upper_extent*scale
    residual = target_physical-prediction_physical
    metric_normalizer = metric_normalizer or normalizer
    prediction_metric = common.to_physical_target(prediction, metric_normalizer)[..., :2]
    target_metric = (common.to_physical_target(target, metric_normalizer)[..., :2]
                     if physical_target is None else target_physical)
    scored = target_metric != 0.
    reference = lower.new_tensor([.01075, .00550]).view(1, 1, 1, 1, 2)
    tau = tau_fraction*reference
    soft_inside = torch.sigmoid((residual+lower)/tau)*torch.sigmoid((upper-residual)/tau)
    efficiency = soft_inside*torch.exp(-(lower+upper)/common.SPS_SIGMA)
    count = scored.sum(dim=(1, 2, 3, 4)).clamp_min(1)
    per_sample = (efficiency*scored).sum(dim=(1, 2, 3, 4))/count
    factor = common.sample_sps_factor(prediction_metric, target_metric, detach=False)
    if prediction.requires_grad:
        assert factor.requires_grad, 'point factor remains detached'
    soft_score = (factor*per_sample).mean()
    return -soft_score, soft_score.detach()


def install(mature, output):
    original = mature.soft_native_sps_objective
    original_parameters = mature.uq_trainable_parameters
    calls = 0

    def objective(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = coupled_objective(mature.common, *args, **kwargs)
        if calls == 1:
            old = original(*args, **kwargs)
            assert torch.equal(old[0], result[0]) and torch.equal(old[1], result[1])
            point_gradient = torch.autograd.grad(result[0], args[0], retain_graph=True)[0]
            assert torch.isfinite(point_gradient).all() and point_gradient.abs().max() > 0
            output.mkdir(parents=True, exist_ok=True)
            proof = dict(forward_bitwise_equal=True,
                         center_factor_gradient_norm=float(point_gradient.norm()),
                         original_objective_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
                         original_parameter_selector_sha256=hashlib.sha256(inspect.getsource(original_parameters).encode()).hexdigest(),
                         evidence='First training batch wiring only; no held quality')
            (output/'coupling_first_batch.json').write_text(json.dumps(proof, indent=2))
            print(json.dumps(proof), flush=True)
        return result

    mature.soft_native_sps_objective = objective
    mature.uq_trainable_parameters = lambda model: tuple(p for p in model.parameters() if p.requires_grad)
