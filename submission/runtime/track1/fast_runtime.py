"""Compile the fixed deployment bundle, with the native path as fallback."""
import os
import shutil
import tempfile
import warnings

import torch
from scale_mean import correct
from shape_model import reconstruct, mix_distribution_shape


def prepare(model, shape_model, directory, device, dtype):
    model._compiled = False
    if device.type != 'cuda':
        return
    for network in (model, shape_model):
        for layer in network.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                layer.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True


def compile_bundle(operation, state, directory):
    fixed = {key: state[key] for key in ('model', 'shape_model', 'dtype',
             'mean_input', 'std_input', 'mean_target', 'std_target', 'quantile_weights')}

    def eager(raw):
        return operation(raw, fixed)

    state['bundle_compiled'] = False
    if state['device'].type != 'cuda':
        return eager
    try:
        state['runtime_dir'] = tempfile.TemporaryDirectory(prefix='realpde_t1_bundle_')
        root = state['runtime_dir'].name
        compiler = os.path.join(root, 'tinycc')
        shutil.copytree(os.path.join(os.path.abspath(directory), 'vendor', 'tinycc'), compiler)
        for name in ('cc', os.path.join('bin', 'tcc')):
            path = os.path.join(compiler, name)
            os.chmod(path, os.stat(path).st_mode | 0o111)
        os.environ['CC'] = os.path.join(compiler, 'cc')
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(root, 'inductor')
        os.environ['TRITON_CACHE_DIR'] = os.path.join(root, 'triton')
        os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
        import torch._inductor.config as config
        config.compile_threads = 1
        with torch.inference_mode():
            # Populate the immutable calibration tensors before tracing.
            eager(torch.zeros(1, 20, 32, 64, 3, device=state['device']))
            compiled = torch.compile(eager, fullgraph=True, dynamic=False)
            for batch in (1, 16):
                compiled(torch.zeros(batch, 20, 32, 64, 3, device=state['device']))
        torch.cuda.synchronize(state['device'])
    except Exception as error:
        warnings.warn('Bundle compiler unavailable: ' + repr(error))
        return eager
    state['bundle_compiled'] = True
    state['model']._compiled = True
    print('[track1] compiled complete bundle ready: batches 1 and 16', flush=True)
    return compiled


def _bundle(raw32, state):
    raw = raw32.to(dtype=state["dtype"])
    normalized = (raw - state["mean_input"]) / state["std_input"]
    pred_norm, _, quantiles, lower_extent_norm, upper_extent_norm, _, _ = (
        state["model"].forward_uq_training(normalized))
    prediction = pred_norm * state["std_target"] + state["mean_target"]
    prediction[..., 2] = 0.0
    lower, upper = prediction.clone(), prediction.clone()
    lower[..., :2] = (
        prediction[..., :2] - lower_extent_norm * state["std_target"][:2])
    upper[..., :2] = (
        prediction[..., :2] + upper_extent_norm * state["std_target"][:2])
    bundle = correct(raw32, torch.stack((prediction, lower, upper)).float())
    changed = reconstruct(bundle[0], state["shape_model"](raw32))
    center = bundle[0, ..., :2] + (
        (quantiles.float() * state["quantile_weights"]).sum(-1)
        * state["std_target"][:2].float())
    center = torch.where(
        (raw32[..., :2] != 0).any(1, keepdim=True),
        center, bundle[0, ..., :2])
    changed = mix_distribution_shape(changed, center)
    bundle[0].copy_(changed)
    return bundle


