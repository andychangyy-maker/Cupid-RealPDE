"""Experimental fixed-B16 model graph; same weights and eager tail/CPU paths."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert hashlib.sha256(a.base.read_bytes()).hexdigest() == '83d60c9aff83b07373b31d9b036e0ca42001402d8c6d12c19a10b8166b62e621'
assert not a.output.exists()
old_forward = '''            pred_norm, lower_extent_norm, upper_extent_norm = (
                state["model"].forward_distribution(normalized))'''
new_forward = '''            captured = state["cuda_graph"]
            if captured is not None and normalized.shape == captured[1].shape:
                captured[1].copy_(normalized)
                captured[0].replay()
                pred_norm, lower_extent_norm, upper_extent_norm = captured[2]
            else:
                pred_norm, lower_extent_norm, upper_extent_norm = (
                    state["model"].forward_distribution(normalized))'''
old_prewarm = '''    _get_model()
    dummy = np.zeros((_BATCH_SIZE, 20, 32, 64, 3), dtype=np.float32)
    predict(dummy, metadata=None)
    torch.cuda.synchronize()'''
new_prewarm = '''    state = _get_model()
    dummy = np.zeros((_BATCH_SIZE, 20, 32, 64, 3), dtype=np.float32)
    with torch.inference_mode():
        state["model"]._graph_base_widths = torch.tensor(
            state["model"].config.uncertainty_base_widths,
            device=state["device"], dtype=state["dtype"])
        state["model"]._graph_quantile_levels = torch.tensor(
            state["model"].config.quantile_levels,
            device=state["device"], dtype=state["dtype"])
        static_input = torch.zeros(dummy.shape, device=state["device"], dtype=state["dtype"])
        static_input.sub_(state["mean_input"]).div_(state["std_input"])
        stream = torch.cuda.Stream(device=state["device"])
        stream.wait_stream(torch.cuda.current_stream(state["device"]))
        with torch.cuda.stream(stream):
            for _ in range(3):
                state["model"].forward_distribution(static_input)
        torch.cuda.current_stream(state["device"]).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = state["model"].forward_distribution(static_input)
        state["cuda_graph"] = (graph, static_input, static_output)
        predict(dummy, metadata=None)
    torch.cuda.synchronize()'''
with zipfile.ZipFile(a.base) as src, zipfile.ZipFile(a.output, 'x') as dst:
    for info in src.infolist():
        data = src.read(info.filename)
        if info.filename == 'submission.py':
            text = data.decode()
            assert text.count(old_forward) == text.count(old_prewarm) == 1
            assert text.count('    "model": None,') == 1
            text = text.replace('    "model": None,', '    "model": None,\n    "cuda_graph": None,')
            text = text.replace(old_forward, new_forward).replace(old_prewarm, new_prewarm)
            compile(text, 'submission.py', 'exec')
            data = text.encode()
        elif info.filename == 'models/ensemble_unet2d.py':
            text = data.decode()
            old = '''            reference = member_uv.new_tensor(
                self.config.uncertainty_base_widths).view(
                    1, 1, 1, 1, 2)'''
            new = '''            reference = getattr(self, "_graph_base_widths", None)
            if reference is None:
                reference = member_uv.new_tensor(
                    self.config.uncertainty_base_widths)
            reference = reference.view(1, 1, 1, 1, 2)'''
            assert text.count(old) == 1
            text = text.replace(old, new)
            for tensor in ('quantiles', 'raw'):
                old = f'        levels = {tensor}.new_tensor(self.config.quantile_levels)'
                new = f'''        levels = getattr(self, "_graph_quantile_levels", None)
        if levels is None:
            levels = {tensor}.new_tensor(self.config.quantile_levels)'''
                assert text.count(old) == 1
                text = text.replace(old, new)
            compile(text, info.filename, 'exec')
            data = text.encode()
        dst.writestr(info, data)
print(json.dumps({'zip':str(a.output),'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),
                  'status':'UNTESTED_GRAPH_EXPERIMENT_NOT_FOR_SUBMISSION'}))
