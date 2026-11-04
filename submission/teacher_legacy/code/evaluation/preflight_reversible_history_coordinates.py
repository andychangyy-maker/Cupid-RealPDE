"""Check physical units and cost before training reversible history coordinates."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace
import zipfile

from preflight_shared_point_readout import module


def main(args):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[2]
    ev = module(root / 'code/evaluation/evaluate_b210_transfer.py', 'reversible_ev')
    component = module(root / 'code/models/reversible_history_coordinates.py', 'reversible_component')
    assert ev.sha256(args.zip) == 'b8938d06c100cc68bb4b7e310de112754a9053b4d2834134cb2b4e6f4b5c2a35'
    rows = [r for r in ev.clean_rows(args.manifest) if r['aoa'] != 15]
    batch = next(ev.tail_batches(rows, args.real_dir))[0]
    results = {}
    with tempfile.TemporaryDirectory(prefix='reversible_history_', dir='/dev/shm') as runtime:
        with zipfile.ZipFile(args.zip) as archive:
            archive.extractall(runtime)
        sys.path.insert(0, runtime)
        sub = module(Path(runtime) / 'submission.py', 'reversible_original_submission')
        state = sub._get_model()
        model, base_type = state['model'], type(state['model'])
        normalizer = SimpleNamespace(**{key: state[key] for key in ('mean_input', 'mean_target', 'std_input', 'std_target')})
        identities = {name: id(value) for name, value in model.named_parameters()}
        component.install(model, normalizer)
        assert identities == {name: id(value) for name, value in model.named_parameters()}
        candidate_type = type(model)
        for count in (1, 16):
            data = batch[:count]
            raw = torch.as_tensor(data, device=state['device'], dtype=state['dtype'])
            x = (raw - normalizer.mean_input) / normalizer.std_input
            with torch.inference_mode():
                canonical, shift, scale = model.canonical_history(x)
                base = base_type.forward_uq_training(model, canonical)
                actual = model.forward_uq_training(x)
                fast = model.forward_distribution(x)
                std_i, std_t = normalizer.std_input[:2].float(), normalizer.std_target[:2].float()
                physical_history = x[..., :2].float() * std_i + normalizer.mean_input[:2].float()
                variance, physical_mean = torch.var_mean(physical_history, dim=(1, 2, 3), correction=0, keepdim=True)
                physical_scale = torch.sqrt(variance + 1e-6 * std_i.square())
                checks = [
                    (actual[0][..., :2] * std_t + normalizer.mean_target[:2].float(), base[0][..., :2].float() * physical_scale + physical_mean),
                    (actual[1][..., :2] * std_t + normalizer.mean_target[:2].float(), base[1][..., :2].float() * physical_scale[:, None] + physical_mean[:, None]),
                    (actual[2] * std_t[:, None], base[2].float() * physical_scale[..., None]),
                    (actual[3] * std_t, base[3].float() * physical_scale),
                    (actual[4] * std_t, base[4].float() * physical_scale)]
                unit_error = max(float((left - right).abs().max()) for left, right in checks)
                assert unit_error <= 2e-5, unit_error
                fast_error = max(float((left - right).abs().max()) for left, right in zip(fast, (actual[0], actual[3], actual[4])))
                assert fast_error <= 1e-5, fast_error
                first = model.canonical_history(x[:1])
                input_batch_difference = float((canonical[:1] - first[0]).abs().max())
                statistics_batch_difference = max(float((a[:1] - b).abs().max()) for a, b in zip((shift, scale), first[1:]))
                assert statistics_batch_difference <= 1e-5, statistics_batch_difference
                assert all(torch.isfinite(value).all() for value in actual[:5])
                assert torch.equal(actual[0][..., 2:], base[0][..., 2:])
            for kind in (base_type, candidate_type):
                model.__class__ = kind
                for _ in range(10):
                    sub.predict(data)
            observations = []
            calls = 40 if count == 1 else 10
            for cycle in range(5):
                for label in ('A', 'B', 'B', 'A'):
                    model.__class__ = base_type if label == 'A' else candidate_type
                    torch.cuda.synchronize(); start = time.perf_counter()
                    for _ in range(calls):
                        sub.predict(data)
                    torch.cuda.synchronize()
                    observations.append(dict(cycle=cycle, arm=label, seconds_per_call=(time.perf_counter()-start)/calls))
            times = {label: statistics.median(row['seconds_per_call'] for row in observations if row['arm'] == label) for label in ('A', 'B')}
            results[str(count)] = dict(times=times, relative_increase=times['B']/times['A']-1,
                                      unit_error=unit_error, fast_error=fast_error,
                                      input_batch_difference=input_batch_difference,
                                      statistics_batch_difference=statistics_batch_difference, observations=observations)
            model.__class__ = candidate_type
    passed = all(row['relative_increase'] <= .02 for row in results.values())
    result = dict(decision=('PASS_' if passed else 'STOP_') + 'REVERSIBLE_HISTORY_ENGINEERING',
                  batches=results, added_parameters=0, original_parameter_objects_preserved=True,
                  zip_sha256=ev.sha256(args.zip), component_sha256=ev.sha256(component.__file__),
                  script_sha256=ev.sha256(__file__), time_gate=.02,
                  scope='Fit-input units and inference cost only; no held quality or official Time result.')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('decision', 'batches')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('zip', 'manifest', 'real-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
