"""Final reproduction adapters around the recovered, fixed training recipes."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / 'legacy'
ASSETS = ROOT / 'assets'
MANIFEST = ASSETS / 'manifest.json'
ANGLES = (0, 5, 10, 15, 20)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def paths(source=LEGACY):
    sys.path[:0] = [str(source / 'code'), str(source / 'code/training'),
                    str(source / 'code/evaluation'), str(source / 'runs/private_audit')]


def invoke(module, arguments):
    sys.argv = [str(module.__file__)] + list(map(str, arguments))
    module.main()


def architecture(track):
    return (128, 3) if track == 2 else (210, 4)


def teacher(a):
    paths(ROOT / 'teacher_legacy')
    module = importlib.import_module('train_ensemble_unet2d')
    recipe = next(x for x in read(ASSETS / 'teacher_recipes.json')['teachers']
                  if x['teacher'] == a.scope)
    target = a.work / 'teachers' / a.scope
    if target.exists():
        raise FileExistsError(target)
    config = {**recipe['training_config'], 'output_dir': target,
              'sim_dir': a.data / 'train_sim', 'real_dir': a.data / 'train_real',
              'real_stats': a.data / 'mean_std_real.pt', 'device': 'cuda:0'}
    assert config['start_checkpoint'] is None
    arguments = []
    for key, value in config.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            arguments.append(flag if value else '--no-' + key.replace('_', '-'))
        elif value is not None:
            arguments.append(flag)
            arguments.extend(value if isinstance(value, list) else [value])
    invoke(module, arguments)


def cache(a):
    paths()
    checkpoints = [a.work / 'teachers' / name / 'final.pth' for name in 'ABC']
    module = importlib.import_module('build_sim_p1_point_teacher_cache')
    invoke(module, ['--checkpoints', *checkpoints,
                    '--expected-checkpoint-sha256', *map(sha, checkpoints),
                    '--sim-dir', a.data / 'train_sim', '--output', a.work / 'p1_cache.pt',
                    '--batch-size', 16, '--device', 'cuda:0'])


def closure(a):
    paths()
    module = importlib.import_module('build_clean_real_closure_bank')
    invoke(module, ['--real-dir', a.data / 'train_real', '--train-manifest', MANIFEST,
                    '--train-manifest-sha256', sha(MANIFEST), '--output', a.work / 'closure.pt',
                    '--batch-size', 64, '--device', 'cuda:0'])


def sim(a):
    paths()
    sys.path.insert(0, str(ROOT / 'recipes/architecture'))
    import parametric_original as adapter
    base, depth = architecture(a.track)
    backbone = adapter.import_original(LEGACY)
    info = adapter.install_architecture(backbone, base, depth)
    module = adapter.configure_stage(a.scope, base, depth)
    out = a.work / 'sim' / a.scope
    arguments = ['--sim-dir', a.data / 'train_sim', '--real-stats', a.data / 'mean_std_real.pt',
                 '--device', 'cuda:0', '--num-workers', 0, '--output-dir', out]
    if a.scope == 'pretext':
        arguments += ['--real-dir', a.data / 'train_real', '--clean-manifest', MANIFEST]
    else:
        previous = a.work / 'sim' / ('pretext/causal_pretrain_4000.pth'
                                     if a.scope == 'standard1' else 'standard1/model_10000.pth')
        cache_sha = sha(a.work / 'p1_cache.pt')
        arguments += ['--pretrain-checkpoint', previous, '--pretrain-checkpoint-sha256', sha(previous),
                      '--teacher-cache', a.work / 'p1_cache.pt', '--teacher-cache-sha256', cache_sha,
                      '--research-equivalent-cache-sha256', cache_sha,
                      '--acknowledge-research-cache', backbone.RESEARCH_CACHE_ACKNOWLEDGEMENT,
                      '--real-closure-bank', a.work / 'closure.pt',
                      '--real-closure-bank-sha256', sha(a.work / 'closure.pt')]
    invoke(module, arguments)
    write(out / 'final_architecture.json', info)
    if a.scope == 'standard2':
        import torch
        endpoint = out / 'model_10000.pth'
        payload = torch.load(endpoint, map_location='cpu', weights_only=False, mmap=True)
        lineage = [{'name': stage, 'updates': count, 'checkpoint_path': str(a.work / 'sim' / stage / name),
                    'checkpoint_sha256': sha(a.work / 'sim' / stage / name)}
                   for stage, count, name in [('pretext', 4000, 'causal_pretrain_4000.pth'),
                                              ('standard1', 10000, 'model_10000.pth'),
                                              ('standard2', 10000, 'model_10000.pth')]]
        write(a.work / 'sim/binding.json', {
            'schema': 'architecture_variant_causal_sim_binding_v1',
            'family': 'causal4k_standard20_architecture_variant', 'base': base, 'depth': depth,
            'phase': 'sim', 'iteration': 10000, 'checkpoint_sha256': sha(endpoint),
            'lineage': lineage, 'teacher_cache_sha256': sha(a.work / 'p1_cache.pt'),
            'historical_C_identity_claim': False, 'status': payload['metrics']['status'],
            'training_visibility': 'shared clean81 Real prefix; released Sim; stage calibration folds'})


def real_setup(a, updates):
    paths()
    sys.path.insert(0, str(ROOT / 'recipes/architecture'))
    import parametric_original
    import run_architecture_real as adapter
    if a.track == 2:
        import run_architecture_real_d3_fix1  # installs the recovered D3-only fix
    adapter.import_original = lambda: parametric_original.import_original(LEGACY)
    adapter.SOURCE = LEGACY
    adapter.CACHE_SHA = sha(a.work / 'p1_cache.pt')
    root = a.work / ('point6645' if updates == 6645 else 'point5334')
    root.mkdir(exist_ok=True)
    for name in ('train.py', 'coupling.py', 'point_runtime.py', 'train_matched_uq.py'):
        text = (ROOT / 'recipes/point' / name).read_text()
        if updates == 6645 and name in ('train.py', 'train_matched_uq.py'):
            text = text.replace('5334', '6645')
        target = root / name
        if target.exists():
            assert target.read_text() == text
        else:
            target.write_text(text)
    # Original receipts contain historical hashes; sampling is fixed by the
    # unchanged 3200/seed18/batch8 trainer and the current point receipt instead.
    write(root / 'control_uq_receipts.json', {})
    base, depth = architecture(a.track)
    bindings = adapter.install_real(base, depth, root)
    endpoint, binding = a.work / 'sim/standard2/model_10000.pth', a.work / 'sim/binding.json'
    adapter.validate_sim(read(binding), base, depth, sha(endpoint))
    plan = {'source_root': str(LEGACY), 'source_commit_label': '5b28cf57f609fc2fe481d8f08818a5a0a7d9ba76',
            'source_commit_label_note': 'Historical label; packaged sources have separate SHA256 inventory.',
            'architecture': bindings[-1], 'sim': str(endpoint), 'sim_sha256': sha(endpoint),
            'sim_metadata': str(binding), 'sim_metadata_sha256': sha(binding),
            'manifest': str(MANIFEST), 'real_dir': str(a.data / 'train_real'),
            'point_wall_budget_seconds': 3600}
    write(root / 'plan.json', plan)
    return root


def point(a):
    updates = 6645 if a.track == 2 and a.scope == 'full' else 5334
    root = real_setup(a, updates)
    module = importlib.import_module('train')
    module.main(SimpleNamespace(plan=root / 'plan.json', full_fit=a.scope == 'full',
                                angle=None if a.scope == 'full' else int(a.scope)))


def native_uq(a):
    root = real_setup(a, 6645 if a.track == 2 else 5334)
    module = importlib.import_module('train_matched_uq')
    module.main(root, None, False, True)


def shape(a):
    sys.path.insert(0, str(ROOT / 'recipes/shape'))
    module = importlib.import_module('train_runtime')
    metric = importlib.import_module('metric_source')
    angle = None if a.scope == 'full' else int(a.scope)
    rows = [r for r in read(MANIFEST) if r['aoa'] != angle]
    out = a.work / 'shape' / ('full' if angle is None else f'held{angle}')
    out.mkdir(parents=True, exist_ok=False)
    runtime = SimpleNamespace(angle=angle, plan={'manifest': str(MANIFEST),
                                                'real_dir': str(a.data / 'train_real')})
    module.fit(runtime, rows, metric.errors, out)


def tail_context(a):
    paths()
    source = ROOT / 'recipes/tail/run_exposure_tail.py'
    text = source.read_text()
    old = ("mr['point_sha256']=='17ebd5735f5943b8b55190bcccf03ed9ab3077f4cdfe4410458744e4ec0ec2a5' "
           "and sha(p['point'])=='30d243ba71efd1e17c53c1c7b6325d157d44fc90ebe085481c724172bc04efc4'")
    assert text.count(old) == 2
    text = text.replace(old, "mr['point_sha256']==sha(p['point'])")
    target = a.work / 'tail/current_tail.py'
    target.parent.mkdir(exist_ok=True)
    target.write_text(text)
    module = load_module(target, 'current_tail')
    module.RECIPE_ROOT = ROOT / 'recipes/tail'
    module.EXPORT_ROOT = ROOT / 'recipes'
    angle = None if a.scope == 'full' else int(a.scope)
    name = 'full' if angle is None else f'held{angle}'
    updates = 6645 if a.track == 2 and angle is None else 5334
    point_root = a.work / f'point{updates}'
    tail_args = SimpleNamespace(
        scope=a.scope, point_root=a.work / 'point5334', shape_root=a.work / 'shape',
        tail_root=a.work / 'tail', runtime_source=ROOT / 'runtime' / f'track{a.track}',
        storage_policy=ASSETS / f'track{a.track}_storage_policy.json',
        exporter=ROOT / 'recipes/export.py', source_root=LEGACY,
        manifest=MANIFEST, real_dir=a.data / 'train_real')
    out = tail_args.tail_root / 'outputs' / name
    out.mkdir(parents=True, exist_ok=True)
    artifacts = dict(point=point_root / name / f'ema999_u{updates}_primary.pth',
                     point_receipt=point_root / name / 'point_endpoint.json',
                     uq=point_root / name / 'matched_uq/fulltime_uq3200.pth',
                     uq_receipt=point_root / name / 'matched_uq_done.json',
                     shape=a.work / 'shape' / name / 'ema999_u20000.pth',
                     mean=out / 'mean_calibration.npz', storage_policy=tail_args.storage_policy,
                     blend=out / f'reflection_uq_blend_held{angle}/joint_uq.pth')

    def check_point(options, held, files):
        record = read(files['point_receipt'])
        assert record['held_aoa'] == held and record['updates'] == updates
        assert sha(files['point']) == record['checkpoint_sha256']
        payload = module.load(files['point'])
        assert (payload['model_config']['base'], payload['model_config']['depth']) == architecture(a.track)
        rows = [r for r in read(MANIFEST) if r['aoa'] != held]
        assert {r['file'] for r in rows} == set(record['fit_files'])
        return payload, record, rows

    module.check_point = check_point
    return module, tail_args, angle, out, artifacts


def mean_cache(a):
    module, options, angle, out, files = tail_context(a)
    module.make_mean_cache(options, angle, out, files)


def mean(a):
    module, options, angle, out, files = tail_context(a)
    assert angle is None
    module.fit_mean(options, angle, out, files)


def tail(a):
    assert a.scope in ('uq8000', 'reflection4000', 'blend')
    stage = a.scope
    a.scope = 'full'
    module, options, angle, out, files = tail_context(a)
    runtime = module.configured_recipe(options, angle, out, files)
    if stage == 'uq8000':
        recipe = module.load_tail_recipe(options, 'joint_uq')
        runtime.ROOT = out
        recipe.main(None, adaptive_tau=True, baseline='080', steps_override=8000)
    elif stage == 'reflection4000':
        recipe = module.load_tail_recipe(options, 'distill_reflection')
        runtime.ROOT = out
        recipe.main(None)
    else:
        recipe = module.load_tail_recipe(options, 'reflection_uq_only')
        runtime.ROOT = out
        recipe.main(None, blend=True)
    a.scope = stage


def export(a):
    module, options, angle, out, files = tail_context(a)
    module.pack(options, angle, out, files)
    package = out / 'h51_retrained_package'
    destination = a.work.parent / 'model'
    shutil.copytree(package, destination)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['teacher', 'cache', 'closure', 'sim', 'point', 'native_uq',
                                    'shape', 'mean_cache', 'mean', 'tail', 'export'])
    p.add_argument('--scope', default='full')
    p.add_argument('--track', type=int, choices=[1, 2], default=2)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args()
    a.data, a.work = a.data.resolve(), a.work.resolve()
    import torch
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    globals()[a.stage](a)
    write(a.work / 'resources' / f'{a.stage}_{a.scope}.json', {
        'stage': a.stage, 'scope': a.scope, 'seconds': time.monotonic() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'complete': True})


if __name__ == '__main__':
    main()
