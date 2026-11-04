"""Official Final-phase entry points. Track2 L33 reproduction package."""
from pathlib import Path
import importlib.util
import sys


def train(data_dir, output_dir, device):
    from pipeline import train as run
    return run(data_dir, output_dir, device, track=2)


def get_ttt_model(output_dir, device):
    directory = Path(output_dir).resolve() / 'model'
    if not (directory / 'ensemble_unet2d.pth').is_file():
        raise FileNotFoundError('No trained model; train() must finish before inference.')
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('final_track2_runtime', directory / 'submission.py')
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)
    return runtime.get_ttt_model(str(directory), device)
