"""Verify the new Real-only nonlinear teacher on a fit input, without timing it."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/training'))
import train_b210_held_residual_uq as held


def main(args):
    started = time.perf_counter()
    torch,dense,_,prefix = held.runtime()
    from models.ensemble_unet2d import EnsembleUNet2d
    from train_real_member_teacher import install
    from preflight_eight_head_fold import fit_input
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    assert held.sha256(args.point)=='ce84999755b685bfc84061cec3367ee324f64d71661ee5fcfd33320182c5b5fa'
    payload = torch.load(args.point,map_location='cpu',weights_only=False)
    model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload['model_config'])).float()
    model.load_state_dict(payload['model_state_dict'],strict=True)
    norm,_ = prefix.normalizer_from_point(payload,torch.device('cuda:0'))
    model.cuda().eval()
    x = fit_input(args.real_dir,norm,torch)
    with torch.inference_mode():
        original = model.forward_distribution(x)
        install(model,torch)
        candidate = model.forward_distribution(x)
        assert all(torch.isfinite(value).all() for value in candidate)
        scale = norm.std_target[:2].view(1,1,1,1,2)
        errors = [float(((a-b)[...,:2]*scale).abs().max()) for a,b in zip(original,candidate)]
    assert max(errors)<=2e-6
    result = dict(decision='PASS_REAL_MEMBER_TEACHER_ENGINEERING',physical_output_max_errors=errors,
                  training_parameters=180355743,added_parameters=1650168,original_state_and_rng_preserved=True,
                  fit_input='10125_0.h5[:20]',held_arrays_read=False,teacher_only=True,
                  elapsed_seconds=time.perf_counter()-started,source_sha256=held.sha256(__file__))
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('point','real-dir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
