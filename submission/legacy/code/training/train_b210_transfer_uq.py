#!/usr/bin/env python3
"""Plan, or explicitly execute, one point-frozen Native-UQ arm.

Each arm uses fresh seeded UQ or retains the bound Real4000 point's UQ,
trains exactly 3200 updates, and seals once. Full-time fit support is opt-in;
held trajectories remain excluded in either mode.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

SCHEMA = "b210_transfer_uq_v1"
POINT_SCHEMA = "b210_transfer_real4k_v1_endpoint"
POINT_RECEIPT_SCHEMA = "b210_transfer_real4k_v1_point_receipt"
UPDATES, SEED, BATCH_SIZE = 3200, 18, 8
INITIALIZATIONS = {"fresh": "fresh_same_seed18", "point": "point_checkpoint"}
WEIGHTED_OBJECTIVE = "sps-weighted-quantile"
REFINEMENT_PREFIX = "utility_refinement_head."
C_POINT_RECEIPT_SHA = "3c9d425e1433ecce4e5d3ce5f94e968f17c10b05c32eae50187aa55481e9a891"
C_POINT_CHECKPOINT_SHA = "96c0082c921c2aaeb449ef14ba5f94a7361395164b2064a191c407c337509903"
C_POINT_FROZEN_SHA = "b818e67a39292dc53aaa2af266a25fe7b3f0fbc1539ade1c7bfc52649d34b3f3"
C_INITIAL_UQ_SHA = "2c9f6b3781fc6d9950e08590d77c2d51300ec1e61bfd1d250a994871d598af29"
MANIFEST_SHA = "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e"
PAIR_SCHEMA = "b210_uq_ema999_pair_v1"
UQ_PARAMETERS = 3680422
UQ_STREAM_SHA = "d17a0520286fde93c61c8c29d40342d8c5e629f05c8950ee5f1af075e8d223f3"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uq-physical-reference', action='store_true')
    parser.add_argument("--label", required=True)
    parser.add_argument("--held-aoa", type=int, choices=(0, 5, 10, 15, 20))
    parser.add_argument("--point-label", help="source point arm; defaults to --label")
    parser.add_argument("--seed", type=int, choices=(18, 19), default=SEED)
    parser.add_argument('--batch-size', type=int, choices=(8, 64), default=BATCH_SIZE)
    parser.add_argument('--uq-wide320', action='store_true')
    parser.add_argument('--uq-history-branch', action='store_true')
    parser.add_argument('--uq-sparse-history', action='store_true')
    parser.add_argument("--initialization", choices=tuple(INITIALIZATIONS), default="fresh")
    parser.add_argument("--objective", choices=("sharp2", WEIGHTED_OBJECTIVE), default="sharp2")
    parser.add_argument("--uq-ema999-pair", action="store_true")
    parser.add_argument("--source-commit")
    parser.add_argument('--full-real-point', action='store_true')
    parser.add_argument('--reflection-comparison', action='store_true')
    parser.add_argument('--reflection-exposure-point', action='store_true')
    parser.add_argument('--full-real-final', action='store_true')
    parser.add_argument('--uq-full-time', action='store_true',
                        help='Use all stride20 fit windows with the inherited point and normalizer')
    parser.add_argument('--uq-phase-jitter', action='store_true',
                        help='Fixed allowed-anchor time shifts, at most 19 frames, isolated seed29010')
    parser.add_argument('--uq-physical-reflection', action='store_true',
                        help='Physical H reflection at probability .25, independent CPU seed29009')
    parser.add_argument('--compact-uq', action='store_true',
                        help='Store UQ tensors with the immutable parent point binding')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    for name in ("point-receipt", "clean-manifest", "real-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--point-receipt-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wall-budget-seconds", type=float)
    args = parser.parse_args(argv)
    if args.uq_sparse_history and (not args.uq_physical_reflection or args.uq_wide320
            or args.uq_history_branch or args.held_aoa != 10 or not args.compact_uq):
        parser.error('sparse history requires the fixed held10 compact reflection combination')
    if args.uq_history_branch and (not args.uq_physical_reflection or args.uq_wide320
                                  or args.held_aoa not in (10, 20) or not args.compact_uq):
        parser.error('history branch requires the fixed held10/20 compact reflection combination')
    if args.uq_physical_reflection and (not args.uq_full_time or not args.uq_phase_jitter
            or args.batch_size != 8 or args.uq_wide320 or args.initialization != 'point'):
        parser.error('UQ reflection combination requires inherited B8 full-time jitter')
    if args.compact_uq and not args.uq_physical_reflection:
        parser.error('compact endpoint is limited to the frozen-point reflection combination')
    if args.batch_size == 64 and (not args.uq_full_time or not args.uq_phase_jitter or args.initialization != 'point'):
        parser.error('B64 requires the fixed inherited full-time phase-jitter experiment')
    if args.uq_wide320 and (args.batch_size != 8 or not args.uq_full_time or not args.uq_phase_jitter
                           or args.initialization != 'point' or args.held_aoa != 15):
        parser.error('wide320 requires the fixed held15 inherited B8 full-time jitter experiment')
    if args.seed != SEED and (args.objective != "sharp2" or args.uq_ema999_pair or args.held_aoa is not None):
        parser.error("seed19 is only the fixed full-prefix raw-UQ replication")
    if args.point_label is None:
        args.point_label = args.label
    if args.full_real_final and (not args.full_real_point or args.held_aoa is not None or args.seed != SEED):
        parser.error('final fit requires full point, seed18 and no held angle')
    if args.uq_full_time and not args.full_real_point:
        parser.error('full-time UQ requires the sealed full-Real point protocol')
    if args.uq_phase_jitter and (not args.full_real_point or args.seed != 18):
        parser.error('phase jitter requires inherited full-Real point and seed18')
    if args.reflection_exposure_point and (not args.full_real_point or args.seed != 18
                                          or args.uq_full_time or args.label == 'control'):
        parser.error('exposure point requires the fixed reflected Real5334 and prefix UQ3200 protocol')
    if args.reflection_comparison and (not args.full_real_point or args.held_aoa is None
                                      or args.label not in ('control', 'reflection')):
        parser.error('reflection comparison requires full held-AoA control/reflection')
    full_labels = ('control', 'reflection') if args.reflection_comparison else ('full',)
    if args.full_real_point and ((args.held_aoa is None and not args.full_real_final) or args.label not in full_labels
                                or args.point_label != 'C' or args.initialization != 'point'
                                or args.uq_ema999_pair):
        parser.error('full Real point requires held-AoA full/C inherited sharp2 UQ')
    if args.held_aoa is not None and (args.objective != 'sharp2' or args.uq_ema999_pair):
        parser.error('stage LOAO uses original sharp2/raw endpoint only')
    if args.objective == WEIGHTED_OBJECTIVE and (
            args.label != "weighted" or args.point_label != "C" or args.initialization != "point"):
        parser.error("weighted objective requires --label weighted --point-label C --initialization point")
    if args.uq_ema999_pair and (
            args.label != "raw" or args.point_label != "C"
            or args.initialization != "point" or args.objective != "sharp2"):
        parser.error("EMA pair requires --label raw --point-label C --initialization point --objective sharp2")
    return args


def plan(args):
    seed = getattr(args, "seed", SEED)
    result = {
        "schema": SCHEMA, "label": args.label, "decision": "PLAN_ONLY_NO_DATA_NO_GPU",
        "held_aoa": getattr(args, 'held_aoa', None),
        "updates": UPDATES, "seed": seed, "batch_size": getattr(args, 'batch_size', BATCH_SIZE),
        "uq_wide320": getattr(args, 'uq_wide320', False),
        "uq_history_branch": getattr(args, 'uq_history_branch', False),
        "uq_sparse_history": getattr(args, 'uq_sparse_history', False),
        "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-5,
                      "cosine_t_max": UPDATES, "grad_clip": 1.0},
        "initialization": (f"fresh_same_seed{seed}" if args.initialization == "fresh" else "point_checkpoint"),
        "point_source_label": args.point_label,
        "fit": ("clean81 prefix2544 only, no augmentation or teacher" if getattr(args, 'held_aoa', None) is None
                else "other-four AoA prefixes only; shared upstream Sim is not held out"),
        "objective": ("existing sharp2 Native-UQ objective, unchanged"
                      if args.objective == "sharp2" else WEIGHTED_OBJECTIVE),
        "objective_kind": args.objective,
        "checkpoint_selection": False, "intermediate_evaluation": False,
        "tail_arrays_read": 0, "point_quality_gate_before_uq": False,
        "budget_note": "3200 is a common new budget, not a replay of v3's historical optimizer lifecycle",
    }
    if getattr(args, "uq_ema999_pair", False):
        result["paired_ema"] = {"schema": PAIR_SCHEMA, "decay": .999,
                                "scope": "UQ parameters only; point and buffers copied unchanged",
                                "parameters": UQ_PARAMETERS, "labels": ["raw", "ema"],
                                "source_commit": args.source_commit, "wall_cap_seconds": 1200}
    if getattr(args, 'uq_full_time', False):
        result.update(fit='complete fit trajectories, stride20; held trajectories excluded',
                      uq_full_time=True, tail_arrays_read='fit trajectories only')
    if getattr(args, 'uq_phase_jitter', False):
        support = 'complete fit trajectories' if args.uq_full_time else 'allowed prefixes'
        result.update(fit=support + '; stride20 anchors with uniform shift0..19 clipped at support end',
                      uq_phase_jitter=True, source_commit=args.source_commit)
    if getattr(args, 'uq_physical_reflection', False):
        result['physical_reflection'] = {'probability': .25, 'isolated_cpu_seed': 29009,
                                         'space': 'physical_uvp_before_normalization'}
    if getattr(args, 'reflection_exposure_point', False):
        result.update(point_updates=5334, point_quality_gate_before_uq=True,
                      development_screen='full-fit overlapping point precheck used before stage confirmation',
                      source_commit=args.source_commit)
    return result


def validate_point_receipt(receipt, label, full_real_point=False, full_real_final=False,
                           expected_updates=4000):
    if receipt.get('schema') == 'physical_units_b210_real4000_v1':
        if (label != 'C' or not full_real_point or full_real_final or expected_updates != 4000
                or receipt.get('representation') != 'physical_units_b210_v1'
                or receipt.get('updates') != 4000 or receipt.get('seed') != 4 or receipt.get('ema_decay') != .999
                or receipt.get('held_aoa') not in (0, 5, 10, 15, 20) or receipt.get('held_data_read') is not False
                or receipt.get('optimizer_support') != 'full_other_four_trajectories'
                or receipt.get('normalizer_support') != 'other_four_prefixes'
                or not receipt.get('fit_files') or not receipt.get('checkpoint_path')
                or not re.fullmatch(r'[0-9a-f]{40}', receipt.get('source_commit', ''))
                or any(not re.fullmatch(r'[0-9a-f]{64}', receipt.get(k, '')) for k in
                       ('start_sim_sha256', 'sim_receipt_sha256', 'sample_stream_sha256',
                        'normalizer_statistics_sha256', 'checkpoint_sha256'))):
            raise ValueError('physical-units UQ requires its genuine sealed held-AoA Real4000 endpoint')
        return
    if expected_updates not in (4000, 5334):
        raise ValueError('unsupported point endpoint')
    if expected_updates == 5334 and (label != 'C' or not full_real_point
            or not re.fullmatch(r'[0-9a-f]{64}', receipt.get('start_sim_sha256', ''))
            or not re.fullmatch(r'[0-9a-f]{64}', receipt.get('start_metadata_sha256', ''))
            or receipt.get('auxiliary_enabled') is not False
            or receipt.get('vertical_reflection') != {'probability':.25, 'isolated_cpu_seed':29009,
                                                      'space':'physical_uvp_before_normalization'}
            or receipt.get('duration_intervention') != 'ceil(4000/0.75)=5334; approximately match original-view exposures'):
        raise ValueError('5334 endpoint requires the exact reflection exposure intervention')
    if (receipt.get("schema") != POINT_RECEIPT_SCHEMA
            or receipt.get("arm") != label or receipt.get("updates") != expected_updates
            or receipt.get("ema_decay") != .999 or receipt.get("tail_read") is not full_real_point
            or receipt.get("seed") != 4 or not receipt.get("start_sim_sha256")
            or not receipt.get("sample_stream_sha256")
            or not receipt.get("checkpoint_path") or not receipt.get("checkpoint_sha256")):
        raise ValueError(f"point receipt must bind this label to the expected EMA999-u{expected_updates} endpoint")
    if full_real_final:
        if (not full_real_point or receipt.get('held_aoa') is not None
                or receipt.get('optimizer_support') != 'full_clean81_trajectories'
                or receipt.get('normalizer_support') != 'clean81_prefix2544'
                or receipt.get('final_fit') is not True
                or receipt.get('clean716_overlaps_point_training') is not True
                or not re.fullmatch(r'[0-9a-f]{40}', receipt.get('source_commit', ''))):
            raise ValueError('final full Real point must disclose training overlap and prefix normalizer')
    elif full_real_point and (receipt.get('held_aoa') not in (0, 5, 10, 15, 20)
                           or receipt.get('optimizer_support') != 'full_other_four_trajectories'
                           or receipt.get('normalizer_support') != 'other_four_prefixes'
                           or receipt.get('held_data_read') is not False
                           or not re.fullmatch(r'[0-9a-f]{40}', receipt.get('source_commit', ''))):
        raise ValueError('full Real point must preserve held-AoA exclusion and prefix normalizer')


def load_point_receipt(args):
    if sha256(args.point_receipt) != args.point_receipt_sha256.lower():
        raise ValueError("point receipt SHA changed")
    receipt = json.loads(args.point_receipt.read_text(encoding="utf-8"))
    validate_point_receipt(receipt, args.point_label, getattr(args, 'full_real_point', False),
                           getattr(args, 'full_real_final', False),
                           5334 if getattr(args, 'reflection_exposure_point', False) else 4000)
    if receipt.get('held_aoa') != getattr(args, 'held_aoa', None):
        raise ValueError('point and UQ stage fold mismatch')
    if getattr(args, 'reflection_comparison', False):
        expected = {'probability': .25, 'isolated_cpu_seed': 29009,
                    'space': 'physical_uvp_before_normalization'} if args.label == 'reflection' else None
        if receipt.get('vertical_reflection') != expected:
            raise ValueError('reflection UQ label does not match sealed point intervention')
    if args.objective == WEIGHTED_OBJECTIVE and (
            args.point_receipt_sha256.lower() != C_POINT_RECEIPT_SHA
            or receipt["checkpoint_sha256"] != C_POINT_CHECKPOINT_SHA):
        raise ValueError("weighted objective requires the exact original C point receipt/checkpoint")
    if getattr(args, "uq_ema999_pair", False) and (
            args.point_receipt_sha256.lower() != C_POINT_RECEIPT_SHA
            or receipt["checkpoint_sha256"] != C_POINT_CHECKPOINT_SHA):
        raise ValueError("EMA pair requires the exact original C point receipt/checkpoint")
    return receipt


def validate_point_payload(point, receipt):
    if receipt.get('schema') == 'physical_units_b210_real4000_v1':
        start = point.get('start', {})
        if (point.get('schema') != POINT_SCHEMA or point.get('phase') != 'real'
                or point.get('iteration') != 4000 or point.get('endpoint') != 'ema_u4000'
                or point.get('primary') is not True or point.get('ema', {}).get('updates') != 4000
                or point.get('ema', {}).get('decay') != .999
                or point.get('sample_stream_sha256') != receipt['sample_stream_sha256']
                or start.get('checkpoint_sha256') != receipt['start_sim_sha256']
                or start.get('sim_receipt_sha256') != receipt['sim_receipt_sha256']
                or start.get('stage_held_aoa') != receipt['held_aoa']
                or start.get('representation') != receipt['representation']
                or bool(start.get('joint_uq_physical_reference',False)) != bool(receipt.get('joint_uq_physical_reference',False))
                or not all('units_' + k in point['model_state_dict'] for k in
                           ('mean_input', 'mean_target', 'std_input', 'std_target'))):
            raise ValueError('physical-units payload differs from its sealed training receipt')
        return
    updates = receipt.get('updates', 4000)
    if (point.get("schema") != POINT_SCHEMA or point.get("phase") != "real"
            or updates not in (4000, 5334, 6645)
            or point.get("iteration") != updates or point.get("endpoint") != f"ema_u{updates}"
            or point.get("primary") is not True
            or point.get("ema", {}).get("updates") != updates
            or point.get("ema", {}).get("decay") != .999
            or point.get("sample_stream_sha256") != receipt["sample_stream_sha256"]
            or point.get("start", {}).get("transfer_arm") != receipt["arm"]
            or point.get("start", {}).get("checkpoint_sha256") != receipt["start_sim_sha256"]):
        raise ValueError("point checkpoint does not match its fixed Real4k receipt")
    if point.get('start', {}).get('stage_held_aoa') != receipt.get('held_aoa'):
        raise ValueError('point payload stage fold differs from receipt')
    if receipt.get('tail_read') and any(point.get('start', {}).get(k) != receipt.get(k)
                                      for k in ('optimizer_support', 'normalizer_support', 'held_data_read', 'source_commit')):
        raise ValueError('full Real point payload visibility differs from receipt')
    if updates in (5334, 6645) and any(point.get('start', {}).get(k) != receipt.get(k)
                              for k in ('duration_intervention','vertical_reflection')):
        raise ValueError('exposure intervention differs between point payload and receipt')


def reset_uq_state(model, fresh, prefixes):
    """Copy the constructor's native UQ initialization; never reset point state."""
    state = model.state_dict()
    fresh_state = fresh.state_dict()
    names = [key for key in state if key.startswith(prefixes)]
    if not names:
        raise ValueError("no native UQ state found")
    for key in names:
        state[key] = fresh_state[key].detach().clone()
    model.load_state_dict(state, strict=True)


def initialize_uq_state(model, prefixes, initialization="fresh", seed=SEED):
    """Reset or retain the loaded point's UQ without reseeding training RNG."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("unknown UQ initialization")
    if initialization == "point":
        if not any(key.startswith(prefixes) for key in model.state_dict()):
            raise ValueError("no native UQ state found")
        return
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        fresh = type(model)(model.config).float()
    reset_uq_state(model, fresh, prefixes)


def make_uq_ema(model, prefixes):
    """Alias only trainable UQ modules; constructing the view consumes no RNG."""
    import torch
    import fp32_ema_shadow as ema_lib
    view = torch.nn.ModuleDict({prefix[:-1]: getattr(model, prefix[:-1])
                               for prefix in prefixes if getattr(model, prefix[:-1], None) is not None})
    expected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    actual = dict(view.named_parameters())
    if (not actual or set(actual) != set(expected)
            or any(not name.startswith(prefixes) or actual[name] is not expected[name] for name in actual)):
        raise ValueError("EMA must cover every trainable UQ parameter and no point parameter")
    return view, ema_lib.FP32EMAShadow(view, .999)


def materialize_uq_ema(raw_state, view, ema):
    """Replace UQ parameter values only; all buffers and raw tensors remain intact."""
    report = ema.seal(view)
    averaged = report.pop("model_state")
    parameter_names = {name for name, _ in view.named_parameters()}
    state = {name: averaged[name] if name in parameter_names else value
             for name, value in raw_state.items()}
    return state, report


def remove_utility_refinement(model, model_config):
    """Remove only the decision refinement after strict original-state loading."""
    if (model_config.get("utility_refinement") is not True
            or not model.config.utility_refinement or model.utility_refinement_head is None):
        raise ValueError("weighted objective requires original utility refinement")
    model.config = replace(model.config, utility_refinement=False)
    model.utility_refinement_head = None
    return {**model_config, "utility_refinement": False}


def sps_weighted_quantile_loss(model, prediction, quantiles, target, normalizer):
    """Proper pinball under detached point-SPS weights; original physical mask."""
    import torch
    import train_hybrid_fno2d as common
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    residual = (target[..., :2] - prediction[..., :2].detach()) * scale
    error = residual.unsqueeze(-1) - quantiles * scale.unsqueeze(-1)
    levels = quantiles.new_tensor(model.config.quantile_levels).view(1, 1, 1, 1, 1, -1)
    pinball = torch.maximum(levels * error, (levels - 1.0) * error)
    reference = quantiles.new_tensor([0.01075, 0.00550]).view(1, 1, 1, 1, 2, 1)
    target_physical = common.to_physical_target(target, normalizer)[..., :2]
    prediction_physical = common.to_physical_target(prediction.detach(), normalizer)[..., :2]
    factor = common.sample_sps_factor(prediction_physical, target_physical).detach()
    scored = target_physical != 0.0
    weights = (factor.view(-1, 1, 1, 1, 1, 1)
               * scored.unsqueeze(-1))
    denominator = (factor * scored.sum(dim=(1, 2, 3, 4))).sum() * quantiles.shape[-1]
    return ((pinball / reference) * weights).sum() / denominator


def load_uq_rows(real_dir, rows, full_time=False):
    import train_b210_small4k_uq_release as native
    from train_b210_full_prefix_uq_refit import PrefixOnlyAudit
    audit = PrefixOnlyAudit()
    if full_time:
        from train_b210_full_released_real_support import load_full_rows, support_windows
        trajectories, metadata = load_full_rows(real_dir, rows, 'fit', audit)
        windows = support_windows(trajectories, metadata, support='full', stride=20)
        expected = sum((r['frames'] - 40) // 20 + 1 for r in rows)
    else:
        trajectories, metadata, windows = native.dr1.load_prefix_rows(real_dir, rows, 'fit', audit)
        expected = sum(len(r['train_starts']) for r in rows)
    report = audit.report(rows)
    report.update(prefix_only=not full_time, tail_arrays_read=len(rows) if full_time else 0,
                  held_arrays_read=0, optimizer_stride=20)
    if len(windows) != expected or not report['allowlist_exact']:
        raise RuntimeError('UQ support or access allowlist changed')
    return trajectories, metadata, windows, report


def execute(args):
    seed = getattr(args, "seed", SEED)
    batch_size = getattr(args, 'batch_size', BATCH_SIZE)
    paired = getattr(args, "uq_ema999_pair", False)
    pair_started, pair_epoch = time.perf_counter(), time.time()
    required = ("point_receipt", "point_receipt_sha256", "clean_manifest",
                "real_dir", "output_dir", "wall_budget_seconds")
    if any(getattr(args, key) is None for key in required):
        raise ValueError("explicit execution requires all asset bindings and a wall budget")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", label)
           for label in (args.label, args.point_label)):
        raise ValueError("label and point label must be simple identifiers")
    if not math.isfinite(args.wall_budget_seconds) or args.wall_budget_seconds <= 0:
        raise ValueError("wall budget must be positive and finite")
    if batch_size == 64 and (args.wall_budget_seconds > 2400 or not re.fullmatch(r'[0-9a-f]{40}', args.source_commit or '')):
        raise ValueError('B64 requires a bound source commit and at most 2400 seconds')
    if getattr(args, 'uq_wide320', False) and (args.wall_budget_seconds > 1800 or not re.fullmatch(r'[0-9a-f]{40}', args.source_commit or '')):
        raise ValueError('wide320 requires a bound source commit and at most 1800 seconds')
    if paired and (args.wall_budget_seconds > 1200
                   or not re.fullmatch(r"[0-9a-f]{40}", args.source_commit or "")):
        raise ValueError("EMA pair requires a full source commit and at most 1200 seconds")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    receipt = load_point_receipt(args)
    point_path = Path(receipt["checkpoint_path"])
    if sha256(point_path) != receipt["checkpoint_sha256"]:
        raise ValueError("point checkpoint SHA changed")
    if sha256(args.clean_manifest) != MANIFEST_SHA:
        raise ValueError("clean prefix manifest SHA changed")

    for path in (Path(__file__).resolve().parents[1], Path(__file__).parent):
        sys.path.insert(0, str(path))
    import torch
    from torch.utils.data import DataLoader
    import train_b210_full_prefix_uq_refit as prefix
    import train_b210_small4k_uq_release as native
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal UQ execution requires CUDA; plan-only never imports torch")
    deadline = (pair_started if paired else time.perf_counter()) + args.wall_budget_seconds
    point = torch.load(point_path, map_location="cpu", weights_only=False)
    validate_point_payload(point, receipt)
    config = dict(point["model_config"])
    if config.pop("architecture") != "ensemble_unet":
        raise ValueError("wrong point architecture")
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**config)).float()
    physical_units = receipt.get('schema') == 'physical_units_b210_real4000_v1'
    if physical_units:
        from types import SimpleNamespace
        from models.physical_units_b210 import install
        install(model, SimpleNamespace(**point['normalizer']), uq_global=getattr(args, 'uq_physical_reference', False))
    elif getattr(args, 'uq_physical_reference', False):
        raise ValueError('physical UQ reference is only valid for a physical-units point receipt')
    model.load_state_dict(point["model_state_dict"], strict=True)
    frozen_before = native.dr1.model_state_sha256(native.partition_state(model, uq=False))
    initialize_uq_state(model, native.UQ_PREFIXES, args.initialization, seed)
    initial_uq_sha = native.dr1.model_state_sha256(native.partition_state(model, uq=True))
    if native.dr1.model_state_sha256(native.partition_state(model, uq=False)) != frozen_before:
        raise RuntimeError("UQ initialization changed point state")
    if paired and (frozen_before != C_POINT_FROZEN_SHA or initial_uq_sha != C_INITIAL_UQ_SHA):
        raise ValueError("EMA pair must start from original C Real4000 point/UQ, never UQ3200")
    output_config = point["model_config"]
    objective_report = {"objective_kind": args.objective}
    reference_model = None
    if getattr(args, 'uq_sparse_history', False):
        from dataclasses import replace
        model.config = replace(model.config, uq_sparse_history_context=True)
        output_config = {**output_config, 'uq_sparse_history_context': True}
        objective_report['sparse_history'] = {'indices': [0, 4, 8, 12, 16], 'recent_differences_unchanged': True,
                                             'parameter_shapes_unchanged': True}
    if getattr(args, 'uq_history_branch', False):
        from extend_native_uq_history import extend_history_uq
        model, history_branch = extend_history_uq(model, native.UQ_PREFIXES)
        assert native.dr1.model_state_sha256(native.partition_state(model, uq=False)) == frozen_before
        objective_report.update(source_initial_uq_sha256=initial_uq_sha, history_branch=history_branch)
        initial_uq_sha = native.dr1.model_state_sha256(native.partition_state(model, uq=True))
        output_config = {**output_config, 'uq_full_history_context': True}
    if getattr(args, 'uq_wide320', False):
        from widen_native_uq import widen_uq
        assert model.config.uq_hidden_channels == 160
        reference_model = model
        model, widening = widen_uq(reference_model, native.UQ_PREFIXES)
        assert native.dr1.model_state_sha256(native.partition_state(model, uq=False)) == frozen_before
        objective_report.update(source_initial_uq_sha256=initial_uq_sha, widening=widening)
        initial_uq_sha = native.dr1.model_state_sha256(native.partition_state(model, uq=True))
        output_config = {**output_config, 'uq_hidden_channels': 320}
    if args.objective == WEIGHTED_OBJECTIVE:
        if frozen_before != C_POINT_FROZEN_SHA or initial_uq_sha != C_INITIAL_UQ_SHA:
            raise ValueError("weighted objective original C point/UQ state identity changed")
        output_config = remove_utility_refinement(model, output_config)
        if native.dr1.model_state_sha256(native.partition_state(model, uq=False)) != frozen_before:
            raise RuntimeError("removing UQ refinement changed frozen point state")
        objective_report.update({
            "source_initial_uq_sha256": initial_uq_sha,
            "retained_initial_uq_sha256": native.dr1.model_state_sha256(native.partition_state(model, uq=True)),
            "removed_state_prefixes": [REFINEMENT_PREFIX],
            "model_config_overrides": {"utility_refinement": False},
        })
    initialization_report = {
        "initialization": (f"fresh_same_seed{seed}" if args.initialization == "fresh" else "point_checkpoint"),
        "point_source_label": args.point_label,
        "point_receipt": receipt, "point_receipt_sha256": args.point_receipt_sha256,
        "initial_uq_sha256": initial_uq_sha,
        **objective_report,
    }
    if physical_units:
        initialization_report.update(point_source_label='physical_units',
                                     representation='physical_units_b210_v1',
                                     uq_physical_reference=getattr(args, 'uq_physical_reference', False))
    normalizer, normalizer_report = prefix.normalizer_from_point(point, device)
    model = model.to(device)
    rows, manifest_report = native.dr1.load_manifest(args.clean_manifest, native.dr1.FORMAL_MANIFEST_SHA256)
    from b210_stage_fold import select_rows
    rows = select_rows(rows, getattr(args, 'held_aoa', None))
    if getattr(args, 'held_aoa', None) is not None and receipt.get('fit_files') != [r['file'] for r in rows]:
        raise ValueError('Real and UQ training allowlists differ')
    full_time = getattr(args, 'uq_full_time', False)
    trajectories, metadata, windows, access_report = load_uq_rows(args.real_dir, rows, full_time)
    dataset = native.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    if getattr(args, 'uq_phase_jitter', False):
        from prefix_time_jitter_dataset import TimeJitterDataset
        dataset = TimeJitterDataset(dataset, full_time=full_time)
        access_report.update(optimizer_stride=None, base_anchor_stride=20)
    tagged = native.ensemble_train.TrajectoryTaggedDataset(dataset)
    native.uq_train.seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    parameters = native.configure_uq_only(model)
    loader = DataLoader(tagged, batch_size=batch_size, shuffle=True, num_workers=0,
                        pin_memory=True, generator=torch.Generator().manual_seed(seed))
    batches = native.ensemble_train.common.repro.cycling(loader)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    uq_view, uq_ema = make_uq_ema(model, native.UQ_PREFIXES) if paired else (None, None)
    if paired and sum(parameter.numel() for parameter in uq_view.parameters()) != UQ_PARAMETERS:
        raise ValueError("EMA pair UQ parameter count changed")
    objective = native.sharp2_args()
    sample_stream = hashlib.sha256()
    samples_seen = 0
    reflection_enabled = getattr(args, 'uq_physical_reflection', False)
    if reflection_enabled:
        from vertical_reflection_dataset import reflect_uvp
        reflection_rng = torch.Generator().manual_seed(29009)
        reflection_stream = hashlib.sha256()
        reflected = 0
    started = time.perf_counter()
    for step in range(1, UPDATES + 1):
        if time.perf_counter() > deadline:
            raise TimeoutError("UQ budget expired; no endpoint is sealed")
        native.set_uq_training_mode(model)
        history, target, _trajectory, indices = next(batches)
        samples_seen += len(indices)
        sample_stream.update(indices.numpy().astype("<i8", copy=False).tobytes())
        if reflection_enabled:
            mask = torch.rand(len(history), generator=reflection_rng) < .25
            reflected += int(mask.sum())
            reflection_stream.update(mask.numpy().astype('uint8').tobytes())
            history[mask] = reflect_uvp(history[mask])
            target[mask] = reflect_uvp(target[mask])
        x, y = normalizer.preprocess(history.to(device), target.to(device))
        if reference_model is not None:
            from widen_native_uq import check_initial_function
            objective_report['widening']['initial_function'] = check_initial_function(reference_model, model, x, normalizer)
            print(json.dumps({'widening': objective_report['widening']}), flush=True)
            del reference_model
            reference_model = None
            torch.cuda.empty_cache()
            native.set_uq_training_mode(model)
        optimizer.zero_grad(set_to_none=True)
        prediction, quantiles, extents, logits = native.uq_train.quantile_forward(model, x)
        if args.objective == "sharp2":
            loss, _ = native.uq_train.uq_objective(objective, model, prediction, quantiles, y,
                                                  normalizer, interval_extents=extents, coverage_logits=logits)
        else:
            loss = sps_weighted_quantile_loss(model, prediction, quantiles, y, normalizer)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite UQ objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not bool(torch.isfinite(norm)) or (step == 1 and float(norm) == 0):
            raise FloatingPointError("invalid UQ gradient")
        optimizer.step()
        if uq_ema is not None:
            uq_ema.update_after_optimizer_step(uq_view, optimizer_step=step)
        scheduler.step()
        if step % 400 == 0:
            print(json.dumps({"label": args.label, "step": step, "training_loss": float(loss),
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    frozen_after = native.dr1.model_state_sha256({key: value for key, value in state.items()
                                                 if not key.startswith(native.UQ_PREFIXES)})
    if frozen_after != frozen_before:
        raise RuntimeError("UQ changed frozen point/backbone/member/controller state")
    if getattr(args, 'uq_history_branch', False):
        start, end = history_branch['inserted_channels']
        learned = model.uq_fusion[0].weight.detach()[:, start:end]
        norm = float(learned.norm())
        assert bool(torch.isfinite(learned).all()) and norm > 0
        history_branch.update(new_channel_weight_norm=norm, new_channels_updated=True)
    if getattr(args, 'uq_phase_jitter', False):
        jitter_report = dataset.report()
        if jitter_report['max_support_overrun'] or min(jitter_report['shift_histogram']) == 0:
            raise RuntimeError('time jitter boundary or coverage check failed')
        initialization_report['sampling_intervention'] = jitter_report
        access_report['sampling_intervention'] = jitter_report
    if reflection_enabled:
        initialization_report['physical_reflection'] = {
            **plan(args)['physical_reflection'], 'draws': samples_seen,
            'reflected': reflected, 'mask_sha256': reflection_stream.hexdigest()}
    pair_report = None
    if paired:
        if uq_ema.updates != UPDATES or sample_stream.hexdigest() != UQ_STREAM_SHA:
            raise RuntimeError("EMA count or full inherited-C sampling stream changed")
        ema_state, ema_report = materialize_uq_ema(state, uq_view, uq_ema)
        parameter_names = {name for name, _ in uq_view.named_parameters()}
        if any(not torch.equal(value, ema_state[name]) for name, value in state.items()
               if name not in parameter_names):
            raise RuntimeError("UQ EMA changed a point parameter or buffer")
        pair_report = {
            "schema": PAIR_SCHEMA, "started_epoch": pair_epoch,
            "source_commit": args.source_commit, "source_sha256": sha256(Path(__file__)),
            "point_receipt_sha256": args.point_receipt_sha256,
            "point_checkpoint_sha256": receipt["checkpoint_sha256"],
            "initial_uq_sha256": initial_uq_sha, "point_frozen_sha256": frozen_after,
            "sample_stream_sha256": sample_stream.hexdigest(), "seed": seed, "updates": UPDATES,
            "model_config_sha256": native.dr1.canonical_sha256(output_config),
            "normalizer_sha256": normalizer_report["sha256"], "parameters": UQ_PARAMETERS,
            "gpu": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda, "precision": "FP32; float32_matmul_precision=high",
            "raw_uq_state_sha256": native.dr1.model_state_sha256(native.partition_state(model, uq=True)),
            "ema_uq_state_sha256": native.dr1.model_state_sha256(
                {name: value for name, value in ema_state.items() if name.startswith(native.UQ_PREFIXES)}),
            "non_uq_and_all_buffers_bitwise": True,
        }
        pair_report["trajectory_sha256"] = native.dr1.canonical_sha256(pair_report)
        if time.perf_counter() > deadline:
            raise TimeoutError("EMA pair budget expired before endpoint writing")
    args.output_dir.mkdir(parents=True)
    endpoint = args.output_dir / ('fulltime_uq3200.pth' if full_time else 'prefix_uq3200.pth')
    compact = getattr(args, 'compact_uq', False)
    saved_state = {k: v for k, v in state.items() if k.startswith(native.UQ_PREFIXES)} if compact else state
    payload = {"schema": SCHEMA + "_endpoint", "phase": "real", "iteration": UPDATES,
               "model_config": output_config, "model_state_dict": saved_state,
               "compact_uq": compact,
               "normalizer": point["normalizer"], "normalizer_sha256": normalizer_report["sha256"],
               **initialization_report}
    if paired:
        payload.update(variant="raw", paired_training=pair_report, ema=None)
    torch.save(payload, endpoint)
    seal = {"schema": SCHEMA + "_receipt", "label": args.label, "sealed": True,
            "held_aoa": getattr(args, 'held_aoa', None),
            "holdout_scope": "Real/UQ stages only; shared Sim ancestors are not held out",
            "checkpoint_path": str(endpoint.resolve()), "checkpoint_sha256": sha256(endpoint),
            "compact_uq": compact,
            **initialization_report,
            "updates": UPDATES, "seed": seed, "fit_windows": len(windows),
            "batch_size": batch_size, "samples_seen": samples_seen,
            "manifest_sha256": MANIFEST_SHA, "manifest": manifest_report,
            "normalizer_sha256": normalizer_report["sha256"],
            "model_config_sha256": native.dr1.canonical_sha256(output_config),
            "point_frozen_sha256": frozen_after, "point_state_bitwise": True,
            "sample_stream_sha256": sample_stream.hexdigest(), "tail_arrays_read": access_report['tail_arrays_read'],
            "checkpoint_selection": False, "access_audit": access_report,
            "training": {**plan(args), "decision": "FIXED_TERMINAL_SEALED",
                         "objective": (vars(objective) if args.objective == "sharp2" else {
                             "kind": WEIGHTED_OBJECTIVE, "coefficient": 1.0,
                             "quantile_levels": list(model.config.quantile_levels),
                             "channel_references": [0.01075, 0.00550],
                             "point_factor": "detached common.sample_sps_factor",
                             "reduction": "weighted scored element and quantile mean"})},
            "wall_seconds": time.perf_counter() - started}
    if physical_units:
        seal['holdout_scope'] = 'Pure Sim pretraining used no Real; Real/UQ exclude this AoA; discovery angles reused'
    if paired:
        ema_path = args.output_dir / "prefix_uq3200_ema999.pth"
        torch.save({**payload, "model_state_dict": ema_state, "variant": "ema999", "ema": ema_report}, ema_path)
        ema_sha256 = sha256(ema_path)
        if time.perf_counter() > deadline:
            raise TimeoutError("EMA pair budget expired; dual endpoints not admitted")
        seal.update(variant="raw", paired_training=pair_report, ema=None,
                    wall_seconds=time.perf_counter() - pair_started)
        ema_seal = {**seal, "label": "ema", "variant": "ema999", "ema": ema_report,
                    "checkpoint_path": str(ema_path.resolve()), "checkpoint_sha256": ema_sha256}
        native.dr1.write_json_atomic(args.output_dir / "uq_ema999_endpoint.json", ema_seal)
    native.dr1.write_json_atomic(args.output_dir / "uq_endpoint.json", seal)
    if paired:
        native.dr1.write_json_atomic(args.output_dir / "uq_ema999_pair.json", {
            **pair_report, "sealed": True,
            "receipts": {label: {"path": str((args.output_dir / name).resolve()),
                                 "sha256": sha256(args.output_dir / name)}
                         for label, name in (("raw", "uq_endpoint.json"), ("ema", "uq_ema999_endpoint.json"))}})
    return seal


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(execute(arguments) if arguments.execute else plan(arguments), indent=2))
