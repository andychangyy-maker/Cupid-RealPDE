"""Full Real4000 after sealed pure-Sim control or physical-units Sim10000."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace

import train_b210_transfer_real4k as base


def main(args):
    started=time.perf_counter()
    base.require_sha(args.sim_receipt,args.sim_receipt_sha256)
    sim=json.loads(args.sim_receipt.read_text())
    warm=sim['schema']=='physical_units_csim_warmstart_v1'
    units=sim['schema']=='physical_units_b210_sim10000_v1' or warm
    joint_reference=getattr(args,'joint_uq_physical_reference',False)
    assert bool(sim.get('joint_uq_physical_reference',False))==joint_reference
    assert not joint_reference or units
    assert sim['schema'] in ('physical_units_b210_sim10000_v1','b210_pure_sim_start_v1','physical_units_csim_warmstart_v1')
    if warm:
        assert sim['initial_checkpoint_sha256']==base.C_START_SIM_SHA256
        assert sim['start']=='sealed_C_causal4k_standard20' and joint_reference
    assert sim['sealed'] and sim['updates']==10000 and sim['seed']==0
    assert sim['real_arrays_read']==sim['real_statistics_read']==sim['teacher_reads']==0
    assert sim['p1_distill_weight']==0 and sim['sample_stream_sha256']=='bd52d8197e98364285405ba700d0e5acf286f143d14405757b751ecaaf8e4130'
    assert not args.output_dir.exists()
    assert 0<args.wall_budget_seconds<=1800
    checkpoint=Path(sim['checkpoint_path']);base.require_sha(checkpoint,sim['checkpoint_sha256'])
    dense=base.load_runtime()
    torch,dr1,smu1,ft12=dense.torch,dense.dr1,dense.smu1,dense.ft12
    from b210_stage_fold import select_rows
    from train_b210_full_released_real_support import load_full_rows
    from models.ensemble_unet2d import EnsembleUNet2d
    from models.physical_units_b210 import install
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    device=torch.device('cuda:0')
    base.require_sha(args.manifest,'efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e')
    rows,manifest=dr1.load_manifest(args.manifest,ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    rows=select_rows(rows,args.held_aoa)
    audit=dense.PrefixAudit()
    prefix,metadata,prefix_windows=dr1.load_prefix_rows(args.real_dir,rows,'fit',audit)
    statistics=smu1.fold_statistics(smu1.window_sufficient_statistics(prefix,prefix_windows),range(len(prefix_windows)))
    normalizer=smu1.FoldNormalizer(statistics,device)
    del prefix
    full_audit=dense.PrefixAudit()
    trajectories,metadata=load_full_rows(args.real_dir,rows,'fit',full_audit)
    assert set(audit.opened)==set(full_audit.opened)=={r['file'] for r in rows}
    windows=dense.dense_windows(trajectories,5)
    assert len(windows)==sum((r['frames']-40)//5+1 for r in rows)
    dataset=ft12.npt1.ensemble_train.common.repro.WindowDataset(trajectories,windows)
    schedule=smu1.paired_batch_schedule(range(len(windows)),windows,updates=4000,seed=4);schedule['updates']=4000
    ft12.npt1.ensemble_train.seed_everything(4)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=EnsembleUNet2d(ft12.validate_b210_model_config(payload['model_config'])).float()
    if units:install(model,SimpleNamespace(**payload['normalizer']),uq_global=joint_reference)
    model.load_state_dict(payload['model_state_dict'],strict=True)
    model=model.to(device)
    ft12.npt1.ensemble_train.common.set_model_stats(model,normalizer)
    del payload
    start=dict(path=str(checkpoint),checkpoint_sha256=sim['checkpoint_sha256'],iteration=10000,
               status=sim['decision'],representation='physical_units_b210_v1' if units else 'original_global_coordinates',
               source_commit=args.source_commit,sim_receipt_sha256=args.sim_receipt_sha256,
               stage_held_aoa=args.held_aoa,training_visibility='full_other_four_trajectories')
    start['joint_uq_physical_reference']=joint_reference
    if warm:
        start['inherited_pretraining']=sim['inherited_pretraining']
        start['initial_checkpoint_sha256']=sim['initial_checkpoint_sha256']
    args.output_dir.mkdir(parents=True)
    torch.save(tuple(v.detach().cpu() for v in statistics),args.output_dir/'point_normalizer.pt')
    access=dense.ArmSeal('units' if units else 'control')
    with base.endpoint_contract(dense.small4k,args.held_aoa,4):
        training,seal=dense.small4k.train_candidate(model,dataset,schedule,normalizer,device,
                started+args.wall_budget_seconds,args.output_dir,start,access,
                input_augmentation=dense.AUGMENTATION,gradient_ratio_diagnostics_only=True)
    assert access.value==seal
    result=dict(schema='physical_units_b210_real4000_v1',updates=4000,seed=4,ema_decay=.999,
                held_aoa=args.held_aoa,fit_files=[r['file'] for r in rows],held_data_read=False,
                checkpoint_path=seal['path'],checkpoint_sha256=seal['file_sha256'],
                start_sim_sha256=sim['checkpoint_sha256'],sim_receipt_sha256=args.sim_receipt_sha256,
                representation=start['representation'],sample_stream_sha256=schedule['sha256'],
                normalizer_statistics_sha256=smu1.statistics_digest(statistics),
                optimizer_support='full_other_four_trajectories',normalizer_support='other_four_prefixes',
                joint_uq_physical_reference=joint_reference,
                source_commit=args.source_commit,source_sha256=base.sha256(Path(__file__)),
                elapsed_seconds=time.perf_counter()-started,training=training)
    if warm:
        result['inherited_pretraining']=sim['inherited_pretraining']
        result['initial_checkpoint_sha256']=sim['initial_checkpoint_sha256']
    (args.output_dir/'point_endpoint.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('representation','held_aoa','checkpoint_path','checkpoint_sha256','elapsed_seconds')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('sim-receipt','manifest','real-dir','output-dir'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--sim-receipt-sha256',required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--held-aoa',type=int,choices=(0,5,10,15,20),required=True)
    p.add_argument('--joint-uq-physical-reference',action='store_true')
    p.add_argument('--wall-budget-seconds',type=float,required=True)
    main(p.parse_args())
