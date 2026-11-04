"""Reject duration or intervention substitution before loading a point model."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_b210_transfer_uq as subject


def receipt():
    return {'schema':subject.POINT_RECEIPT_SCHEMA,'arm':'C','updates':5334,'ema_decay':.999,
            'tail_read':True,'seed':4,'held_aoa':None,'optimizer_support':'full_clean81_trajectories',
            'normalizer_support':'clean81_prefix2544','final_fit':True,'clean716_overlaps_point_training':True,
            'source_commit':'a'*40,'sample_stream_sha256':'b'*64,'checkpoint_path':'sealed.pth',
            'checkpoint_sha256':'c'*64,'auxiliary_enabled':False,
            'start_sim_sha256':'628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b',
            'start_metadata_sha256':'282307574088299de967a7d64597a417150dd81cb18ea7b8d2897ee29b00e96f',
            'duration_intervention':'ceil(4000/0.75)=5334; approximately match original-view exposures',
            'vertical_reflection':{'probability':.25,'isolated_cpu_seed':29009,'space':'physical_uvp_before_normalization'}}


def test_5334_requires_explicit_mode_and_original_4000_does_not_enter_it():
    r=receipt()
    with pytest.raises(ValueError):subject.validate_point_receipt(r,'C',True,True)
    subject.validate_point_receipt(r,'C',True,True,5334)
    r['updates']=4000
    with pytest.raises(ValueError):subject.validate_point_receipt(r,'C',True,True,5334)
    subject.validate_point_receipt(r,'C',True,True)


@pytest.mark.parametrize('field,value',[('duration_intervention',''),('vertical_reflection',None),
                                      ('start_sim_sha256','d'*64),('auxiliary_enabled',True)])
def test_other_interventions_cannot_enter_5334_uq(field,value):
    r=receipt();r[field]=value
    with pytest.raises(ValueError):subject.validate_point_receipt(r,'C',True,True,5334)


def test_payload_duration_and_exposure_are_bound_to_receipt():
    r=receipt()
    point={'schema':subject.POINT_SCHEMA,'phase':'real','iteration':5334,'endpoint':'ema_u5334','primary':True,
           'ema':{'updates':5334,'decay':.999},'sample_stream_sha256':r['sample_stream_sha256'],
           'start':{**r,'transfer_arm':'C','checkpoint_sha256':r['start_sim_sha256'],'stage_held_aoa':None}}
    subject.validate_point_payload(point,r)
    bad=deepcopy(point);bad['ema']['updates']=4000
    with pytest.raises(ValueError):subject.validate_point_payload(bad,r)
    bad=deepcopy(point);bad['start']['duration_intervention']=''
    with pytest.raises(ValueError):subject.validate_point_payload(bad,r)


def test_exposure_cli_keeps_prefix_only_seed18_and_discloses_screen():
    flags=['--label','full','--point-label','C','--full-real-point','--full-real-final',
           '--initialization','point','--reflection-exposure-point']
    args=subject.parse_args(flags)
    assert subject.plan(args)['point_updates']==5334
    assert subject.plan(args)['point_quality_gate_before_uq']
    with pytest.raises(SystemExit):subject.parse_args(flags+['--uq-full-time'])
    with pytest.raises(SystemExit):subject.parse_args(flags+['--seed','19'])
