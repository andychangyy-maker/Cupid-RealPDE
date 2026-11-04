import json,copy
from types import SimpleNamespace
from architecture_health_policy import apply_policy,install,POLICY
checks=dict(finite_metrics=True,update0_bitwise_identity=True,tke_controller_gradient_ratio=False,controller_state_updated=True,controller_projection_updated=True,controller_front_updated=True,controller_projection_gradient_seen=True,controller_front_gradient_seen=True,point_not_catastrophic=False,components_not_catastrophic=False,sps_not_catastrophic=False,hgrad_audits_finite=True)
raw=dict(health_checks=checks,health_pass=False,status='STOP',directional_pass=False,directional_is_hard_gate=False)
def run(ratios=None,changes=None):
 x=copy.deepcopy(raw)
 x['health_checks'].update(changes or {})
 return apply_policy(x,[2.0]*200 if ratios is None else ratios,200,'PASS','STOP')
x=run();assert x['health_pass'] and x['original_gate_result']==raw and raw['health_pass'] is False
assert not run([2.0]*199)['health_pass']
assert not run([float('nan')]*200)['health_pass']
assert not run([-1.0]*200)['health_pass']
for k in ('finite_metrics','update0_bitwise_identity','controller_state_updated','controller_projection_updated','controller_front_updated','controller_projection_gradient_seen','controller_front_gradient_seen','hgrad_audits_finite'):
 assert not run(changes={k:False})['health_pass'],k
original=lambda *args:raw
module=SimpleNamespace(prefix_gate=original)
assert install(module,210,4) is False and module.prefix_gate is original
print(json.dumps(dict(passed=True,cpu_only=True,score_and_ratio_range_diagnostic=True,all_original_engineering_checks_preserved=True,baseline210_4_unchanged=True,policy=POLICY),indent=2))