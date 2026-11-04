"""Candidate-only continuation policy; original diagnostics remain complete."""
import copy
import math

POLICY = {
 'schema': 'architecture_sim_health_policy_v1',
 'scope': 'base/depth variants only; original 210/4 unchanged',
 'reason': 'User requires Real validation regardless of finite Sim relative scores.',
 'diagnostic_only': ['point_not_catastrophic','components_not_catastrophic','sps_not_catastrophic','tke_controller_gradient_ratio_range'],
 'unchanged': 'losses, optimizer, samples, fixed updates, original diagnostics, nonfinite/OOM/runtime safeguards',
}

def apply_policy(original, gradient_ratios, expected_samples, pass_status, stop_status):
 result=copy.deepcopy(original)
 checks=dict(original['health_checks'])
 removed={k:checks.pop(k) for k in POLICY['diagnostic_only'][:3]}
 removed['tke_controller_gradient_ratio_range']=checks.pop('tke_controller_gradient_ratio')
 checks['controller_gradient_ratio_samples_finite_nonnegative']=(len(gradient_ratios)==expected_samples and all(math.isfinite(float(x)) and float(x)>=0 for x in gradient_ratios))
 result['original_gate_result']=copy.deepcopy(original)
 result['effective_checks']=checks
 result['diagnostic_only_checks']=removed
 result['health_checks']=checks
 result['health_pass']=all(checks.values())
 result['status']=pass_status if result['health_pass'] else stop_status
 result['architecture_health_policy']=copy.deepcopy(POLICY)
 return result

def install(b210,base,depth):
 if (base,depth)==(210,4):return False
 original=b210.prefix_gate
 def gate(initial,control,candidate,gradient_ratios,update0,controller_audit,hgrad_audits):
  result=original(initial,control,candidate,gradient_ratios,update0,controller_audit,hgrad_audits)
  return apply_policy(result,gradient_ratios,b210.CONTROL_UPDATES,b210.HEALTH_PASS_STATUS,b210.HEALTH_STOP_STATUS)
 b210.prefix_gate=gate
 return True