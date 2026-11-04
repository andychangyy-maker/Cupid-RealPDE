"""Optional Real-only read-only audit synchronization repair. Original CLI unchanged."""
import argparse
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import run_architecture_real as original

PROBE = original.NEW/'architecture_preparation/training_runtime_probe_v2'
GATE_SHA = '62e2cd85c27346b5d3d776aed6bccc2b72c939ad81dd72700fdf72015e6c5e7a'
FOREACH_SHA = 'db7a72a4c513bcabea185f8b357afeefca59b839d614845b2c9b10da52591f05'
AUDIT_SHA = '1b335a7088bd777452b21db445b67bfe4a5ce213da0b00664291bd329b4ac82c'
ORIGINAL_ENTRY_SHA = '61e7991f6e0cd8cd4a4fedf97d6fcadf2e61fe64e02a7cafe7bebce15e36027f'
D3_ENTRY_SHA = 'a0f4d8d5d8699a1387df206d43e3a0ed66a76dc889e5722cff6daacd8cf4cda3'

def function_ast(function):
    return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(function))).body[0], include_attributes=False)

def load_verified_audit():
    import torch
    for name, expected in [('blackwell_foreach_abba.json', GATE_SHA),
                           ('foreach_gradient_audit.py', FOREACH_SHA),
                           ('original_gradient_audit.py', AUDIT_SHA)]:
        if original.sha(PROBE/name) != expected:
            raise ValueError('Verified audit evidence/source changed: '+name)
    gate = json.loads((PROBE/'blackwell_foreach_abba.json').read_text())
    if gate['status']!='PASS' or torch.__version__!=gate['torch_version']:
        raise ValueError('Audit requires the tested Torch environment and PASS gate')
    if any(gate[key]['status']!='PASS' for key in ['cpu_semantics','cpu_norm_semantics','gpu_norm_semantics']):
        raise ValueError('Missing semantic gate')
    sys.path.insert(0,str(PROBE))
    module=importlib.import_module('foreach_gradient_audit')
    if Path(module.__file__).resolve()!= (PROBE/'foreach_gradient_audit.py').resolve():
        raise ValueError('Audit module collision')
    return module, gate

def configure(depth):
    """Called once in a fresh process, before original.main creates any training state."""
    if getattr(original, '_foreach_entry_configured', False):
        raise RuntimeError('Use a fresh process for each Real invocation')
    if original.sha(original.__file__)!=ORIGINAL_ENTRY_SHA:
        raise ValueError('Original Real adapter identity changed')
    d3_path=None
    if depth==3:
        d3_path=Path(__file__).with_name('run_architecture_real_d3_fix1.py')
        if original.sha(d3_path)!=D3_ENTRY_SHA:raise ValueError('D3 repair identity changed')
        importlib.import_module('run_architecture_real_d3_fix1')
    previous_install=original.install_real
    audit,gate=load_verified_audit()
    def install_real(base, actual_depth, runtime_root):
        if actual_depth!=depth:raise ValueError('CLI/install depth mismatch')
        result=previous_install(base,actual_depth,runtime_root)
        b,transfer,dense,uq,native,info=result
        owner=dense.small4k.npt1
        if owner is not dense.ft12.npt1:raise ValueError('Unexpected runtime audit module aliases')
        loop=dense.small4k.train_candidate
        if loop.__globals__.get('npt1') is not owner:
            raise ValueError('Actual point loop does not resolve the expected audit module')
        old=owner._gradient_audit
        if function_ast(old)!=function_ast(audit._gradient_audit):
            raise ValueError('Actual runtime audit differs from verified original function')
        old_source_file=Path(inspect.getsourcefile(old))
        owner._gradient_audit=audit.foreach_gradient_audit
        assert loop.__globals__['npt1']._gradient_audit is audit.foreach_gradient_audit
        assert dense.ft12.npt1._gradient_audit is audit.foreach_gradient_audit
        info['read_only_gradient_audit_engineering']={
            'id':'foreach_infinity_norm_audit_v2', 'scientific_changes':[],
            'scope':'Real point loop only; Q has no call to this audit; Sim untouched',
            'wrapper_sha256':original.sha(__file__),
            'original_real_entry_sha256':ORIGINAL_ENTRY_SHA,
            'd3_repair_sha256':D3_ENTRY_SHA if d3_path else None,
            'original_audit_source_file':str(old_source_file),
            'original_audit_source_file_sha256':original.sha(old_source_file),
            'original_function_ast_sha256':hashlib.sha256(function_ast(old).encode()).hexdigest(),
            'original_audit_extract_sha256':AUDIT_SHA,'foreach_source_sha256':FOREACH_SHA,
            'blackwell_gate':str(PROBE/'blackwell_foreach_abba.json'),
            'blackwell_gate_sha256':GATE_SHA,'torch_version':gate['torch_version'],
            'actual_loop_binding_verified':True,'loss_optimizer_sampling_unchanged':True}
        return result
    original.install_real=install_real
    original._foreach_entry_configured=True
    return original

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--depth',type=int,required=True)
    args,_=parser.parse_known_args()
    configure(args.depth).main()

if __name__=='__main__':main()
