"""One fixed Sim10000 representation intervention; native objective unchanged."""
import json
from pathlib import Path
import sys

import train_b210_pure_sim_start as pure


def main():
    args = pure.parse_args()
    assert args.teacher_cache is None
    assert args.execute
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import train_b210_np_functional_sim as sim
    from models.physical_units_b210 import install
    common = sim.rehearsal.ensemble_train.common
    old_stats = common.set_model_stats

    def stats(model, normalizer):
        old_stats(model, normalizer)
        install(model, normalizer)

    common.set_model_stats = stats
    pure.SCHEMA = 'physical_units_b210_sim10000_v1'
    pure.STATUS = 'PHYSICAL_UNITS_SIM10000_SEALED'
    result = pure.execute(args)
    result['representation'] = dict(schema='physical_units_b210_v1', added_parameters=0,
                                    scale='one physical UV RMS per history; variance floor 1e-8',
                                    condition='physical 80-value history descriptor through existing FiLM',
                                    source_sha256=pure.sha256(Path(__file__).resolve().parents[1] / 'models/physical_units_b210.py'))
    (args.output_dir/'sim_endpoint.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('decision','checkpoint_path','checkpoint_sha256','representation')},indent=2))


if __name__ == '__main__':
    main()
