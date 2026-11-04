"""The fixed Q4 Real5334 recipe; scoped overrides leave Real4000 unchanged."""
import functools
import train_b210_transfer_real4k as point

point.UPDATES = 5334
dense = point.load_runtime()
dense.small4k.UPDATES = 5334
dense.small4k.PRIMARY_ENDPOINT = 'ema_u5334'
dense.small4k.train_candidate = functools.partial(
    dense.small4k.train_candidate, endpoint_filename='ema999_u5334_primary.pth')
original_fields = point.intervention_fields
point.intervention_fields = lambda args: {
    **original_fields(args),
    'duration_intervention': 'ceil(4000/0.75)=5334; approximately match original-view exposures'}

if __name__ == '__main__':
    args = point.parse_args()
    assert args.arm == 'C' and args.vertical_reflection and args.full_real_trajectories
    if args.execute:
        point.run(args)
    else:
        import json
        print(json.dumps(point.plan(args), indent=2))
