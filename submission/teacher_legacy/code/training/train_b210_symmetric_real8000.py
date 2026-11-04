"""Fixed 50% reflection / Real8000 candidate; original Real4000 is unchanged."""
import functools
import train_b210_transfer_real4k as point
import vertical_reflection_dataset as reflection

point.UPDATES = 8000
dense = point.load_runtime()
dense.small4k.UPDATES = 8000
dense.small4k.PRIMARY_ENDPOINT = "ema_u8000"
dense.small4k.train_candidate = functools.partial(
    dense.small4k.train_candidate, endpoint_filename="ema999_u8000_primary.pth")
reflection.VerticalReflectionDataset = functools.partial(
    reflection.VerticalReflectionDataset, probability=.5)
original_fields = point.intervention_fields


def fields(args):
    values = original_fields(args)
    values["vertical_reflection"]["probability"] = .5
    values["duration_intervention"] = "8000 updates; match expected 4000 original-view exposures"
    return values


point.intervention_fields = fields

if __name__ == "__main__":
    args = point.parse_args()
    assert args.arm == "C" and args.vertical_reflection and args.full_real_trajectories
    assert args.execute
    point.run(args)
