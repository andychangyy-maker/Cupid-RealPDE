"""Fixed coordinate-only H reflection; velocity components retain their signs."""
import train_b210_transfer_real4k as point
import vertical_reflection_dataset as reflection


def reflect_coordinates(value):
    return value.flip(-3)


reflection.reflect_uvp = reflect_coordinates
original_fields = point.intervention_fields


def fields(args):
    values = original_fields(args)
    values["vertical_reflection"]["velocity_sign"] = "unchanged"
    values["vertical_reflection"]["interpretation"] = "coordinate_augmentation_not_physical_symmetry"
    return values


point.intervention_fields = fields

if __name__ == "__main__":
    args = point.parse_args()
    assert args.arm == "C" and args.vertical_reflection and args.full_real_trajectories and args.execute
    point.run(args)
