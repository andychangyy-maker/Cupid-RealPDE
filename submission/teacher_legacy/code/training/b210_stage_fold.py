"""Metadata-only split for Real/UQ-stage LOAO; ancestors remain shared."""
AOAS = (0, 5, 10, 15, 20)


def select_rows(rows, held_aoa, *, evaluation=False, inner_held_aoa=None):
    if inner_held_aoa is not None and (evaluation or held_aoa is None
                                      or inner_held_aoa not in AOAS or inner_held_aoa == held_aoa):
        raise ValueError('nested training requires two distinct excluded AoAs')
    if held_aoa is None:
        return rows
    if held_aoa not in AOAS or {int(r['aoa']) for r in rows} != set(AOAS):
        raise ValueError('stage fold requires all five AoAs in the original manifest')
    selected = [r for r in rows if (int(r['aoa']) == held_aoa) == evaluation]
    if inner_held_aoa is not None:
        selected = [r for r in selected if int(r['aoa']) != inner_held_aoa]
    if not selected or any(r['file'] == '7575_0.h5' for r in selected):
        raise ValueError('invalid clean81 stage fold')
    return selected
