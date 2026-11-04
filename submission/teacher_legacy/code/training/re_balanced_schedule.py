"""Fixed seed4, equal-Re window stream for the registered Real4000 backup."""
from collections import Counter, defaultdict
import random


def balanced_batches(batch_lengths, window_re):
    """Preserve batch lengths; cycle shuffled Re groups and windows locally."""
    generator = random.Random(4)
    pools = defaultdict(list)
    for index, re in enumerate(window_re):
        pools[int(re)].append(index)
    groups = sorted(pools)
    assert len(groups) == 18 and len(window_re) == 10380
    total = sum(batch_lengths)
    assert len(batch_lengths) == 4000 and total == 31988
    assert all(length in (4, 8) for length in batch_lengths)
    positions = {re: len(pool) for re, pool in pools.items()}
    stream = []
    while len(stream) < total:
        order = list(groups)
        generator.shuffle(order)
        for re in order:
            if len(stream) == total:
                break
            if positions[re] == len(pools[re]):
                generator.shuffle(pools[re])
                positions[re] = 0
            stream.append(pools[re][positions[re]])
            positions[re] += 1
    batches, offset = [], 0
    for length in batch_lengths:
        batches.append(stream[offset:offset+length])
        offset += length
    counts = Counter(int(window_re[index]) for index in stream)
    assert max(counts.values())-min(counts.values()) <= 1
    assert set(stream) == set(range(len(window_re)))
    return batches, dict(seed=4, generator='python_random_local',
                         re_draw_counts=dict(sorted(counts.items())),
                         sample_count=len(stream), unique_windows=len(set(stream)))
