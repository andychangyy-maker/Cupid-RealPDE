from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from build_p0_oof_absolute_endpoints import reconstruction_order
from train_p1_direct_absolute_endpoint_head import (
    absolute_endpoints,
    select_fold_indices,
)


def test_absolute_endpoints_use_physical_point_and_normalized_extents() -> None:
    prediction = torch.tensor([[[[[2.0, -1.0, 9.0]]]]])
    lower_extent = torch.tensor([[[[[0.5, -0.25]]]]])
    upper_extent = torch.tensor([[[[[1.5, 0.75]]]]])
    mean = torch.tensor([10.0, 20.0, 0.0])
    std = torch.tensor([2.0, 4.0, 1.0])

    point, lower, upper = absolute_endpoints(
        prediction, lower_extent, upper_extent, mean, std)

    torch.testing.assert_close(point, torch.tensor([[[[[14.0, 16.0]]]]]))
    torch.testing.assert_close(lower, torch.tensor([[[[[13.0, 17.0]]]]]))
    torch.testing.assert_close(upper, torch.tensor([[[[[17.0, 19.0]]]]]))


def test_fold_partition_excludes_fold04_and_cache_order_is_explicit() -> None:
    folds = np.asarray([4, 0, 3, 4, 1, 2], dtype=np.int64)

    train, evaluation = select_fold_indices(folds, held_fold=4)

    assert train == [1, 2, 4, 5]
    assert evaluation == [0, 3]
    assert all(folds[index] != 4 for index in train)
    expanded = np.tile(np.arange(5, dtype=np.int64), 145)
    order = reconstruction_order(expanded.tolist())
    assert np.array_equal(expanded[order], np.sort(expanded))
    assert np.array_equal(np.sort(order), np.arange(725))
