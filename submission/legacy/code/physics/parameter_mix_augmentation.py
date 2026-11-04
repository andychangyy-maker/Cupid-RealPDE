"""Phase-aligned parameter interpolation for structured PDE trajectories.

The released Track-1 Sim data form a regular ``Re x AoA`` grid.  This module
creates small, target-consistent moves between adjacent Reynolds-number
trajectories at fixed AoA.  A short observed-history signature selects a phase
shift before interpolation so that vortex shedding is not averaged at
unrelated phases.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class ParameterMixContract:
    probability: float = 0.50
    lambda_low: float = 0.10
    lambda_high: float = 0.30
    maximum_phase_shift: int = 30
    phase_shift_stride: int = 2
    signature_spatial_stride: int = 4
    seed: int = 1
    rng_offset: int = 1_000_033
    include_aoa_neighbors: bool = False

    def validate(self) -> None:
        finite = (self.probability, self.lambda_low, self.lambda_high)
        if any(not math.isfinite(value) for value in finite):
            raise ValueError("parameter-mix probabilities must be finite")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("parameter-mix probability must lie in [0,1]")
        if not 0.0 <= self.lambda_low <= self.lambda_high <= 0.5:
            raise ValueError("parameter-mix lambda range must lie in [0,0.5]")
        if self.maximum_phase_shift < 0 or self.phase_shift_stride <= 0:
            raise ValueError("parameter-mix phase search is invalid")
        if self.signature_spatial_stride <= 0 or self.rng_offset <= 0:
            raise ValueError("parameter-mix signature/RNG contract is invalid")


class PhaseAlignedParameterMixAugmentor:
    """Compose a closure augmentor with phase-aligned adjacent-Re mixing.

    ``dataset`` must expose the immutable ``trajectories`` and ``windows``
    used by the optimizer stream.  The caller supplies the current natural
    sample indices through :meth:`set_batch_context`; this avoids inferring a
    window start from floating-point contents after another augmentation.
    """

    def __init__(
            self, base_augmentor, dataset, metadata: Sequence[Mapping],
            contract: ParameterMixContract | None = None,
            ) -> None:
        self.base = base_augmentor
        self.dataset = dataset
        self.metadata = [dict(value) for value in metadata]
        self.contract = contract or ParameterMixContract()
        self.contract.validate()
        if not hasattr(dataset, "trajectories") or not hasattr(dataset, "windows"):
            raise TypeError("parameter mix requires trajectories and windows")
        if len(dataset.trajectories) != len(self.metadata):
            raise ValueError("parameter-mix metadata/trajectory count changed")
        self._partners = self._build_partner_table()
        self._aligned_starts = self._build_alignment_table()
        self._batch_indices: torch.Tensor | None = None
        self._iteration: int | None = None
        self.last_audit: dict[str, float] = {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def to(self, device: torch.device) -> "PhaseAlignedParameterMixAugmentor":
        self.base.to(device)
        return self

    def _build_partner_table(self) -> list[tuple[int, ...]]:
        by_aoa: dict[int, list[int]] = {}
        by_re: dict[int, list[int]] = {}
        for index, row in enumerate(self.metadata):
            if "aoa" not in row or "re" not in row:
                raise ValueError("parameter-mix metadata needs AoA and Re")
            by_aoa.setdefault(int(row["aoa"]), []).append(index)
            by_re.setdefault(int(row["re"]), []).append(index)
        result_sets: list[set[int]] = [set() for _ in self.metadata]

        def add_axis_neighbors(indices: list[int], field: str) -> None:
            ordered = sorted(
                indices, key=lambda index: int(self.metadata[index][field]))
            for position, index in enumerate(ordered):
                if position:
                    result_sets[index].add(ordered[position - 1])
                if position + 1 < len(ordered):
                    result_sets[index].add(ordered[position + 1])

        for indices in by_aoa.values():
            add_axis_neighbors(indices, "re")
        if self.contract.include_aoa_neighbors:
            for indices in by_re.values():
                add_axis_neighbors(indices, "aoa")
        if any(not neighbors for neighbors in result_sets):
            raise ValueError("each parameter-mix trajectory needs a neighbor")
        return [tuple(sorted(neighbors)) for neighbors in result_sets]

    def _signature(self, trajectory: torch.Tensor, start: int) -> torch.Tensor:
        # The last ten observed frames retain shedding phase while the spatial
        # decimation keeps the one-time alignment audit inexpensive.
        value = trajectory[
            start + 10:start + 20:2,
            ::self.contract.signature_spatial_stride,
            ::self.contract.signature_spatial_stride,
            :2,
        ].float().reshape(-1)
        value = value - value.mean()
        return value / value.square().mean().sqrt().clamp_min(1.0e-6)

    def _aligned_start(self, trajectory: int, start: int,
                       partner: int) -> int:
        source = self._signature(self.dataset.trajectories[trajectory], start)
        partner_values = self.dataset.trajectories[partner]
        maximum_start = int(partner_values.shape[0]) - 40
        low = max(0, start - self.contract.maximum_phase_shift)
        high = min(maximum_start, start + self.contract.maximum_phase_shift)
        candidates = list(range(low, high + 1, self.contract.phase_shift_stride))
        if start <= maximum_start and start not in candidates:
            candidates.append(start)
            candidates.sort()
        errors = torch.stack([
            (self._signature(partner_values, candidate) - source).square().mean()
            for candidate in candidates
        ])
        return int(candidates[int(errors.argmin())])

    def _build_alignment_table(self) -> dict[tuple[int, int], int]:
        result: dict[tuple[int, int], int] = {}
        for sample_index, window in enumerate(self.dataset.windows):
            trajectory = int(window.trajectory)
            start = int(window.start)
            for partner in self._partners[trajectory]:
                result[(sample_index, partner)] = self._aligned_start(
                    trajectory, start, partner)
        return result

    def set_batch_context(self, sample_indices: torch.Tensor,
                          iteration: int) -> None:
        indices = sample_indices.detach().to(device="cpu", dtype=torch.long)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("parameter-mix sample indices must be a nonempty vector")
        if int(indices.min()) < 0 or int(indices.max()) >= len(self.dataset):
            raise IndexError("parameter-mix sample index lies outside the dataset")
        if iteration < 1:
            raise ValueError("parameter-mix iteration must be positive")
        self._batch_indices = indices.clone()
        self._iteration = int(iteration)

    def _parameter_mix(
            self, history: torch.Tensor, target: torch.Tensor,
            trajectory_ids: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._batch_indices is None or self._iteration is None:
            raise RuntimeError("parameter-mix batch context was not set")
        if len(self._batch_indices) != len(history):
            raise ValueError("parameter-mix context/batch size changed")
        actual_trajectories = trajectory_ids.detach().to(
            device="cpu", dtype=torch.long)
        expected_trajectories = torch.tensor([
            int(self.dataset.windows[int(index)].trajectory)
            for index in self._batch_indices
        ], dtype=torch.long)
        if not torch.equal(actual_trajectories, expected_trajectories):
            raise RuntimeError("parameter-mix trajectory IDs do not match sample indices")

        generator = torch.Generator(device="cpu").manual_seed(
            self.contract.seed + self.contract.rng_offset * self._iteration)
        selected_cpu = torch.rand(len(history), generator=generator) < (
            self.contract.probability)
        if not bool(selected_cpu.any()):
            selected = selected_cpu.to(device=history.device)
            self.last_audit = {
                "parameter_mix_selected_fraction": 0.0,
                "parameter_mix_lambda_mean": 0.0,
                "parameter_mix_abs_phase_shift_mean": 0.0,
                "parameter_mix_abs_re_delta_mean": 0.0,
                "parameter_mix_abs_aoa_delta_mean": 0.0,
                "parameter_mix_target_delta_uv_rms": 0.0,
            }
            return history, target, selected

        sequences = []
        lambdas = []
        shifts = []
        re_deltas = []
        aoa_deltas = []
        for batch_index, sample_value in enumerate(self._batch_indices.tolist()):
            window = self.dataset.windows[sample_value]
            trajectory = int(window.trajectory)
            partners = self._partners[trajectory]
            partner_choice = int(torch.randint(
                len(partners), (), generator=generator))
            partner = partners[partner_choice]
            aligned_start = self._aligned_starts[(sample_value, partner)]
            partner_sequence = self.dataset.trajectories[partner][
                aligned_start:aligned_start + 40]
            sequences.append(partner_sequence)
            value = self.contract.lambda_low + (
                self.contract.lambda_high - self.contract.lambda_low
            ) * float(torch.rand((), generator=generator))
            lambdas.append(value if bool(selected_cpu[batch_index]) else 0.0)
            shifts.append(abs(aligned_start - int(window.start)))
            re_deltas.append(abs(
                int(self.metadata[partner]["re"])
                - int(self.metadata[trajectory]["re"])))
            aoa_deltas.append(abs(
                int(self.metadata[partner]["aoa"])
                - int(self.metadata[trajectory]["aoa"])))

        partner_batch = torch.stack(sequences).to(
            device=history.device, dtype=history.dtype, non_blocking=True)
        sequence = torch.cat((history, target), dim=1)
        weight = sequence.new_tensor(lambdas).view(-1, 1, 1, 1, 1)
        augmented = sequence + weight * (partner_batch - sequence)
        selected = selected_cpu.to(device=history.device)
        target_delta = augmented[:, history.shape[1]:, ..., :2] - target[..., :2]
        chosen = selected_cpu.nonzero(as_tuple=False).flatten().tolist()
        self.last_audit = {
            "parameter_mix_selected_fraction": float(selected_cpu.float().mean()),
            "parameter_mix_lambda_mean": float(
                sum(lambdas[index] for index in chosen) / len(chosen)),
            "parameter_mix_abs_phase_shift_mean": float(
                sum(shifts[index] for index in chosen) / len(chosen)),
            "parameter_mix_abs_re_delta_mean": float(
                sum(re_deltas[index] for index in chosen) / len(chosen)),
            "parameter_mix_abs_aoa_delta_mean": float(
                sum(aoa_deltas[index] for index in chosen) / len(chosen)),
            "parameter_mix_target_delta_uv_rms": float(
                target_delta[selected].float().square().mean().sqrt()),
        }
        return (
            augmented[:, :history.shape[1]],
            augmented[:, history.shape[1]:], selected)

    def augment(self, history: torch.Tensor, target: torch.Tensor,
                trajectory_ids: torch.Tensor, **kwargs):
        mixed_history, mixed_target, mix_selected = self._parameter_mix(
            history, target, trajectory_ids)
        closed_history, closed_target, closure_selected = self.base.augment(
            mixed_history, mixed_target, trajectory_ids, **kwargs)
        return closed_history, closed_target, mix_selected | closure_selected

    def augment_future(self, *args, **kwargs):
        return self.base.augment_future(*args, **kwargs)
