"""Sim augmentation driven by unresolved closure innovations from Real PIV."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from physics.vortex_augmentation import _advect_perturbation


class ClosureResidualAugmentor:
    """Inject matched Real closure innovations into released Sim windows."""

    def __init__(self, records: list[dict], sim_metadata: list[dict],
                 candidate_count: int = 3,
                 source_contract: Mapping | None = None):
        if not records or not sim_metadata:
            raise ValueError("closure records and Sim metadata are required")
        if candidate_count <= 0:
            raise ValueError("candidate count must be positive")
        self.records = records
        self.sim_metadata = sim_metadata
        self.source_contract = dict(source_contract or {})
        self.candidates = []
        for sim in sim_metadata:
            same_aoa = [
                index for index, record in enumerate(records)
                if int(record["aoa"]) == int(sim["aoa"])]
            if not same_aoa:
                raise ValueError(f"no closure record for AoA={sim['aoa']}")
            same_aoa.sort(key=lambda index: abs(
                float(records[index]["re"]) - float(sim["re"])))
            self.candidates.append(same_aoa[:candidate_count])

    @classmethod
    def load(cls, path: Path, sim_metadata: list[dict],
             candidate_count: int = 3) -> "ClosureResidualAugmentor":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        supported = {
            "realpde_real_closure_bank_v1",
            "realpde_real_closure_innovation_bank_v1",
            "realpde_clean_real_closure_bank_v2",
        }
        if payload.get("format") not in supported:
            raise ValueError("unsupported closure bank format")
        return cls(
            payload["records"], sim_metadata, candidate_count,
            payload.get("source_contract"))

    def to(self, device: torch.device) -> "ClosureResidualAugmentor":
        """Preload the immutable bank on the training device."""
        for record in self.records:
            key = "innovation" if "innovation" in record else "residual"
            record[key] = record[key].to(device=device, non_blocking=True)
        return self

    def _sample_innovations(
            self, trajectory_ids: torch.Tensor, selected: torch.Tensor,
            steps: int, reference: torch.Tensor, decay: float,
            ) -> torch.Tensor:
        batch, _time, height, width, _channels = reference.shape
        innovations = reference.new_zeros((batch, steps, height, width, 2))
        for batch_index in torch.nonzero(
                selected, as_tuple=False).flatten().tolist():
            trajectory = int(trajectory_ids[batch_index])
            candidates = self.candidates[trajectory]
            record_index = candidates[int(torch.randint(
                len(candidates), ()).item())]
            record = self.records[record_index]
            if "innovation" in record:
                source = record["innovation"]
                required = steps
                if source.shape[0] < required:
                    raise ValueError(
                        "innovation record is shorter than training window")
                start = int(torch.randint(
                    source.shape[0] - required + 1, ()).item())
                clip = source[start:start + required]
            else:
                source = record["residual"]
                required = steps + 1
                if source.shape[0] < required:
                    raise ValueError(
                        "closure record is shorter than training window")
                start = int(torch.randint(
                    source.shape[0] - required + 1, ()).item())
                residual = source[start:start + required]
                clip = residual[1:] - decay * residual[:-1]
            innovations[batch_index] = clip.to(
                device=reference.device, dtype=reference.dtype)
        return innovations

    def augment(
            self, history: torch.Tensor, target: torch.Tensor,
            trajectory_ids: torch.Tensor, *, probability: float,
            amplitude: float, closure_decay: float = 0.85,
            perturbation_decay: float = 0.5,
            advection_cells: float = 0.5,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a continuous nearby Sim trajectory from Real innovations.

        ``amplitude`` is the final perturbation RMS divided by the released
        Sim u/v RMS.  Real residuals were already divided by their freestream,
        so they share the nondimensional velocity units of released Sim.
        """
        if history.ndim != 5 or target.ndim != 5:
            raise ValueError("history and target must be rank-five")
        if history.shape[0] != target.shape[0] or history.shape[2:] != target.shape[2:]:
            raise ValueError("history/target batch and field shapes must match")
        if trajectory_ids.numel() != history.shape[0]:
            raise ValueError("one trajectory id is required per sample")
        if not 0.0 <= probability <= 1.0 or amplitude < 0.0:
            raise ValueError("probability/amplitude are invalid")
        if not 0.0 <= closure_decay <= 1.0:
            raise ValueError("closure decay must lie in [0,1]")
        if not 0.0 <= perturbation_decay <= 1.0 or advection_cells < 0.0:
            raise ValueError("perturbation dynamics are invalid")
        if probability == 0.0 or amplitude == 0.0:
            selected = torch.zeros(
                history.shape[0], device=history.device, dtype=torch.bool)
            return history, target, selected

        sequence = torch.cat((history, target), dim=1)
        batch, total_steps, _height, _width, _channels = sequence.shape
        selected = torch.rand(batch, device=sequence.device) < probability
        if not selected.any():
            return history, target, selected
        innovations = self._sample_innovations(
            trajectory_ids, selected, total_steps - 1,
            sequence, closure_decay)
        fluid = sequence[..., :2].abs().sum(dim=-1).ne(0.0)
        perturbation = sequence.new_zeros((
            batch, 2, sequence.shape[2], sequence.shape[3]))
        perturbations = [perturbation]
        for index in range(total_steps - 1):
            mask = fluid[:, index].unsqueeze(1).to(sequence.dtype)
            base_velocity = sequence[:, index, ..., :2].permute(0, 3, 1, 2)
            transported = _advect_perturbation(
                perturbation, base_velocity, mask, advection_cells)
            forcing = innovations[:, index].permute(0, 3, 1, 2)
            next_mask = fluid[:, index + 1].unsqueeze(1).to(sequence.dtype)
            perturbation = (
                perturbation_decay * transported + forcing) * next_mask
            perturbations.append(perturbation)
        perturbation_sequence = torch.stack(
            perturbations, dim=1).permute(0, 1, 3, 4, 2)

        fluid_value = fluid.unsqueeze(-1).to(sequence.dtype)
        velocity_rms = torch.sqrt(
            (sequence[..., :2].square() * fluid_value).sum(
                dim=(1, 2, 3, 4))
            / fluid_value.sum(dim=(1, 2, 3, 4)).clamp_min(1).mul(2))
        perturbation_rms = torch.sqrt(
            (perturbation_sequence.square() * fluid_value).sum(
                dim=(1, 2, 3, 4))
            / fluid_value.sum(dim=(1, 2, 3, 4)).clamp_min(1).mul(2)
        ).clamp_min(1.0e-8)
        scale = amplitude * velocity_rms / perturbation_rms
        scale = scale * selected.to(scale.dtype)
        augmented = sequence.clone()
        augmented[..., :2].add_(
            perturbation_sequence * scale[:, None, None, None, None]
            * fluid_value)
        return (
            augmented[:, :history.shape[1]],
            augmented[:, history.shape[1]:], selected)

    def augment_future(
            self, history: torch.Tensor, target: torch.Tensor,
            trajectory_ids: torch.Tensor, *, probability: float,
            amplitude: float, closure_decay: float = 0.85,
            perturbation_decay: float = 0.5,
            advection_cells: float = 0.5,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Branch only the unobserved future while preserving history exactly.

        The perturbation state is zero at the observation/forecast boundary.
        Real closure innovations then drive the 20 future steps.  Unlike
        :meth:`augment`, this method never edits an observed frame, so several
        calls can represent different futures conditional on the same input.
        """
        if history.ndim != 5 or target.ndim != 5:
            raise ValueError("history and target must be rank-five")
        if history.shape[0] != target.shape[0] or history.shape[2:] != target.shape[2:]:
            raise ValueError("history/target batch and field shapes must match")
        if history.shape[1] < 1:
            raise ValueError("at least one observed frame is required")
        if trajectory_ids.numel() != history.shape[0]:
            raise ValueError("one trajectory id is required per sample")
        if not 0.0 <= probability <= 1.0 or amplitude < 0.0:
            raise ValueError("probability/amplitude are invalid")
        if not 0.0 <= closure_decay <= 1.0:
            raise ValueError("closure decay must lie in [0,1]")
        if not 0.0 <= perturbation_decay <= 1.0 or advection_cells < 0.0:
            raise ValueError("perturbation dynamics are invalid")
        if probability == 0.0 or amplitude == 0.0:
            selected = torch.zeros(
                history.shape[0], device=history.device, dtype=torch.bool)
            return target, selected

        # The boundary frame supplies the initial velocity/mask, but its
        # perturbation is identically zero and it is never returned or edited.
        sequence = torch.cat((history[:, -1:], target), dim=1)
        batch, _total_steps, _height, _width, _channels = sequence.shape
        selected = torch.rand(batch, device=sequence.device) < probability
        if not selected.any():
            return target, selected
        innovations = self._sample_innovations(
            trajectory_ids, selected, target.shape[1],
            sequence, closure_decay)
        fluid = sequence[..., :2].abs().sum(dim=-1).ne(0.0)
        perturbation = sequence.new_zeros((
            batch, 2, sequence.shape[2], sequence.shape[3]))
        future_perturbations = []
        for index in range(target.shape[1]):
            mask = fluid[:, index].unsqueeze(1).to(sequence.dtype)
            base_velocity = sequence[:, index, ..., :2].permute(0, 3, 1, 2)
            transported = _advect_perturbation(
                perturbation, base_velocity, mask, advection_cells)
            forcing = innovations[:, index].permute(0, 3, 1, 2)
            next_mask = fluid[:, index + 1].unsqueeze(1).to(sequence.dtype)
            perturbation = (
                perturbation_decay * transported + forcing) * next_mask
            future_perturbations.append(perturbation)
        perturbation_sequence = torch.stack(
            future_perturbations, dim=1).permute(0, 1, 3, 4, 2)

        target_fluid = fluid[:, 1:].unsqueeze(-1).to(sequence.dtype)
        velocity_rms = torch.sqrt(
            (target[..., :2].square() * target_fluid).sum(
                dim=(1, 2, 3, 4))
            / target_fluid.sum(dim=(1, 2, 3, 4)).clamp_min(1).mul(2))
        perturbation_rms = torch.sqrt(
            (perturbation_sequence.square() * target_fluid).sum(
                dim=(1, 2, 3, 4))
            / target_fluid.sum(dim=(1, 2, 3, 4)).clamp_min(1).mul(2)
        ).clamp_min(1.0e-8)
        scale = amplitude * velocity_rms / perturbation_rms
        scale = scale * selected.to(scale.dtype)
        branched = target.clone()
        branched[..., :2].add_(
            perturbation_sequence * scale[:, None, None, None, None]
            * target_fluid)
        return branched, selected
