from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Mapping
import torch
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Subset,
    SubsetRandomSampler,
    TensorDataset,
)
from .config import ALL_METHODS, ExperimentConfig
from .evidence import EvidenceOutputs, supervised_lesion_loss
from .losses import compute_class_weights, compute_method_loss
from .metrics import (
    binary_metrics,
    clinical_reliability_metrics,
    is_better_checkpoint,
    macro_source_score,
    select_source_threshold,
)
from .ordinal import (
    OrdinalBridge,
    apply_level_map,
    build_shuffled_level_maps,
    estimate_ordinal_bridge,
    head_coupling_loss,
    supervised_ordinal_loss,
)

ORDINAL_HEAD_METHODS = frozenset(
    {
        "fedavg_ord",
        "fedevi_dg",
        "fedevi_mtl",
        "fedevi_shuffle",
        "fedevi_mask_shift",
        "fedevi_no_taskagg",
        "fedevi_monotonic",
    }
)
CLINICAL_HEAD_METHODS = ORDINAL_HEAD_METHODS
EVIDENCE_METHODS = frozenset(
    {
        "fedevi_dg",
        "fedevi_mtl",
        "fedevi_seg_only",
        "fedevi_shuffle",
        "fedevi_mask_shift",
        "fedevi_no_taskagg",
        "fedevi_monotonic",
    }
)


@dataclass
class TrainingResult:
    best_round: int
    best_state_dict: dict[str, torch.Tensor]
    source_val_metrics: dict[str, dict]
    source_threshold: float
    target_metrics: dict
    target_evaluation_count: int
    selected_checkpoint_kind: str
    selection_domains: tuple[str, ...]
    target_domain: str
    training_history: list[dict]
    method_metadata: dict[str, object] | None = None


@dataclass
class _ClientPayload:
    state_dict: dict[str, torch.Tensor]
    num_samples: int
    mean_loss: float
    ordinal_supervision_samples: int = 0
    mean_ordinal_loss: float = 0.0
    lesion_supervision_samples: int = 0
    mean_lesion_loss: float = 0.0


@dataclass
class _Checkpoint:
    round_index: int
    key: tuple[float, float, int]
    state_dict: dict[str, torch.Tensor]
    source_metrics: dict[str, dict]


def _clone_state(
    state_dict: Mapping[str, torch.Tensor], *, cpu: bool = False
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to("cpu").clone() if cpu else value.detach().clone()
        for name, value in state_dict.items()
    }


def _weighted_average(payloads: list[_ClientPayload]) -> dict[str, torch.Tensor]:
    if not payloads:
        raise ValueError("federated averaging requires at least one client")
    total_samples = sum((payload.num_samples for payload in payloads))
    if total_samples <= 0:
        raise ValueError("federated averaging requires positive sample counts")
    reference_keys = tuple(payloads[0].state_dict)
    for payload in payloads:
        if payload.num_samples <= 0:
            raise ValueError("client sample counts must be positive")
        if tuple(payload.state_dict) != reference_keys:
            raise ValueError("client state dictionaries must have identical keys")
    averaged: dict[str, torch.Tensor] = {}
    for name in reference_keys:
        values = [payload.state_dict[name] for payload in payloads]
        reference = values[0]
        if any(
            (
                value.shape != reference.shape or value.dtype != reference.dtype
                for value in values
            )
        ):
            raise ValueError(f"client state mismatch for {name}")
        if reference.is_floating_point() or reference.is_complex():
            accumulator = torch.zeros_like(reference)
            for payload, value in zip(payloads, values, strict=True):
                accumulator.add_(value, alpha=payload.num_samples / total_samples)
            averaged[name] = accumulator
        else:
            averaged[name] = reference.clone()
    return averaged


def _task_aware_weighted_average(
    payloads: list[_ClientPayload],
    *,
    aggregate_clinical: bool = True,
    aggregate_lesion: bool = True,
) -> dict[str, torch.Tensor]:
    """Average clinical heads only across clinically annotated clients.

    Shared parameters use all clients; auxiliary heads use valid annotation counts.
    """
    averaged = _weighted_average(payloads)
    supervised: list[_ClientPayload] = []
    clinical_total = 0
    weight_of = lambda payload: payload.ordinal_supervision_samples
    if aggregate_clinical:
        supervised = [
            payload for payload in payloads if payload.ordinal_supervision_samples > 0
        ]
        if not supervised:
            raise ValueError("task-aware aggregation requires an annotated client")
        clinical_total = sum(
            (payload.ordinal_supervision_samples for payload in supervised)
        )
    for name, reference in tuple(averaged.items()):
        clinical_parameter = any((token in name for token in ("ordinal_head.",)))
        lesion_parameter = "lesion_head." in name
        if clinical_parameter and (not aggregate_clinical):
            continue
        if lesion_parameter and (not aggregate_lesion):
            continue
        if not clinical_parameter and (not lesion_parameter):
            continue
        if lesion_parameter:
            selected = [
                payload
                for payload in payloads
                if payload.lesion_supervision_samples > 0
            ]
            if not selected:
                raise ValueError(
                    "task-aware aggregation requires a lesion-annotated client"
                )
            total = sum((payload.lesion_supervision_samples for payload in selected))
            parameter_weight = lambda payload: payload.lesion_supervision_samples
        else:
            selected = supervised
            parameter_weight = weight_of
            total = clinical_total
        if reference.is_floating_point() or reference.is_complex():
            value = torch.zeros_like(reference)
            for payload in selected:
                value.add_(
                    payload.state_dict[name], alpha=parameter_weight(payload) / total
                )
            averaged[name] = value
        else:
            averaged[name] = selected[0].state_dict[name].clone()
    return averaged


def _model_logits(
    model: nn.Module, raw_image: torch.Tensor, content_mask: torch.Tensor
) -> torch.Tensor:
    if getattr(model, "requires_content_mask", False):
        return model(raw_image, content_mask=content_mask)
    return model(raw_image)


def _model_outputs(
    model: nn.Module, raw_image: torch.Tensor, content_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    forward = getattr(model, "forward_with_ordinal", None)
    if forward is None:
        raise TypeError("ordinal methods require a model with forward_with_ordinal")
    binary, ordinal = forward(raw_image, content_mask=content_mask)
    if binary.ndim != 2 or binary.shape[1] != 2:
        raise ValueError("ordinal model binary head must return shape (N, 2)")
    ordinal = torch.as_tensor(ordinal).flatten()
    if ordinal.shape != binary.shape[:1]:
        raise ValueError("ordinal head must return one risk logit per sample")
    return (binary, ordinal)


def _model_evidence_outputs(
    model: nn.Module, raw_image: torch.Tensor, content_mask: torch.Tensor
) -> EvidenceOutputs:
    forward = getattr(model, "forward_with_evidence", None)
    if forward is None:
        raise TypeError("evidence methods require a model with forward_with_evidence")
    outputs = forward(raw_image, content_mask=content_mask)
    if not isinstance(outputs, EvidenceOutputs):
        raise TypeError("forward_with_evidence must return EvidenceOutputs")
    if outputs.binary_logits.ndim != 2 or outputs.binary_logits.shape[1] != 2:
        raise ValueError("evidence binary head must return shape (N, 2)")
    if outputs.ordinal_logits.shape != outputs.binary_logits.shape[:1]:
        raise ValueError("evidence ordinal head must return one value per sample")
    if outputs.lesion_logits.shape[:2] != (outputs.binary_logits.shape[0], 1):
        raise ValueError("evidence lesion head must return shape (N, 1, H, W)")
    return outputs


class FederatedTrainer:

    def __init__(
        self,
        global_model: nn.Module,
        method: str,
        source_loaders: Mapping[str, Mapping[str, DataLoader]],
        target_loader: DataLoader,
        config: ExperimentConfig,
        signatures: Mapping[str, object],
        device: torch.device,
    ) -> None:
        if method not in ALL_METHODS:
            raise ValueError(f"unknown method: {method}")
        if method != config.method:
            raise ValueError("method must match config.method")
        if config.local_epochs != 1:
            raise ValueError("federated training requires exactly one local epoch")
        if config.rounds <= 0:
            raise ValueError("rounds must be positive")
        if config.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if config.target_domain in signatures:
            raise ValueError("target signature must not be present in the bank")
        expected_sources = set(config.domains) - {config.target_domain}
        if set(source_loaders) != expected_sources:
            raise ValueError(
                "source loader domains must exactly match config domains excluding target"
            )
        if len(expected_sources) < 2:
            raise ValueError("at least two source domains are required")
        self.source_domains = tuple(sorted(source_loaders))
        for domain in self.source_domains:
            if (
                "train" not in source_loaders[domain]
                or "val" not in source_loaders[domain]
            ):
                raise ValueError(f"source {domain} requires train and val loaders")
        self._validate_loader_isolation(source_loaders, target_loader)
        for domain in self.source_domains:
            self._validate_dataset_domain(
                source_loaders[domain]["train"].dataset, domain
            )
            self._validate_dataset_domain(source_loaders[domain]["val"].dataset, domain)
        self._validate_dataset_domain(target_loader.dataset, config.target_domain)
        self.method = method
        self.source_loaders = source_loaders
        self._target_loader = target_loader
        self.config = config
        self.signatures = dict(signatures)
        self.device = torch.device(device)
        self.global_model = global_model.to(self.device)
        self.initial_state_dict = _clone_state(self.global_model.state_dict(), cpu=True)
        self._target_evaluation_count = 0
        self._target_ready = False
        self._frozen_source_threshold: float | None = None
        self._best_source_metrics: dict[str, dict] | None = None
        self.ordinal_domains: tuple[str, ...] = ()
        self.ordinal_bridge: OrdinalBridge | None = None
        self.ordinal_level_maps: dict[str, dict[float, float]] = {}
        if method in CLINICAL_HEAD_METHODS:
            required_forward = (
                "forward_with_evidence"
                if method in EVIDENCE_METHODS
                else "forward_with_ordinal"
            )
            if not hasattr(self.global_model, required_forward):
                raise TypeError(f"{method} requires a model with {required_forward}")
            requested = config.birads_domains
            if requested is not None:
                invalid = set(requested) - set(self.source_domains)
                if invalid:
                    raise ValueError(
                        "birads_domains must be source clients only; invalid: "
                        + ", ".join(sorted(invalid))
                    )
            records = self._ordinal_records(
                requested_domains=set(requested) if requested is not None else None
            )
            if method in {"fedevi_shuffle"}:
                self.ordinal_level_maps = build_shuffled_level_maps(
                    records, seed=config.ordinal_shuffle_seed
                )
                records = {
                    domain: (
                        labels,
                        apply_level_map(levels, self.ordinal_level_maps[domain]),
                    )
                    for domain, (labels, levels) in records.items()
                }
            self.ordinal_bridge = estimate_ordinal_bridge(records)
            self.ordinal_domains = tuple(sorted(self.ordinal_bridge.scales))

    @property
    def target_evaluation_count(self) -> int:
        return self._target_evaluation_count

    @property
    def source_threshold(self) -> float | None:
        return self._frozen_source_threshold

    @staticmethod
    def _dataset_object_ids(dataset: object) -> set[int]:
        identities = {id(dataset)}
        if isinstance(dataset, Subset):
            identities.update(FederatedTrainer._dataset_object_ids(dataset.dataset))
        return identities

    @staticmethod
    def _validate_loader_isolation(
        source_loaders: Mapping[str, Mapping[str, DataLoader]],
        target_loader: DataLoader,
    ) -> None:
        source_loader_ids: set[int] = set()
        source_dataset_ids: set[int] = set()
        for loaders in source_loaders.values():
            for split in ("train", "val"):
                loader = loaders[split]
                loader_id = id(loader)
                dataset_ids = FederatedTrainer._dataset_object_ids(loader.dataset)
                if loader_id in source_loader_ids or dataset_ids & source_dataset_ids:
                    raise ValueError("source loader alias is not allowed")
                source_loader_ids.add(loader_id)
                source_dataset_ids.update(dataset_ids)
        if (
            id(target_loader) in source_loader_ids
            or FederatedTrainer._dataset_object_ids(target_loader.dataset)
            & source_dataset_ids
        ):
            raise ValueError("target loader or dataset alias is not allowed")

    @classmethod
    def _dataset_domains(
        cls, dataset: object, indices: list[int] | None = None
    ) -> set[str] | None:
        if indices is None:
            indices = list(range(len(dataset)))
        if isinstance(dataset, Subset):
            mapped = [int(dataset.indices[index]) for index in indices]
            return cls._dataset_domains(dataset.dataset, mapped)
        frame = getattr(dataset, "frame", None)
        if frame is not None and "dataset" in frame:
            return {str(value) for value in frame.iloc[indices]["dataset"].tolist()}
        samples = getattr(dataset, "samples", None)
        if samples is not None and len(samples) > 0:
            selected = [samples[index] for index in indices]
            if all(
                (
                    isinstance(sample, Mapping) and "dataset" in sample
                    for sample in selected
                )
            ):
                return {str(sample["dataset"]) for sample in selected}
        dataset_field = getattr(dataset, "dataset", None)
        if isinstance(dataset_field, str):
            return {dataset_field}
        return None

    @classmethod
    def _validate_dataset_domain(cls, dataset: object, expected: str) -> None:
        domains = cls._dataset_domains(dataset)
        if domains is not None and domains != {expected}:
            observed = ", ".join(sorted(domains)) or "<empty>"
            raise ValueError(
                f"dataset domain {observed} does not match mapping key {expected}"
            )

    @staticmethod
    def _validate_batch_domain(batch: object, expected: str) -> None:
        if not isinstance(batch, Mapping) or "dataset" not in batch:
            return
        values = batch["dataset"]
        if isinstance(values, str):
            domains = {values}
        else:
            domains = {str(value) for value in values}
        if domains != {expected}:
            observed = ", ".join(sorted(domains)) or "<empty>"
            raise ValueError(
                f"batch dataset domain {observed} does not match {expected}"
            )

    def train(self) -> TrainingResult:
        if self._target_ready:
            raise RuntimeError("training already produced a best checkpoint")
        best: _Checkpoint | None = None
        history: list[dict] = []
        for round_index in range(1, self.config.rounds + 1):
            global_state = _clone_state(self.global_model.state_dict())
            payloads = [
                self.train_source(domain, global_state, round_index, client_index)
                for client_index, domain in enumerate(self.source_domains)
            ]
            task_aware = (
                self.method in CLINICAL_HEAD_METHODS or self.method == "fedevi_seg_only"
            ) and self.method != "fedevi_no_taskagg"
            averaged = (
                _task_aware_weighted_average(
                    payloads,
                    aggregate_clinical=self.method in CLINICAL_HEAD_METHODS,
                    aggregate_lesion=self.method in EVIDENCE_METHODS,
                )
                if task_aware
                else _weighted_average(payloads)
            )
            self.global_model.load_state_dict(averaged)
            history_entry: dict[str, object] = {
                "round": round_index,
                "client_train_loss": {
                    domain: payload.mean_loss
                    for domain, payload in zip(
                        self.source_domains, payloads, strict=True
                    )
                },
                "client_ordinal_supervision_samples": {
                    domain: payload.ordinal_supervision_samples
                    for domain, payload in zip(
                        self.source_domains, payloads, strict=True
                    )
                },
                "client_ordinal_loss": {
                    domain: payload.mean_ordinal_loss
                    for domain, payload in zip(
                        self.source_domains, payloads, strict=True
                    )
                },
                "client_lesion_supervision_samples": {
                    domain: payload.lesion_supervision_samples
                    for domain, payload in zip(
                        self.source_domains, payloads, strict=True
                    )
                },
                "client_lesion_loss": {
                    domain: payload.mean_lesion_loss
                    for domain, payload in zip(
                        self.source_domains, payloads, strict=True
                    )
                },
                "source_validation": None,
                "macro_source_auc": None,
                "macro_source_loss": None,
                "selected_checkpoint": False,
            }
            if (
                round_index % self.config.eval_interval == 0
                or round_index == self.config.rounds
            ):
                source_metrics = self.evaluate_all_source_val()
                score = macro_source_score(source_metrics)
                history_entry.update(
                    {
                        "source_validation": {
                            domain: {
                                metric: float(source_metrics[domain][metric])
                                for metric in (
                                    "auc",
                                    "balanced_accuracy",
                                    "sensitivity",
                                    "specificity",
                                    "macro_f1",
                                    "loss",
                                )
                            }
                            for domain in self.source_domains
                        },
                        "macro_source_auc": score.auc,
                        "macro_source_loss": score.loss,
                    }
                )
                key = (score.auc, score.loss, round_index)
                if best is None or is_better_checkpoint(key, best.key):
                    best = _Checkpoint(
                        round_index=round_index,
                        key=key,
                        state_dict=_clone_state(
                            self.global_model.state_dict(), cpu=True
                        ),
                        source_metrics=source_metrics,
                    )
            history.append(history_entry)
        if best is None:
            raise RuntimeError("training produced no evaluated checkpoint")
        history[best.round_index - 1]["selected_checkpoint"] = True
        self.global_model.load_state_dict(best.state_dict)
        self._best_source_metrics = copy.deepcopy(best.source_metrics)
        threshold = self.derive_source_threshold()
        self._frozen_source_threshold = threshold
        self._target_ready = True
        target_metrics = self.evaluate_target_once(threshold)
        return TrainingResult(
            best_round=best.round_index,
            best_state_dict=_clone_state(best.state_dict, cpu=True),
            source_val_metrics=copy.deepcopy(best.source_metrics),
            source_threshold=threshold,
            target_metrics=target_metrics,
            target_evaluation_count=self.target_evaluation_count,
            selected_checkpoint_kind="global",
            selection_domains=self.source_domains,
            target_domain=self.config.target_domain,
            training_history=history,
            method_metadata=(
                {
                    "ordinal_bridge": (
                        self.ordinal_bridge.to_dict()
                        if self.ordinal_bridge is not None
                        else None
                    ),
                    "ordinal_domains": list(self.ordinal_domains),
                    "clinical_head_aggregation": (
                        "annotated_clients_only"
                        if self.method in CLINICAL_HEAD_METHODS
                        else None
                    ),
                    "ordinal_head_aggregation": (
                        "annotated_clients_only"
                        if self.method in ORDINAL_HEAD_METHODS
                        else None
                    ),
                    "clinical_supervision": (
                        "ordered_continuous_risk"
                        if self.method in ORDINAL_HEAD_METHODS
                        else None
                    ),
                    "diagnostic_evidence_bottleneck": self.method in EVIDENCE_METHODS
                    and self.method not in {"fedevi_mtl", "fedevi_seg_only"},
                    "active_auxiliary_tasks": (
                        ["lesion_segmentation"]
                        if self.method == "fedevi_seg_only"
                        else (
                            ["lesion_segmentation", "birads_ordinal"]
                            if self.method in EVIDENCE_METHODS
                            else ["birads_ordinal"]
                        )
                    ),
                    "monotonic_ordinal_evidence": self.method
                    in {
                        "fedevi_monotonic",
                        "fedevi_shuffle",
                        "fedevi_mask_shift",
                        "fedevi_no_taskagg",
                    },
                    "lesion_head_aggregation": (
                        "all_clients_naive"
                        if self.method == "fedevi_no_taskagg"
                        else (
                            "mask_annotated_clients_only"
                            if self.method in EVIDENCE_METHODS
                            else None
                        )
                    ),
                    "ordinal_level_maps": {
                        domain: {
                            str(source): destination
                            for source, destination in sorted(mapping.items())
                        }
                        for domain, mapping in sorted(self.ordinal_level_maps.items())
                    },
                }
                if self.ordinal_bridge is not None or self.method in EVIDENCE_METHODS
                else None
            ),
        )

    def train_source(
        self,
        domain: str,
        global_state: Mapping[str, torch.Tensor],
        round_index: int,
        client_index: int = 0,
    ) -> _ClientPayload:
        loader = self.source_loaders[domain]["train"]
        training_labels, num_samples = self._training_labels_and_count(loader)
        if num_samples <= 0:
            raise ValueError(f"source {domain} train loader must be non-empty")
        local_model = copy.deepcopy(self.global_model)
        local_model.load_state_dict(global_state)
        local_model.train()
        optimizer = torch.optim.AdamW(
            local_model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        class_weights = compute_class_weights(training_labels).to(self.device)
        total_loss = 0.0
        seen_samples = 0
        ordinal_loss_total = 0.0
        lesion_loss_total = 0.0
        ordinal_supervision_samples = 0
        lesion_supervision_samples = 0
        for batch in loader:
            self._validate_batch_domain(batch, domain)
            raw_image = torch.as_tensor(batch["raw_image"]).to(self.device)
            content_mask = torch.as_tensor(batch["content_mask"]).to(self.device)
            labels = torch.as_tensor(batch["label"], dtype=torch.long).to(self.device)
            optimizer.zero_grad(set_to_none=True)
            original_risk_logits = None
            evidence_outputs = None
            if self.method in EVIDENCE_METHODS:
                evidence_outputs = _model_evidence_outputs(
                    local_model, raw_image, content_mask
                )
                original_logits = evidence_outputs.binary_logits
                original_risk_logits = evidence_outputs.ordinal_logits
            elif self.method in ORDINAL_HEAD_METHODS:
                original_logits, original_risk_logits = _model_outputs(
                    local_model, raw_image, content_mask
                )
            else:
                original_logits = _model_logits(local_model, raw_image, content_mask)
            augmented_logits = None
            augmented_risk_logits = None
            base_method = {
                "fedavg_ord": "fedavg",
                "fedevi_dg": "fedavg",
                "fedevi_mtl": "fedavg",
                "fedevi_seg_only": "fedavg",
                "fedevi_shuffle": "fedavg",
                "fedevi_mask_shift": "fedavg",
                "fedevi_no_taskagg": "fedavg",
                "fedevi_monotonic": "fedavg",
            }.get(self.method, self.method)
            loss = compute_method_loss(
                base_method, original_logits, augmented_logits, labels, class_weights
            ).total
            ordinal_component = loss.new_zeros(())
            lesion_component = loss.new_zeros(())
            if self.method in EVIDENCE_METHODS:
                if evidence_outputs is None:
                    raise RuntimeError("diagnostic evidence outputs are unavailable")
                lesion_mask = batch.get("lesion_mask")
                lesion_available = batch.get("lesion_mask_available")
                if lesion_mask is None or lesion_available is None:
                    raise RuntimeError(
                        "fedevi_dg requires lesion mask availability fields"
                    )
                lesion_mask = torch.as_tensor(
                    lesion_mask, dtype=torch.float32, device=self.device
                )
                lesion_available = torch.as_tensor(
                    lesion_available, dtype=torch.bool, device=self.device
                )
                if self.method == "fedevi_mask_shift":
                    lesion_mask = torch.roll(
                        lesion_mask,
                        shifts=(lesion_mask.shape[-2] // 2, lesion_mask.shape[-1] // 2),
                        dims=(-2, -1),
                    )
                lesion_component = supervised_lesion_loss(
                    evidence_outputs.lesion_logits, lesion_mask, lesion_available
                )
                lesion_supervision_samples += int(lesion_available.sum())
                loss = loss + self.config.lesion_weight * lesion_component
            if self.method in CLINICAL_HEAD_METHODS:
                if self.ordinal_bridge is None:
                    raise RuntimeError("clinical auxiliary state is unavailable")
                raw_levels = batch.get("birads_ordinal")
                if raw_levels is None:
                    raw_levels = torch.full_like(
                        labels, float("nan"), dtype=torch.float64
                    )
                raw_levels = torch.as_tensor(
                    raw_levels, dtype=torch.float64, device=self.device
                )
                if domain not in self.ordinal_domains:
                    raw_levels = torch.full_like(raw_levels, float("nan"))
                elif self.method in {"fedevi_shuffle"}:
                    raw_levels = apply_level_map(
                        raw_levels, self.ordinal_level_maps[domain]
                    ).to(self.device)
                normalized_levels = self.ordinal_bridge.normalize(
                    domain, raw_levels
                ).to(self.device)
                finite_levels = torch.isfinite(normalized_levels)
                ordinal_supervision_samples += int(finite_levels.sum())
                if original_risk_logits is None:
                    raise RuntimeError("ordinal risk logits are unavailable")
                ordinal_component = supervised_ordinal_loss(
                    original_risk_logits,
                    augmented_risk_logits,
                    normalized_levels,
                    pairwise_weight=self.config.ordinal_pairwise_weight,
                )
                loss = loss + self.config.ordinal_supervised_weight * ordinal_component
            if (
                self.method
                in {
                    "fedevi_dg",
                    "fedevi_shuffle",
                    "fedevi_mask_shift",
                    "fedevi_no_taskagg",
                    "fedevi_monotonic",
                }
                and original_risk_logits is not None
            ):
                loss = loss + self.config.head_coupling_weight * head_coupling_loss(
                    original_logits, original_risk_logits
                )
            loss.backward()
            optimizer.step()
            batch_size = int(labels.numel())
            total_loss += float(loss.detach()) * batch_size
            ordinal_loss_total += float(ordinal_component.detach()) * batch_size
            lesion_loss_total += float(lesion_component.detach()) * batch_size
            seen_samples += batch_size
        payload = _ClientPayload(
            state_dict=_clone_state(local_model.state_dict()),
            num_samples=num_samples,
            mean_loss=total_loss / seen_samples,
            ordinal_supervision_samples=ordinal_supervision_samples,
            mean_ordinal_loss=ordinal_loss_total / seen_samples,
            lesion_supervision_samples=lesion_supervision_samples,
            mean_lesion_loss=lesion_loss_total / seen_samples,
        )
        del optimizer, local_model
        return payload

    @staticmethod
    def _loader_indices(loader: DataLoader) -> list[int]:
        batch_sampler = loader.batch_sampler
        if type(batch_sampler) is not BatchSampler:
            raise ValueError("training requires the standard BatchSampler")
        if (
            batch_sampler.sampler is not loader.sampler
            or batch_sampler.batch_size != loader.batch_size
            or batch_sampler.drop_last != loader.drop_last
        ):
            raise ValueError(
                "BatchSampler must match loader sampler, batch_size, and drop_last"
            )
        sampler = loader.sampler
        if type(sampler) is SequentialSampler:
            indices = list(range(len(loader.dataset)))
            random_order = False
        elif type(sampler) is SubsetRandomSampler:
            indices = [int(index) for index in sampler.indices]
            random_order = True
        elif type(sampler) is RandomSampler:
            if sampler.replacement or sampler.num_samples != len(loader.dataset):
                raise ValueError(
                    "cannot determine actual training labels for random replacement sampler"
                )
            indices = list(range(len(loader.dataset)))
            random_order = True
        else:
            raise ValueError("unsupported training sampler")
        if batch_sampler.drop_last:
            remainder = len(indices) % batch_sampler.batch_size
            if remainder:
                if random_order:
                    raise ValueError(
                        "cannot determine dropped training labels for randomized sampler"
                    )
                indices = indices[:-remainder]
        return indices

    @classmethod
    def _labels_for_indices(cls, dataset: object, indices: list[int]) -> list[int]:
        if isinstance(dataset, Subset):
            mapped = [int(dataset.indices[index]) for index in indices]
            return cls._labels_for_indices(dataset.dataset, mapped)
        if isinstance(dataset, TensorDataset):
            if len(dataset.tensors) < 2:
                raise ValueError("cannot determine training labels from TensorDataset")
            labels_tensor = torch.as_tensor(dataset.tensors[1])
            return [int(labels_tensor[index].item()) for index in indices]
        frame = getattr(dataset, "frame", None)
        if frame is not None and "label" in frame:
            return [int(value) for value in frame.iloc[indices]["label"].tolist()]
        elif hasattr(dataset, "labels"):
            labels = getattr(dataset, "labels")
            return [int(torch.as_tensor(labels[index]).item()) for index in indices]
        elif hasattr(dataset, "targets"):
            labels = getattr(dataset, "targets")
            return [int(torch.as_tensor(labels[index]).item()) for index in indices]
        elif hasattr(dataset, "samples"):
            samples = getattr(dataset, "samples")
            selected = [samples[index] for index in indices]
            if selected and all(
                (
                    isinstance(sample, Mapping) and "label" in sample
                    for sample in selected
                )
            ):
                return [
                    int(torch.as_tensor(sample["label"]).item()) for sample in selected
                ]
        raise ValueError("cannot determine training labels from dataset")

    @classmethod
    def _field_for_indices(
        cls, dataset: object, indices: list[int], field: str
    ) -> list[object]:
        if isinstance(dataset, Subset):
            mapped = [int(dataset.indices[index]) for index in indices]
            return cls._field_for_indices(dataset.dataset, mapped, field)
        frame = getattr(dataset, "frame", None)
        if frame is not None and field in frame:
            return frame.iloc[indices][field].tolist()
        samples = getattr(dataset, "samples", None)
        if samples is not None:
            selected = [samples[index] for index in indices]
            if all((isinstance(sample, Mapping) for sample in selected)):
                return [sample.get(field, float("nan")) for sample in selected]
        values = []
        for index in indices:
            sample = dataset[index]
            if not isinstance(sample, Mapping):
                raise ValueError(f"cannot determine {field} from dataset")
            values.append(sample.get(field, float("nan")))
        return values

    def _ordinal_records(
        self, requested_domains: set[str] | None
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        records: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for domain in self.source_domains:
            if requested_domains is not None and domain not in requested_domains:
                continue
            loader = self.source_loaders[domain]["train"]
            indices = self._loader_indices(loader)
            labels = torch.tensor(
                self._labels_for_indices(loader.dataset, indices), dtype=torch.long
            )
            raw_levels = self._field_for_indices(
                loader.dataset, indices, "birads_ordinal"
            )
            numeric_levels = []
            for value in raw_levels:
                if torch.is_tensor(value):
                    value = value.item()
                try:
                    numeric_levels.append(
                        float("nan")
                        if value is None or str(value).strip() == ""
                        else float(value)
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid birads_ordinal value for {domain}: {value!r}"
                    ) from error
            levels = torch.tensor(numeric_levels, dtype=torch.float64)
            if torch.isfinite(levels).any():
                records[domain] = (labels, levels)
        if requested_domains is not None:
            unavailable = requested_domains - set(records)
            if unavailable:
                raise ValueError(
                    "selected birads_domains lack finite training annotations: "
                    + ", ".join(sorted(unavailable))
                )
        return records

    @classmethod
    def _training_labels_and_count(cls, loader: DataLoader) -> tuple[list[int], int]:
        indices = cls._loader_indices(loader)
        if not indices:
            raise ValueError("training sampler must select at least one sample")
        return (cls._labels_for_indices(loader.dataset, indices), len(indices))

    def _class_weights(self, loader: DataLoader) -> torch.Tensor:
        labels, _ = self._training_labels_and_count(loader)
        return compute_class_weights(labels)

    def _evaluate_loader(
        self, loader: DataLoader, threshold: float, expected_domain: str
    ) -> dict:
        labels: list[int] = []
        scores: list[float] = []
        ordinal_scores: list[float] = []
        lesion_dice: list[float] = []
        lesion_iou: list[float] = []
        self.global_model.eval()
        with torch.no_grad():
            for batch in loader:
                self._validate_batch_domain(batch, expected_domain)
                raw_image = torch.as_tensor(batch["raw_image"]).to(self.device)
                content_mask = torch.as_tensor(batch["content_mask"]).to(self.device)
                if self.method in EVIDENCE_METHODS:
                    evidence = _model_evidence_outputs(
                        self.global_model, raw_image, content_mask
                    )
                    logits = evidence.binary_logits
                    if self.method != "fedevi_seg_only":
                        ordinal_scores.extend(
                            torch.sigmoid(evidence.ordinal_logits).cpu().tolist()
                        )
                    available = torch.as_tensor(
                        batch["lesion_mask_available"], dtype=torch.bool
                    )
                    if torch.any(available):
                        predicted = (
                            torch.sigmoid(evidence.lesion_logits).cpu() >= 0.5
                        ) & content_mask.cpu().bool()
                        target = torch.as_tensor(batch["lesion_mask"]).bool()
                        for prediction, truth in zip(
                            predicted[available], target[available], strict=True
                        ):
                            intersection = int((prediction & truth).sum())
                            predicted_count = int(prediction.sum())
                            truth_count = int(truth.sum())
                            union = int((prediction | truth).sum())
                            lesion_dice.append(
                                (2.0 * intersection + 1.0)
                                / (predicted_count + truth_count + 1.0)
                            )
                            lesion_iou.append((intersection + 1.0) / (union + 1.0))
                else:
                    logits = _model_logits(self.global_model, raw_image, content_mask)
                probabilities = torch.softmax(logits, dim=1)[:, 1]
                labels.extend(
                    torch.as_tensor(batch["label"], dtype=torch.long).tolist()
                )
                scores.extend(probabilities.detach().cpu().tolist())
        metrics = binary_metrics(labels, scores, threshold)
        if self.method in EVIDENCE_METHODS:
            metrics.update(
                {
                    "ordinal_scores": ordinal_scores,
                    "lesion_dice": (
                        sum(lesion_dice) / len(lesion_dice) if lesion_dice else None
                    ),
                    "lesion_iou": (
                        sum(lesion_iou) / len(lesion_iou) if lesion_iou else None
                    ),
                    "lesion_evaluation_samples": len(lesion_dice),
                }
            )
            metrics.update(clinical_reliability_metrics(labels, scores))
        return metrics

    def _evaluate_source_loader(
        self, loader: DataLoader, domain: str, threshold: float
    ) -> dict:
        if loader is self._target_loader or self._dataset_object_ids(
            loader.dataset
        ) & self._dataset_object_ids(self._target_loader.dataset):
            raise ValueError("source evaluation path rejects target loader")
        if domain not in self.source_domains:
            raise ValueError("source evaluation requires a configured source domain")
        return self._evaluate_loader(loader, threshold, domain)

    def _evaluate_target_loader(self, threshold: float) -> dict:
        return self._evaluate_loader(
            self._target_loader, threshold, self.config.target_domain
        )

    def evaluate_all_source_val(self) -> dict[str, dict]:
        return {
            domain: self._evaluate_source_loader(
                self.source_loaders[domain]["val"], domain, threshold=0.5
            )
            for domain in self.source_domains
        }

    def derive_source_threshold(self) -> float:
        if self._frozen_source_threshold is not None:
            return self._frozen_source_threshold
        if self._best_source_metrics is None:
            raise RuntimeError("source validation metrics are unavailable")
        labels_by_source = {
            domain: metrics["labels"]
            for domain, metrics in self._best_source_metrics.items()
        }
        scores_by_source = {
            domain: metrics["scores"]
            for domain, metrics in self._best_source_metrics.items()
        }
        return select_source_threshold(labels_by_source, scores_by_source)

    def evaluate_target_once(self, threshold: float) -> dict:
        if self._target_evaluation_count != 0:
            raise RuntimeError("target already evaluated")
        if not self._target_ready or self._frozen_source_threshold is None:
            raise RuntimeError("target evaluation is not ready")
        if threshold != self._frozen_source_threshold:
            raise ValueError("threshold must match frozen source threshold")
        metrics = self._evaluate_target_loader(threshold)
        self._target_evaluation_count = 1
        return metrics
