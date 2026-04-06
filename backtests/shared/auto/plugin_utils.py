from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .phase_state import PhaseState
from .types import Experiment, GreedyResult, GreedyRound, ScoredCandidate

DEFAULT_AUTO_WORKERS = 3
_FALLBACK_EXCEPTIONS = (PermissionError, OSError, EOFError, BrokenPipeError, RuntimeError)


def deserialize_experiments(raw: list[dict[str, Any]] | None) -> list[Experiment]:
    experiments: list[Experiment] = []
    for item in raw or []:
        if isinstance(item, Experiment):
            experiments.append(item)
        elif isinstance(item, dict):
            experiments.append(Experiment(name=item.get("name", ""), mutations=item.get("mutations", {})))
    return experiments


def greedy_result_to_dict(greedy_result: GreedyResult) -> dict[str, Any]:
    return {
        "base_score": greedy_result.base_score,
        "final_score": greedy_result.final_score,
        "accepted_count": greedy_result.accepted_count,
        "total_candidates": greedy_result.total_candidates,
        "kept_features": greedy_result.kept_features,
        "rounds": [asdict(round_result) for round_result in greedy_result.rounds],
        "final_metrics": greedy_result.final_metrics,
    }


def greedy_result_from_state(
    state: PhaseState,
    *,
    phase: int,
    final_metrics: dict[str, float],
) -> GreedyResult:
    phase_result = state.phase_results.get(phase, {})
    rounds = [GreedyRound(**round_result) for round_result in phase_result.get("rounds", [])]
    return GreedyResult(
        base_score=float(phase_result.get("base_score", 0.0)),
        final_score=float(phase_result.get("final_score", 0.0)),
        final_mutations=dict(phase_result.get("final_mutations", state.cumulative_mutations)),
        kept_features=list(phase_result.get("kept_features", [])),
        rounds=rounds,
        final_metrics=final_metrics,
        total_candidates=int(phase_result.get("total_candidates", len(rounds))),
        accepted_count=int(phase_result.get("accepted_count", len(phase_result.get("kept_features", [])))),
        elapsed_seconds=float(phase_result.get("elapsed_seconds", 0.0)),
    )


def seen_experiment_names(state: PhaseState) -> set[str]:
    seen: set[str] = set()
    for phase_result in state.phase_results.values():
        seen.update(name for name in phase_result.get("kept_features", []) if name)
        for item in phase_result.get("suggested_experiments", []):
            if isinstance(item, Experiment):
                if item.name:
                    seen.add(item.name)
            elif isinstance(item, dict):
                name = item.get("name", "")
                if name:
                    seen.add(name)
    return seen


def resolve_worker_processes(max_workers: int | None) -> int:
    if max_workers is not None:
        return max(1, int(max_workers))

    cpu_count = max(1, os.cpu_count() or DEFAULT_AUTO_WORKERS)
    return min(DEFAULT_AUTO_WORKERS, cpu_count)


def mutation_signature(mutations: dict[str, Any]) -> str:
    encoded = json.dumps(
        mutations,
        sort_keys=True,
        default=_signature_default,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def clone_scored_candidate(result: ScoredCandidate, *, name: str | None = None) -> ScoredCandidate:
    return ScoredCandidate(
        name=name or result.name,
        score=float(result.score),
        rejected=bool(result.rejected),
        reject_reason=str(result.reject_reason or ""),
        metrics=dict(result.metrics or {}),
    )


class CachedBatchEvaluator:
    def __init__(
        self,
        delegate,
        *,
        cache: dict[str, ScoredCandidate] | None = None,
        seed_results: dict[str, ScoredCandidate] | None = None,
        signature_prefix: str = "",
        metrics_cache: dict[str, dict[str, float]] | None = None,
    ):
        self._delegate = delegate
        self._cache = cache if cache is not None else {}
        self._signature_prefix = signature_prefix
        self._metrics_cache = metrics_cache
        for key, result in (seed_results or {}).items():
            self._cache[self._cache_key(key)] = clone_scored_candidate(result)
            if self._metrics_cache is not None and result.metrics:
                self._metrics_cache[key] = dict(result.metrics)

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]) -> list[ScoredCandidate]:
        if not candidates:
            return []

        results: list[ScoredCandidate | None] = [None] * len(candidates)
        pending_candidates: list[Experiment] = []
        pending_keys: list[str] = []
        key_to_indexes: dict[str, list[int]] = {}

        for index, candidate in enumerate(candidates):
            merged = dict(current_mutations)
            merged.update(candidate.mutations)
            key = mutation_signature(merged)
            cached = self._cache.get(self._cache_key(key))
            if cached is not None:
                results[index] = clone_scored_candidate(cached, name=candidate.name)
                continue
            key_to_indexes.setdefault(key, []).append(index)
            if len(key_to_indexes[key]) == 1:
                pending_keys.append(key)
                pending_candidates.append(candidate)

        if pending_candidates:
            scored = self._delegate(pending_candidates, current_mutations)
            if len(scored) != len(pending_candidates):
                raise RuntimeError(
                    f"Batch evaluator returned {len(scored)} results for {len(pending_candidates)} candidates."
                )
            for key, prototype, result in zip(pending_keys, pending_candidates, scored, strict=True):
                template = clone_scored_candidate(result, name=prototype.name)
                self._cache[self._cache_key(key)] = template
                if self._metrics_cache is not None and result.metrics:
                    self._metrics_cache[key] = dict(result.metrics)
                for index in key_to_indexes.get(key, []):
                    results[index] = clone_scored_candidate(template, name=candidates[index].name)

        return [result for result in results if result is not None]

    def close(self) -> None:
        close_fn = getattr(self._delegate, "close", None)
        if callable(close_fn):
            close_fn()

    def _cache_key(self, mutation_key: str) -> str:
        if not self._signature_prefix:
            return mutation_key
        return f"{self._signature_prefix}:{mutation_key}"


class ResilientBatchEvaluator:
    def __init__(
        self,
        preferred_factory: Callable[[], Any],
        fallback_factory: Callable[[], Any],
        *,
        description: str = "batch evaluator",
        logger: logging.Logger | None = None,
        retryable_exceptions: tuple[type[BaseException], ...] = _FALLBACK_EXCEPTIONS,
    ):
        self._preferred_factory = preferred_factory
        self._fallback_factory = fallback_factory
        self._description = description
        self._logger = logger or logging.getLogger(__name__)
        self._retryable_exceptions = retryable_exceptions
        self._using_fallback = False
        self._delegate = self._build_delegate()

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        try:
            return self._delegate(candidates, current_mutations)
        except self._retryable_exceptions as exc:
            self._switch_to_fallback(exc)
            return self._delegate(candidates, current_mutations)

    def close(self) -> None:
        self._close_delegate(force=False)

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    def _build_delegate(self):
        try:
            return self._preferred_factory()
        except self._retryable_exceptions as exc:
            self._using_fallback = True
            self._logger.warning(
                "%s initialization failed; falling back to local execution: %s",
                self._description,
                exc,
            )
            return self._fallback_factory()

    def _switch_to_fallback(self, exc: BaseException) -> None:
        if self._using_fallback:
            raise exc
        self._using_fallback = True
        self._logger.warning(
            "%s failed during parallel execution; retrying locally: %s",
            self._description,
            exc,
        )
        self._close_delegate(force=True)
        self._delegate = self._fallback_factory()

    def _close_delegate(self, *, force: bool) -> None:
        delegate = getattr(self, "_delegate", None)
        if delegate is None:
            return
        try:
            terminate_fn = getattr(delegate, "terminate", None)
            if force and callable(terminate_fn):
                terminate_fn()
                return
            close_fn = getattr(delegate, "close", None)
            if callable(close_fn):
                close_fn()
        finally:
            self._delegate = None


def _signature_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)
