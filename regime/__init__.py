"""MR-AWQ Meta Allocator v2 — weekly regime-based allocation signal engine."""

from .config import MetaConfig
from .engine import run_signal_engine

__all__ = ["MetaConfig", "run_signal_engine"]
