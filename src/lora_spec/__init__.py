"""LoRA-Spec package."""

from typing import Any

from .config import AdapterConfig, ExperimentConfig, ModelPairConfig, ResultRecord

__all__ = [
    "AdapterConfig",
    "ExperimentConfig",
    "ModelPairConfig",
    "ResultRecord",
    "center_logit_shift_rows",
    "effective_rank",
    "spectral_analysis",
]


def __getattr__(name: str) -> Any:
    if name in {"center_logit_shift_rows", "effective_rank", "spectral_analysis"}:
        from . import theory

        return getattr(theory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
