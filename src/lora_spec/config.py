from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelPairConfig(BaseModel):
    target_model: str
    draft_model: str
    tensor_parallel_degree: int = Field(1, ge=1)


class AdapterConfig(BaseModel):
    rank: int = Field(..., ge=1)
    domain: str
    epochs: int | None = Field(default=None, ge=1)
    hf_path: str
    magnitude_scale: float = Field(1.0, gt=0.0)
    target_model: str | None = None


class ExperimentConfig(BaseModel):
    model_pair: ModelPairConfig
    adapter: AdapterConfig | None = None
    num_prompts: int = Field(..., ge=1)
    dataset: str
    seed: int = 7
    speculation_length: int = Field(4, ge=1)
    max_tokens: int = Field(128, ge=1)
    warmup_prompts: int = Field(2, ge=1)
    warmup_tokens: int = Field(8, ge=1)
    gpu_memory_utilization: float = Field(0.85, gt=0.0, le=1.0)
    trust_remote_code: bool = False


class ResultRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    config_hash: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    acceptance_rate_overall: float
    acceptance_rate_per_position: list[float]
    throughput_tps: float
    ttft_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("acceptance_rate_per_position")
    @classmethod
    def validate_acceptance_rate_per_position(
        cls,
        values: list[float],
    ) -> list[float]:
        if not values:
            raise ValueError("acceptance_rate_per_position must not be empty")
        return values
