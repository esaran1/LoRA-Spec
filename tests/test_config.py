from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from lora_spec.config import AdapterConfig, ResultRecord


def test_adapter_config_allows_zero_magnitude_null_control() -> None:
    config = AdapterConfig(
        rank=8,
        domain="code",
        epochs=1,
        hf_path="org/adapter",
        magnitude_scale=0.0,
        replicate_id="code-r8-seed-1",
        training_seed=1,
    )

    assert config.magnitude_scale == 0.0


def test_result_record_rejects_invalid_acceptance_rates() -> None:
    with pytest.raises(ValidationError):
        ResultRecord(
            config_hash="abc",
            timestamp=datetime.now(timezone.utc),
            acceptance_rate_overall=1.1,
            acceptance_rate_per_position=[0.8],
            throughput_tps=1.0,
            ttft_ms=1.0,
        )
