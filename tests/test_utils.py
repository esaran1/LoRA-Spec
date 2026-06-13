from __future__ import annotations

import json
from pathlib import Path

from lora_spec.utils import write_json_result


def test_write_json_result_normalizes_nonfinite_values(tmp_path: Path) -> None:
    output = write_json_result(
        payload={"nan": float("nan"), "positive_infinity": float("inf")},
        output_dir=tmp_path,
        stem="finite_json",
        config={"seed": 7},
        cwd=tmp_path,
    )
    raw = output.read_text(encoding="utf-8")
    parsed = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert parsed["nan"] is None
    assert parsed["positive_infinity"] is None
