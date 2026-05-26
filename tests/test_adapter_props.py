from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from safetensors.torch import save_file

from lora_spec.adapter_props import compute_adapter_properties, load_lora_matrices


def test_load_lora_matrices_and_compute_properties(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    save_file(
        {
            "layer0.lora_A.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "layer0.lora_B.weight": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            "layer1.lora_A.weight": torch.tensor([[1.0, 1.0]]),
            "layer1.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    matrices = load_lora_matrices(adapter_dir)
    assert set(matrices) == {"layer0", "layer1"}

    base_model = torch.nn.Linear(2, 2, bias=False)
    properties = compute_adapter_properties(adapter_dir, base_model=base_model)
    assert properties.adapted_parameter_count == 10
    assert properties.frobenius_norm_sum == pytest.approx(9.0710678118, rel=1e-5)
    assert properties.max_spectral_norm == pytest.approx(5.0, rel=1e-5)
    assert properties.layer_frobenius_norms["layer0"] == pytest.approx(2.8284271247, rel=1e-5)
    assert properties.adapted_parameter_fraction == pytest.approx(2.5)
