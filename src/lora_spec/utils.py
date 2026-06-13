from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from pydantic import BaseModel


def setup_logging(verbose: bool = False, logger_name: str = "lora_spec") -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        return {
            field: _to_jsonable(getattr(value, field))
            for field in value.__dataclass_fields__
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json(data: Any) -> str:
    return json.dumps(_to_jsonable(data), sort_keys=True, separators=(",", ":"))


def compute_config_hash(config: Any) -> str:
    payload = canonical_json(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def get_git_hash(cwd: str | Path | None = None) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return output.strip()


def get_git_dirty(cwd: str | Path | None = None) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def get_runtime_metadata() -> dict[str, Any]:
    package_versions: dict[str, str] = {}
    for package_name in ("transformers", "peft", "vllm", "datasets", "safetensors"):
        try:
            package_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package_name] = "not-installed"
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "package_versions": package_versions,
    }
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        metadata["gpu_count"] = device_count
        metadata["gpu_names"] = [torch.cuda.get_device_name(index) for index in range(device_count)]
        metadata["cuda_version"] = torch.version.cuda
    else:
        metadata["gpu_count"] = 0
        metadata["gpu_names"] = []
        metadata["cuda_version"] = None
    return metadata


def resolve_torch_dtype(
    value: str | None = "auto",
    device: str | torch.device | None = None,
) -> torch.dtype:
    normalized = (value or "auto").lower()
    if normalized == "auto":
        target_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if target_device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[normalized]


def write_json_result(
    payload: Mapping[str, Any] | BaseModel | Any,
    output_dir: str | Path,
    stem: str,
    config: Mapping[str, Any] | BaseModel | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    cwd: str | Path | None = None,
) -> Path:
    directory = ensure_dir(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_hash = compute_config_hash(config if config is not None else payload)
    data = _to_jsonable(payload)
    if not isinstance(data, dict):
        data = {"result": data}
    metadata = dict(data.get("metadata", {}))
    metadata.update(extra_metadata or {})
    metadata.setdefault("git_hash", get_git_hash(cwd=cwd))
    metadata.setdefault("git_dirty", get_git_dirty(cwd=cwd))
    metadata.setdefault("timestamp", timestamp)
    metadata.setdefault("runtime", get_runtime_metadata())
    if config is not None:
        data["full_config"] = _to_jsonable(config)
    data["config_hash"] = config_hash
    data["metadata"] = metadata
    path = directory / f"{stem}_{timestamp}_{config_hash}.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {path} must contain a mapping at the top level")
    return data


def _coerce_override(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    for converter in (int, float):
        try:
            return converter(value)
        except ValueError:
            continue
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def apply_cli_overrides(base: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    merged = dict(base)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be in key=value form: {item}")
        key, raw_value = item.split("=", 1)
        value = _coerce_override(raw_value)
        cursor: dict[str, Any] = merged
        parts = key.split(".")
        for part in parts[:-1]:
            child = cursor.get(part)
            if child is None:
                child = {}
                cursor[part] = child
            if not isinstance(child, dict):
                raise ValueError(f"Cannot assign nested override into non-mapping key: {key}")
            cursor = child
        cursor[parts[-1]] = value
    return merged


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="CLI override in dotted.key=value form. May be repeated.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_config(config_path: str | None, overrides: list[str] | None) -> dict[str, Any]:
    config_data: dict[str, Any] = {}
    if config_path:
        config_data = load_yaml(config_path)
    return apply_cli_overrides(config_data, overrides)


def get_config_value(
    config: Mapping[str, Any],
    args: argparse.Namespace,
    key: str,
    default: Any = None,
) -> Any:
    if key in config:
        return config[key]
    return getattr(args, key, default)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
