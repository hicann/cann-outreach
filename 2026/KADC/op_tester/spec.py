from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

_DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "int64": torch.int64,
    "long": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def resolve_dtype(name: str | List[str]) -> Any:
    if isinstance(name, list):
        return [resolve_dtype(n) for n in name]
    key = str(name).lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"unknown dtype: {name!r} (supported: {sorted(_DTYPE_MAP)})")
    return _DTYPE_MAP[key]


@dataclass
class AttrSpec:
    name: str
    type: str = "float"
    default: Any = None
    gen: str = "default"
    low: Any = None
    high: Any = None
    choices: List[Any] = field(default_factory=list)

    def sample(self, rng: torch.Generator, hint: Dict[str, Any] | None = None) -> Any:
        hint = hint or {}
        if self.name in hint:
            return hint[self.name]
        if self.gen == "default":
            return self.default
        if self.gen == "choice":
            if not self.choices:
                return self.default
            idx = int(torch.randint(0, len(self.choices), (1,), generator=rng).item())
            return self.choices[idx]
        if self.gen == "dim":
            ndim = hint.get("ndim", 1)
            return int(torch.randint(0, ndim, (1,), generator=rng).item())
        if self.gen == "random_int":
            lo = int(self.low if self.low is not None else 0)
            hi = int(self.high if self.high is not None else 10)
            return int(torch.randint(lo, hi + 1, (1,), generator=rng).item())
        if self.gen == "random_float":
            lo = float(self.low if self.low is not None else -1.0)
            hi = float(self.high if self.high is not None else 1.0)
            return float(lo + (hi - lo) * torch.rand(1, generator=rng).item())
        return self.default


@dataclass
class ShapeGenConfig:
    ndim_min: int = 1
    ndim_max: int = 4
    dim_min: int = 1
    dim_max: int = 4096
    num_cases: int = 10
    seed: int = 0
    power_of_two_bias: float = 0.3
    max_elements: int = 16_000_000

    def sample_ndim(self, rng: torch.Generator) -> int:
        if self.ndim_max <= self.ndim_min:
            return self.ndim_min
        return int(torch.randint(self.ndim_min, self.ndim_max + 1, (1,), generator=rng).item())

    def sample_dim(self, rng: torch.Generator) -> int:
        lo = max(1, self.dim_min)
        hi = max(lo, self.dim_max)
        if torch.rand(1, generator=rng).item() < self.power_of_two_bias:
            p = int(torch.randint(4, 13, (1,), generator=rng).item())
            return min(hi, max(lo, 1 << p))
        return int(torch.randint(lo, hi + 1, (1,), generator=rng).item())


@dataclass
class OpSpec:
    name: str
    so: str
    kind: str = "elementwise"
    num_inputs: int = 1
    num_outputs: int = 1
    dtype: Any = None
    input_dtypes: List[Any] = field(default_factory=list)
    output_dtypes: List[Any] = field(default_factory=list)
    input_init: List[str] = field(default_factory=list)
    attrs: List[AttrSpec] = field(default_factory=list)
    atol: float = 1e-6
    rtol: float = 1e-6
    golden_module: str = "golden"
    golden_func: str = "compute"
    shape_gen: ShapeGenConfig = field(default_factory=ShapeGenConfig)
    explicit_shapes: List[List[List[int]]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def input_dtype(self, i: int) -> Any:
        if self.input_dtypes:
            return self.input_dtypes[i % len(self.input_dtypes)]
        return self.dtype

    def input_init_strategy(self, i: int) -> str:
        if self.input_init:
            return self.input_init[i % len(self.input_init)]
        return "random"

    def attr_dict(self) -> Dict[str, AttrSpec]:
        return {a.name: a for a in self.attrs}
