from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .spec import OpSpec, ShapeGenConfig


def _rand_shape(cfg: ShapeGenConfig, rng: torch.Generator) -> List[int]:
    ndim = cfg.sample_ndim(rng)
    shape: List[int] = []
    product = 1
    for _ in range(ndim):
        d = cfg.sample_dim(rng)
        if product > 0 and d > max(1, cfg.max_elements // product):
            d = max(1, cfg.max_elements // product)
        shape.append(d)
        product *= d
    return shape


def gen_input_shapes(spec: OpSpec, rng: torch.Generator) -> Tuple[List[List[int]], Dict[str, int]]:
    kind = spec.kind
    hint: Dict[str, int] = {}
    if kind in ("elementwise", "reduction", "custom"):
        base = _rand_shape(spec.shape_gen, rng)
        hint["ndim"] = len(base)
        shapes = [list(base) for _ in range(spec.num_inputs)]
        return shapes, hint
    if kind == "broadcast":
        base = _rand_shape(spec.shape_gen, rng)
        hint["ndim"] = len(base)
        shapes: List[List[int]] = []
        for i in range(spec.num_inputs):
            s = list(base)
            if i > 0:
                k = int(torch.randint(0, len(s) + 1, (1,), generator=rng).item())
                for j in range(k):
                    if torch.rand(1, generator=rng).item() < 0.5:
                        s[j] = 1
            shapes.append(s)
        return shapes, hint
    if kind == "matmul":
        cfg = spec.shape_gen
        m = cfg.sample_dim(rng)
        k = cfg.sample_dim(rng)
        n = cfg.sample_dim(rng)
        hint["ndim"] = 2
        if spec.num_inputs == 2:
            return [[m, k], [k, n]], hint
        shapes = [[m, k], [k, n]]
        for _ in range(2, spec.num_inputs):
            shapes.append([m, n])
        return shapes, hint
    base = _rand_shape(spec.shape_gen, rng)
    hint["ndim"] = len(base)
    return [list(base) for _ in range(spec.num_inputs)], hint


def make_input_tensor(shape: List[int], dtype: torch.dtype, strategy: str, rng: torch.Generator) -> torch.Tensor:
    if strategy == "zeros":
        return torch.zeros(shape, dtype=dtype)
    if strategy == "ones":
        return torch.ones(shape, dtype=dtype)
    if strategy == "small":
        return (torch.rand(shape, generator=rng) * 0.2 - 0.1).to(dtype)
    if strategy == "mixed":
        t = torch.randn(shape, dtype=dtype, generator=rng)
        mask = torch.randint(0, 2, shape, generator=rng).to(dtype)
        return t * mask
    if strategy == "int":
        if dtype.is_floating_point:
            return torch.randint(-100, 101, shape, generator=rng).to(dtype)
        return torch.randint(-100, 101, shape, generator=rng).to(dtype)
    return torch.randn(shape, dtype=dtype, generator=rng)
