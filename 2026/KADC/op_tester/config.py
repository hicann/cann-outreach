from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, List

import yaml

from .spec import AttrSpec, OpSpec, ShapeGenConfig, resolve_dtype


def _load_golden(spec: OpSpec, config_dir: str):
    mod_name = spec.golden_module
    if mod_name.endswith(".py") or os.path.sep in mod_name or "/" in mod_name:
        path = mod_name if os.path.isabs(mod_name) else os.path.join(config_dir, mod_name)
    else:
        path = os.path.join(config_dir, f"{mod_name}.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"golden module not found: {path}")
    unique = f"_op_tester_golden_{abs(hash(path))}"
    loader = importlib.util.spec_from_file_location(unique, path)
    mod = importlib.util.module_from_spec(loader)
    sys.modules[unique] = mod
    loader.loader.exec_module(mod)
    func = getattr(mod, spec.golden_func, None)
    if func is None:
        raise AttributeError(f"golden module {path} has no function '{spec.golden_func}'")
    return func


def load_config(path: str) -> tuple[OpSpec, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(path))

    op_raw = raw.get("op", raw)
    name = op_raw["name"]
    so = op_raw.get("so") or op_raw.get("so_path") or f"build/lib{name}_ops.so"

    dtype_raw = op_raw.get("dtype")
    input_dtypes = []
    if "input_dtypes" in op_raw:
        input_dtypes = resolve_dtype(op_raw["input_dtypes"])
    output_dtypes = []
    if "output_dtypes" in op_raw:
        output_dtypes = resolve_dtype(op_raw["output_dtypes"])

    attrs = []
    for a in op_raw.get("attrs", []) or []:
        attrs.append(AttrSpec(
            name=a["name"],
            type=a.get("type", "float"),
            default=a.get("default"),
            gen=a.get("gen", "default"),
            low=a.get("low"),
            high=a.get("high"),
            choices=a.get("choices", []) or [],
        ))

    sg_raw = raw.get("shape_gen", {})
    shape_gen = ShapeGenConfig(
        ndim_min=int(sg_raw.get("ndim_min", 1)),
        ndim_max=int(sg_raw.get("ndim_max", 4)),
        dim_min=int(sg_raw.get("dim_min", 1)),
        dim_max=int(sg_raw.get("dim_max", 4096)),
        num_cases=int(sg_raw.get("num_cases", 10)),
        seed=int(sg_raw.get("seed", 0)),
        power_of_two_bias=float(sg_raw.get("power_of_two_bias", 0.3)),
    )

    explicit = raw.get("shapes", []) or []

    golden_raw = op_raw.get("golden", {}) or {}
    golden_module = golden_raw.get("module", "golden")
    golden_func = golden_raw.get("func", "compute")

    spec = OpSpec(
        name=name,
        so=so,
        kind=op_raw.get("kind", "elementwise"),
        num_inputs=int(op_raw.get("num_inputs", 1)),
        num_outputs=int(op_raw.get("num_outputs", 1)),
        dtype=resolve_dtype(dtype_raw) if dtype_raw else None,
        input_dtypes=input_dtypes,
        output_dtypes=output_dtypes,
        input_init=list(op_raw.get("input_init", []) or []),
        attrs=attrs,
        atol=float(op_raw.get("atol", 1e-6)),
        rtol=float(op_raw.get("rtol", 1e-6)),
        golden_module=golden_module,
        golden_func=golden_func,
        shape_gen=shape_gen,
        explicit_shapes=explicit,
        extra=op_raw.get("extra", {}) or {},
    )
    if spec.dtype is None and not spec.input_dtypes:
        spec.dtype = resolve_dtype("float32")
    return spec, config_dir


def resolve_so_path(spec: OpSpec, config_dir: str) -> str:
    so = spec.so
    if os.path.isabs(so) and os.path.isfile(so):
        return so
    candidates = [
        os.path.join(config_dir, so),
        so,
        os.path.join(config_dir, "build", os.path.basename(so)),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


def load_golden_fn(spec: OpSpec, config_dir: str):
    return _load_golden(spec, config_dir)
