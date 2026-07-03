from __future__ import annotations

import dataclasses
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .shape_gen import gen_input_shapes, make_input_tensor
from .spec import OpSpec


@dataclass
class CaseResult:
    label: str
    shapes: List[List[int]]
    attrs: Dict[str, Any]
    passed: bool
    max_diff: float
    mean_diff: float
    msg: str = ""
    elapsed: float = 0.0


def _to_cpu_list(x):
    if isinstance(x, (tuple, list)):
        return [_to_cpu_list(t) for t in x]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def _flatten_outputs(x) -> List[torch.Tensor]:
    if isinstance(x, (tuple, list)):
        out = []
        for t in x:
            out.extend(_flatten_outputs(t))
        return out
    return [x]


def _compare(npu_out, golden_out, atol: float, rtol: float) -> Tuple[bool, float, float, str]:
    npu_list = _flatten_outputs(_to_cpu_list(npu_out))
    gold_list = _flatten_outputs(_to_cpu_list(golden_out))
    if len(npu_list) != len(gold_list):
        return False, float("inf"), float("inf"), f"output count mismatch: npu={len(npu_list)} golden={len(gold_list)}"
    max_diff = 0.0
    mean_diff = 0.0
    total = 0
    for a, b in zip(npu_list, gold_list):
        if a.shape != b.shape:
            return False, float("inf"), float("inf"), f"shape mismatch: npu={tuple(a.shape)} golden={tuple(b.shape)}"
        if a.dtype != b.dtype:
            b = b.to(a.dtype)
        d = (a.float() - b.float()).abs()
        max_diff = max(max_diff, float(d.max().item()) if d.numel() else 0.0)
        mean_diff += float(d.mean().item()) * d.numel()
        total += d.numel()
    mean_diff = mean_diff / total if total else 0.0
    ok = True
    for a, b in zip(npu_list, gold_list):
        if a.dtype != b.dtype:
            b = b.to(a.dtype)
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            ok = False
            break
    return ok, max_diff, mean_diff, ""


def _build_inputs(spec: OpSpec, shapes: List[List[int]], rng: torch.Generator) -> List[torch.Tensor]:
    inputs = []
    for i, s in enumerate(shapes):
        dt = spec.input_dtype(i)
        strat = spec.input_init_strategy(i)
        inputs.append(make_input_tensor(s, dt, strat, rng))
    return inputs


def _sample_attrs(spec: OpSpec, rng: torch.Generator, hint: Dict[str, int]) -> Dict[str, Any]:
    out = {}
    for a in spec.attrs:
        out[a.name] = a.sample(rng, hint)
    return out


def _call_npu_op(op_fn, inputs_npu: List[torch.Tensor], attrs: Dict[str, Any]):
    return op_fn(*inputs_npu, **attrs)


def run_one(
    spec: OpSpec,
    op_fn,
    golden_fn,
    shapes: List[List[int]],
    attrs: Dict[str, Any],
    label: str,
) -> CaseResult:
    rng = torch.Generator().manual_seed(sum(hash(tuple(s)) for s in shapes) & 0xFFFFFFFF)
    inputs = _build_inputs(spec, shapes, rng)
    try:
        golden_out = golden_fn(list(inputs), attrs)
    except TypeError:
        golden_out = golden_fn(*inputs, **attrs)
    inputs_npu = [t.npu() for t in inputs]
    t0 = time.perf_counter()
    try:
        npu_out = _call_npu_op(op_fn, inputs_npu, attrs)
    except Exception as e:
        return CaseResult(label, shapes, attrs, False, float("inf"), float("inf"), f"npu call error: {e}", time.perf_counter() - t0)
    torch.npu.synchronize() if hasattr(torch, "npu") and torch.npu.is_available() else None
    elapsed = time.perf_counter() - t0
    ok, mx, mn, msg = _compare(npu_out, golden_out, spec.atol, spec.rtol)
    return CaseResult(label, shapes, attrs, ok, mx, mn, msg, elapsed)


def _ensure_op_fn(spec: OpSpec, so_path: str):
    if not os.path.isfile(so_path):
        raise FileNotFoundError(f"shared library not found: {so_path} (build the operator first)")
    torch.ops.load_library(so_path)
    ns, _, opname = spec.name.rpartition(".")
    ns = ns or "npu"
    try:
        ns_obj = getattr(torch.ops, ns)
    except AttributeError:
        raise AttributeError(f"torch.ops.{ns} namespace not found after loading {so_path}")
    op_fn = getattr(ns_obj, opname, None)
    if op_fn is None:
        raise AttributeError(f"torch.ops.{ns}.{opname} not found; available: {dir(ns_obj)}")
    return op_fn


def _fmt_shape(shapes: List[List[int]]) -> str:
    return "x".join("[" + ",".join(str(d) for d in s) + "]" for s in shapes)


def run_tests(
    spec: OpSpec,
    config_dir: str,
    *,
    extra_shapes: Optional[List[List[List[int]]]] = None,
    num_cases: Optional[int] = None,
    verbose: bool = True,
    load_so: bool = True,
    op_fn=None,
    golden_fn=None,
) -> List[CaseResult]:
    from .config import load_golden_fn, resolve_so_path

    if golden_fn is None:
        golden_fn = load_golden_fn(spec, config_dir)
    if op_fn is None and load_so:
        so_path = resolve_so_path(spec, config_dir)
        op_fn = _ensure_op_fn(spec, so_path)
    elif op_fn is None:
        raise RuntimeError("op_fn not provided and load_so=False")

    rng = torch.Generator().manual_seed(spec.shape_gen.seed)
    cases: List[Tuple[str, List[List[int]], Dict[str, Any]]] = []

    n_auto = num_cases if num_cases is not None else spec.shape_gen.num_cases
    for i in range(n_auto):
        shapes, hint = gen_input_shapes(spec, rng)
        attrs = _sample_attrs(spec, rng, hint)
        cases.append((f"A{i+1}", shapes, attrs))

    explicit = list(spec.explicit_shapes)
    if extra_shapes:
        explicit.extend(extra_shapes)
    for j, entry in enumerate(explicit):
        if isinstance(entry, dict):
            shapes = entry["shapes"]
            attrs = entry.get("attrs", {})
        else:
            shapes = entry
            attrs = {}
        shapes = [list(s) for s in shapes]
        full_attrs = _sample_attrs(spec, rng, {"ndim": len(shapes[0]) if shapes else 1})
        full_attrs.update(attrs)
        cases.append((f"S{j+1}", shapes, full_attrs))

    results: List[CaseResult] = []
    if verbose:
        print(f"\n{'='*78}")
        print(f"Op: {spec.name}  kind={spec.kind}  inputs={spec.num_inputs}  outputs={spec.num_outputs}")
        print(f"atol={spec.atol}  rtol={spec.rtol}  cases={len(cases)}")
        print(f"{'='*78}")
        print(f"{'case':<6}{'shapes':<48}{'result':<10}{'max_diff':>14}{'time(ms)':>10}")
        print("-" * 78)
    for label, shapes, attrs in cases:
        r = run_one(spec, op_fn, golden_fn, shapes, attrs, label)
        results.append(r)
        if verbose:
            status = "PASS" if r.passed else "FAIL"
            print(f"{label:<6}{_fmt_shape(shapes):<48}{status:<10}{r.max_diff:>14.3e}{r.elapsed*1000:>10.2f}")
            if not r.passed and r.msg:
                print(f"       └─ {r.msg}")
            if not r.passed and r.attrs:
                print(f"       └─ attrs={r.attrs}")
    if verbose:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        print("-" * 78)
        print(f"Total: {total}  Passed: {passed}  Failed: {total - passed}  =>  {'PASSED' if passed == total else 'FAILED'}")
        print("=" * 78)
    return results
