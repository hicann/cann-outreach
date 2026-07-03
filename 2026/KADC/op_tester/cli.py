from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .config import load_config
from .runner import run_tests


def _parse_shape_arg(s: str) -> List[List[int]]:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        outer = json.loads(s)
        return [list(inner) for inner in outer]
    parts = [p.strip() for p in s.split(";") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"invalid shape spec: {s!r}")
    shapes = []
    for p in parts:
        if p.startswith("[") and p.endswith("]"):
            shapes.append(json.loads(p))
        else:
            shapes.append([int(x) for x in p.replace(",", " ").split()])
    return shapes


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="op-tester",
        description="Generic Ascend C operator tester (PyTorch pathway vs golden).",
    )
    parser.add_argument("config", help="Path to operator YAML config file")
    parser.add_argument("--so", help="Override shared library path", default=None)
    parser.add_argument("-n", "--num-cases", type=int, default=None, help="Number of auto-generated shape cases")
    parser.add_argument("--shape", action="append", type=_parse_shape_arg, default=[],
                        help="Explicit per-input shapes, e.g. '8,16;8,16' or '[[8,16],[8,16]]' (repeatable)")
    parser.add_argument("--no-auto", action="store_true", help="Skip auto shape generation, only run --shape cases")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--json", dest="json_out", default=None, help="Write results as JSON to this path")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    spec, config_dir = load_config(args.config)
    if args.so:
        spec.so = args.so
    if args.seed is not None:
        spec.shape_gen.seed = args.seed
    if args.atol is not None:
        spec.atol = args.atol
    if args.rtol is not None:
        spec.rtol = args.rtol
    if args.no_auto:
        spec.shape_gen.num_cases = 0
    extra_shapes = args.shape or []

    results = run_tests(
        spec,
        config_dir,
        extra_shapes=extra_shapes,
        num_cases=args.num_cases,
        verbose=not args.quiet,
    )

    if args.json_out:
        import dataclasses
        payload = [dataclasses.asdict(r) for r in results]
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
