#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify AtanhCustom output")
    parser.add_argument("output", type=Path, help="NPU output binary")
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "test_data"
        / "data"
        / "golden.bin",
    )
    args = parser.parse_args()
    actual = np.fromfile(args.output, dtype=np.float16)
    expected = np.fromfile(args.golden, dtype=np.float16)
    if actual.size != expected.size:
        raise SystemExit(
            f"size mismatch: actual={actual.size}, expected={expected.size}"
        )
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)
    print(f"PASS: verified {actual.size} float16 elements")


if __name__ == "__main__":
    main()
