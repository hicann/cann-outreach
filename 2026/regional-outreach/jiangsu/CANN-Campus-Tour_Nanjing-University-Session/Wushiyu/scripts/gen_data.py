#!/usr/bin/env python3
from pathlib import Path

import numpy as np


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "test_data" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260916)
    shape = (2, 3, 17, 19)
    x = rng.uniform(-0.9, 0.9, size=shape).astype(np.float16)
    golden = np.arctanh(x.astype(np.float32)).astype(np.float16)
    x.tofile(data_dir / "x.bin")
    golden.tofile(data_dir / "golden.bin")
    print(f"Generated shape={shape}, dtype=float16 in {data_dir}")


if __name__ == "__main__":
    main()
