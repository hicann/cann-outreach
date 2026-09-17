#!/usr/bin/python3
# coding=utf-8
import sys
import numpy as np

relative_tol = 1e-3
absolute_tol = 1e-5
error_tol = 1e-3


def verify_result(output, golden):
    output = np.fromfile(output, dtype=np.float16).reshape(-1)
    golden = np.fromfile(golden, dtype=np.float16).reshape(-1)
    close = np.isclose(output, golden, rtol=relative_tol, atol=absolute_tol, equal_nan=True)
    bad = np.where(close == False)[0]
    for index in range(min(len(bad), 100)):
        i = bad[index]
        g, o = golden[i], output[i]
        rdiff = abs(float(o) - float(g)) / (abs(float(g)) + 1e-12)
        print("data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" % (i, g, o, rdiff))
    error_ratio = float(bad.size) / golden.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol


if __name__ == "__main__":
    try:
        if not verify_result(sys.argv[1], sys.argv[2]):
            raise ValueError("[ERROR] result error")
        print("test pass")
    except Exception as e:
        print(e)
        sys.exit(1)
