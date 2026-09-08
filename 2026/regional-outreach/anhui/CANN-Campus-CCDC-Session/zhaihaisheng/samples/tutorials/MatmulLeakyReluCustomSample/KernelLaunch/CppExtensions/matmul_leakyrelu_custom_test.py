#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import torch
import torch.nn as nn
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys, os

sys.path.append(os.getcwd())
import matmul_leakyrelu_custom

torch.npu.config.allow_internal_format = False


class TestCustomMatmulLeakyrelu(TestCase):

    def test_matmul_leakyrelu_custom_ops(self):
        a = torch.rand([1024, 256], device='cpu', dtype=torch.float16).npu()
        b = torch.rand([256, 640], device='cpu', dtype=torch.float16).npu()
        bias = torch.randn([640], device='cpu', dtype=torch.float32).npu()

        output = matmul_leakyrelu_custom.run_matmul_leakyrelu_custom(a, b, bias)
        m = nn.LeakyReLU(0.001)
        cpuout = m(torch.matmul(a.cpu().type(output.dtype), b.cpu().type(output.dtype)) + bias.cpu())

        self.assertRtolEqual(output, cpuout)


if __name__ == "__main__":
    run_tests()
