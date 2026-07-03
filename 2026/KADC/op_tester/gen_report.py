from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Case:
    case_id: str
    shape: str
    result: str
    max_diff: str
    mean_diff: str
    time_ms: str
    note: str


@dataclass
class Op:
    name: str
    kind: str
    pathway: str
    dtype: str
    cases: List[Case] = field(default_factory=list)

    @property
    def num(self): return len(self.cases)
    @property
    def npass(self): return sum(1 for c in self.cases if c.result == "PASS")
    @property
    def nfail(self): return sum(1 for c in self.cases if c.result == "FAIL")


def P(cid, shape, md, mn, t, note=""): return Case(cid, shape, "PASS", md, mn, t, note)
def F(cid, shape, md, mn, t, note): return Case(cid, shape, "FAIL", md, mn, t, note)


def build_ops() -> List[Op]:
    ops: List[Op] = []

    abs_ = Op("abs", "elementwise", "PyTorch(.so)", "fp32")
    abs_.cases = [
        P("C1", "[256]", "0.000e+00", "0.000e+00", "0.18"),
        P("C2", "[512]", "0.000e+00", "0.000e+00", "0.16"),
        P("C3", "[1024]", "0.000e+00", "0.000e+00", "0.17"),
        P("C4", "[4096]", "0.000e+00", "0.000e+00", "0.19"),
        P("C5", "[32768]", "0.000e+00", "0.000e+00", "0.24"),
        P("C6", "[1000]", "0.000e+00", "0.000e+00", "0.17", "非对齐"),
        P("C7", "[13]", "0.000e+00", "0.000e+00", "0.15", "极小 shape"),
        P("C8", "[512,512]", "0.000e+00", "0.000e+00", "0.22"),
        P("C9", "[1024,1024]", "0.000e+00", "0.000e+00", "0.28"),
        P("C10", "[128,128,128]", "0.000e+00", "0.000e+00", "0.31"),
        P("C11", "[64,32,16,8]", "0.000e+00", "0.000e+00", "0.20", "4D"),
        P("C12", "[65536]", "0.000e+00", "0.000e+00", "0.35"),
    ]
    ops.append(abs_)

    add = Op("add", "elementwise", "PyTorch(triton)", "fp16")
    add.cases = [
        P("C1", "[1024,1024]", "0.000e+00", "0.000e+00", "0.42"),
        P("C2", "[8192]", "0.000e+00", "0.000e+00", "0.19"),
        P("C3", "[256,256,256]", "0.000e+00", "0.000e+00", "0.38", "3D"),
        P("C4", "[4096,4096]", "0.000e+00", "0.000e+00", "1.12"),
        P("C5", "[128,128]", "0.000e+00", "0.000e+00", "0.14"),
        P("C6", "[65536]", "0.000e+00", "0.000e+00", "0.21"),
        F("C7", "[10000]", "1.953e-03", "4.882e-04", "0.18", "fp16 非对齐尾部精度"),
    ]
    ops.append(add)

    softmax = Op("softmax", "reduction", "PyTorch(.so)", "fp32")
    softmax.cases = [
        P("C1", "[128,128] dim=-1", "1.192e-07", "2.980e-08", "0.33"),
        P("C2", "[16,256,32] dim=1", "1.490e-07", "3.725e-08", "0.41", "中间维归约"),
        P("C3", "[1024,1024] dim=-1", "2.384e-07", "5.960e-08", "0.88"),
        P("C4", "[256] dim=0", "5.960e-08", "1.490e-08", "0.12", "1D"),
        P("C5", "[64,64,64] dim=1", "1.788e-07", "4.470e-08", "0.46"),
        P("C6", "[2048,512] dim=-1", "2.682e-07", "6.705e-08", "1.34"),
    ]
    ops.append(softmax)

    sg = Op("sparse_gemm", "matmul(2:4 sparse)", "Bin(exec)", "fp32")
    sg.cases = [
        P("C1", "[128,128]x[128,128]", "0.000e+00", "0.000e+00", "0.74"),
        P("C2", "[2,128,128] batched", "0.000e+00", "0.000e+00", "1.18", "batched"),
        P("C3", "[256,256]x[256,256]", "0.000e+00", "0.000e+00", "2.06"),
        P("C4", "[64,64]x[64,64]", "0.000e+00", "0.000e+00", "0.31"),
        P("C5", "[512,512]x[512,512]", "0.000e+00", "0.000e+00", "4.87"),
    ]
    ops.append(sg)

    tril = Op("tril", "elementwise", "PyTorch(.so)", "fp32")
    tril.cases = [
        P("C1", "[64,64] diag=0", "0.000e+00", "0.000e+00", "0.11"),
        P("C2", "[128,128] diag=-1", "0.000e+00", "0.000e+00", "0.16", "严格下三角"),
        P("C3", "[4,32,32] diag=1", "0.000e+00", "0.000e+00", "0.14", "batched"),
        P("C4", "[256,256] diag=0", "0.000e+00", "0.000e+00", "0.28"),
        P("C5", "[512,512] diag=2", "0.000e+00", "0.000e+00", "0.52"),
        P("C6", "[1024,1024] diag=0", "0.000e+00", "0.000e+00", "0.91"),
    ]
    ops.append(tril)

    mamba = Op("mamba_selective_scan", "scan", "PyTorch(.so)", "fp32")
    mamba.cases = [
        P("C1", "[2,128,32] d_state=8", "8.545e-05", "2.137e-05", "3.82"),
        P("C2", "[4,256,32] d_state=8", "1.526e-04", "3.815e-05", "11.46"),
        P("C3", "[2,512,32] d_state=8", "2.289e-04", "5.722e-05", "21.08", "长序列"),
        P("C4", "[1,64,32] d_state=8", "4.768e-05", "1.192e-05", "1.23", "小 batch"),
        F("C5", "[8,1024,32] d_state=8", "7.629e-04", "1.907e-04", "48.71", "大 batch+长序列 累积误差超容差"),
    ]
    ops.append(mamba)

    sumop = Op("sum", "reduction", "PyTorch(.so)", "fp32")
    sumop.cases = [
        P("C1", "[1024] dim=0", "0.000e+00", "0.000e+00", "0.09"),
        P("C2", "[128,128] dim=1", "1.907e-06", "4.768e-07", "0.21"),
        P("C3", "[16,32,64] dim=1 keepdim", "3.815e-06", "9.537e-07", "0.34"),
        P("C4", "[2048,512] dim=-1", "7.629e-06", "1.907e-06", "0.95"),
    ]
    ops.append(sumop)

    mm = Op("matmul", "matmul", "PyTorch(.so)", "fp16")
    mm.cases = [
        P("C1", "[128,256]x[256,512]", "3.125e-02", "7.812e-03", "0.58", "fp16"),
        P("C2", "[1024,1024]x[1024,1024]", "6.250e-02", "1.562e-02", "4.12"),
        P("C3", "[512,512]x[512,512]", "4.687e-02", "1.172e-02", "1.36"),
        P("C4", "[64,64]x[64,64]", "1.562e-02", "3.906e-03", "0.12"),
    ]
    ops.append(mm)

    return ops


def write(ops: List[Op], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump([{"op": o.name, "kind": o.kind, "pathway": o.pathway, "dtype": o.dtype,
                    "num": o.num, "pass": o.npass, "fail": o.nfail,
                    "cases": [asdict(c) for c in o.cases]} for o in ops], f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "report.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "kind", "pathway", "dtype", "case_id", "shape", "result", "max_diff", "mean_diff", "time_ms", "note"])
        for o in ops:
            for c in o.cases:
                w.writerow([o.name, o.kind, o.pathway, o.dtype, c.case_id, c.shape, c.result, c.max_diff, c.mean_diff, c.time_ms, c.note])

    tot = sum(o.num for o in ops)
    tp = sum(o.npass for o in ops)
    tf = sum(o.nfail for o in ops)
    L = []
    L.append("# CANNBot 算子测试报告（模拟数据）\n")
    L.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  工具: op_tester  |  数据来源: 模拟（未实跑）\n")
    L.append("\n## 1. 总览\n")
    L.append("| 指标 | 数值 |")
    L.append("|------|------|")
    L.append(f"| 算子数 | {len(ops)} |")
    L.append(f"| 测试用例总数 | {tot} |")
    L.append(f"| 通过 | {tp} |")
    L.append(f"| 失败 | {tf} |")
    L.append(f"| 通过率 | {tp/tot*100:.1f}% |\n")

    L.append("## 2. 算子级汇总\n")
    L.append("| 算子 | 种类 | 通路 | dtype | 用例数 | 通过 | 失败 | 状态 |")
    L.append("|------|------|------|-------|--------|------|------|------|")
    for o in ops:
        st = "✅ PASS" if o.nfail == 0 else "⚠ PARTIAL"
        L.append(f"| {o.name} | {o.kind} | {o.pathway} | {o.dtype} | {o.num} | {o.npass} | {o.nfail} | {st} |")
    L.append("")

    L.append("## 3. 详细测试用例\n")
    for o in ops:
        L.append(f"### {o.name}  ({o.kind}, {o.dtype})")
        L.append("")
        L.append("| 用例 | shape | 结果 | max_diff | mean_diff | 耗时(ms) | 备注 |")
        L.append("|------|-------|------|----------|-----------|----------|------|")
        for c in o.cases:
            r = "✅ PASS" if c.result == "PASS" else "❌ FAIL"
            L.append(f"| {c.case_id} | {c.shape} | {r} | {c.max_diff} | {c.mean_diff} | {c.time_ms} | {c.note} |")
        L.append("")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ops = build_ops()
    out = os.path.join(ROOT, "op_tester", "report")
    write(ops, out)
    tot = sum(o.num for o in ops)
    tp = sum(o.npass for o in ops)
    tf = sum(o.nfail for o in ops)
    print(f"报告已生成于 {out}/  (report.md, report.csv, report.json)")
    print(f"算子: {len(ops)}  用例: {tot}  通过: {tp}  失败: {tf}  通过率: {tp/tot*100:.1f}%")


if __name__ == "__main__":
    main()
