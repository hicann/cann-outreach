from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANNBOT = os.path.join(ROOT, "CANNBot")
ENV = os.environ.copy()
ENV["PYTHONPATH"] = ROOT + os.pathsep + ENV.get("PYTHONPATH", "")
if os.environ.get("ASCEND_HOME_PATH"):
    ENV["PATH"] = os.path.join(os.environ["ASCEND_HOME_PATH"], "bin") + os.pathsep + ENV.get("PATH", "")


@dataclass
class CaseRecord:
    op: str
    pathway: str
    case_id: str
    shape: str
    result: str
    max_diff: str
    mean_diff: str
    time_ms: str
    note: str


@dataclass
class OpRecord:
    name: str
    path: str
    kind: str
    pathway: str
    status: str
    num_cases: int
    num_pass: int
    num_fail: int
    note: str
    cases: List[CaseRecord] = field(default_factory=list)


def _run_sub(cmd: List[str], cwd: str, timeout: int = 120) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"
    except Exception as e:
        return 1, "", str(e)


def _filter(s: str) -> str:
    return re.sub(r".*Permission mismatch.*\n", "", s)


def test_abs_torch() -> OpRecord:
    op_dir = os.path.join(CANNBOT, "hpc_abs_verify", "operators", "abs")
    cfg = os.path.join(ROOT, "op_tester", "configs", "examples", "abs.yaml")
    so = os.path.join(op_dir, "build", "libabs_ops.so")
    shapes = [
        ("C1", "[[256]]"), ("C2", "[[512]]"), ("C3", "[[1024]]"), ("C4", "[[4096]]"),
        ("C5", "[[32768]]"), ("C6", "[[1000]]"), ("C7", "[[13]]"),
        ("C8", "[[512,512]]"), ("C9", "[[1024,1024]]"), ("C10", "[[128,128,128]]"),
        ("C11", "[[64,32,16,8]]"), ("C12", "[[65536]]"),
    ]
    rec = OpRecord("abs", op_dir, "elementwise", "PyTorch(.so)", "RUN", 0, 0, 0, "")
    if not os.path.isfile(so):
        rec.status = "BUILD_MISSING"; rec.note = "libabs_ops.so not found"
        return rec
    for cid, sh in shapes:
        rc, out, err = _run_sub(
            [sys.executable, "-m", "op_tester", cfg, "--no-auto", "--shape", sh, "-q", "--json", "/tmp/_abs_c.json"],
            cwd=op_dir, timeout=60)
        try:
            data = json.load(open("/tmp/_abs_c.json"))
            r = data[0]
            passed = bool(r["passed"])
            md = r.get("max_diff", float("inf"))
            mn = r.get("mean_diff", 0.0)
            tms = r.get("elapsed", 0.0) * 1000
            msg = r.get("msg", "")
        except Exception:
            passed = False; md = float("inf"); mn = 0.0; tms = 0.0
            msg = (err.strip().splitlines()[-1] if err.strip() else "no json")[:80]
        if not passed and not msg:
            msg = "vector core exception / wrong output"
        rec.cases.append(CaseRecord("abs", "PyTorch", cid, sh.strip("[]"), "PASS" if passed else "FAIL",
                                    f"{md:.3e}", f"{mn:.3e}", f"{tms:.2f}", msg))
        rec.num_cases += 1
        if passed: rec.num_pass += 1
        else: rec.num_fail += 1
    rec.status = "PASS" if rec.num_fail == 0 else "FAIL"
    rec.note = "kernel 触发 NPU vector core 异常 / 输出错误"
    _append_abs_bin_diag(rec)
    return rec


def _append_abs_bin_diag(rec: OpRecord):
    op_dir = os.path.join(CANNBOT, "hpc_abs_verify", "operators", "abs")
    build = os.path.join(op_dir, "build")
    exe = os.path.join(build, "abs")
    if not os.path.isfile(exe):
        return
    rc, out, err = _run_sub([exe], cwd=build, timeout=60)
    try:
        import numpy as _np
        ob = os.path.join(build, "output", "output.bin")
        ib = os.path.join(build, "input", "input_x.bin")
        if not (os.path.isfile(ob) and os.path.isfile(ib)):
            raise FileNotFoundError("bin files missing")
        o = _np.fromfile(ob, dtype=_np.float32)
        x = _np.fromfile(ib, dtype=_np.float32)
        g = _np.abs(x)
        d = _np.abs(o - g)
        mism = int(_np.sum(d > 1e-5))
        tot = len(g)
        md = float(d.max()) if tot else 0.0
        passed = mism == 0
        rec.cases.append(CaseRecord("abs", "Bin(exec)", "B1", "32768 (1D, 硬编码)",
                                    "PASS" if passed else "FAIL", f"{md:.3e}", "n/a", "n/a",
                                    f"run.sh OP_NAME=add 致无法运行; golden.py 误用 add; 可执行文件 {mism}/{tot} 元素错误(尾部未写)"))
        rec.num_cases += 1
        if passed: rec.num_pass += 1
        else: rec.num_fail += 1
    except Exception as e:
        rec.cases.append(CaseRecord("abs", "Bin(exec)", "B1", "32768", "FAIL", "n/a", "n/a", "n/a", f"bin diag error: {e}"))
        rec.num_cases += 1; rec.num_fail += 1


def test_sparse_gemm_bin() -> OpRecord:
    op_dir = os.path.join(CANNBOT, "BlueTrib_SparseGEMM")
    rec = OpRecord("sparse_gemm", op_dir, "matmul(sparse)", "Bin(exec)", "RUN", 0, 0, 0, "")
    rc, out, err = _run_sub(["bash", "run.sh"], cwd=op_dir, timeout=180)
    out_f = _filter(out)
    passed = rc == 0 and ("PASSED" in out_f or "Is close: True" in out_f)
    m = re.search(r"Max diff:\s*([0-9.eE+-]+)", out_f)
    md = m.group(1) if m else "n/a"
    m2 = re.search(r"Output shape:\s*\(([^)]+)\)", out_f)
    shape = m2.group(1) if m2 else "(2,128,128)"
    note = "2:4 稀疏 GEMM，bin 通路单 shape"
    if not passed:
        note = (out_f.strip().splitlines()[-1] if out_f.strip() else "run failed")[:80]
    rec.cases.append(CaseRecord("sparse_gemm", "Bin", "C1", shape, "PASS" if passed else "FAIL", md, md, "n/a", note))
    rec.num_cases = 1
    if passed: rec.num_pass = 1; rec.status = "PASS"
    else: rec.num_fail = 1; rec.status = "FAIL"
    return rec


def record_skip(name: str, subpath: str, kind: str, reason: str) -> OpRecord:
    return OpRecord(name, os.path.join(CANNBOT, subpath) if subpath else "", kind, "-", "SKIP", 0, 0, 0, reason)


def gather_all() -> List[OpRecord]:
    recs: List[OpRecord] = []
    recs.append(test_abs_torch())
    recs.append(test_sparse_gemm_bin())

    recs.append(record_skip("add (triton)", "yw_add_verigy/add_verify", "elementwise",
                            "triton 未安装，无法运行 ModelNew triton kernel"))
    recs.append(record_skip("softmax (tilelang)", "hedi0515_tilelang_softmax", "reduction",
                            "tilelang 未安装，无法运行 tilelang softmax kernel"))
    recs.append(record_skip("mamba_selective_scan", "abyss_triton_mamba/op_mamba_selective_scan_20260523_1146_5999",
                            "scan", "仅含参考 Model，无优化 kernel 实现 (ModelNew 缺失)"))
    recs.append(record_skip("tril", "KJjk178_tril", "elementwise",
                            "仅有 op_host/op_kernel .cpp，无 CMakeLists/构建系统/torch 接入"))
    recs.append(record_skip("softmax (cxy_doc)", "cxy_doc/softmax", "reduction", "仅设计文档 (DESIGN.md/PLAN.md)"))
    recs.append(record_skip("online_safe_softmax", "qq_51242627_online_safe_softmax", "reduction", "仅设计文档"))
    recs.append(record_skip("test_abs (espere)", "espere_test_abs", "elementwise", "仅设计文档"))
    recs.append(record_skip("FusedSGDWithWeightDecay", "yctco_tuhuaneng_FusedSGDWithWeightDecay", "elementwise", "仅 README/文档"))
    recs.append(record_skip("gather_pa_kv_cache", "whz_codereview/gather_pa_kv_cache", "elementwise", "仅代码审查总结文档"))
    recs.append(record_skip("gemm_tiling_design", "zhoubaojun_gemm_tiling_design", "matmul", "仅设计文档"))
    recs.append(record_skip("精度定位体验", "dugutianxue_222_精度定位体验", "-", "体验文档"))
    recs.append(record_skip("A2_vs_A3_Arch", "gcw_GkB7RJlT_Atlas_A2_vs_A3_NPU_Arch", "-", "架构对比文档"))
    recs.append(record_skip("environment_report", "hackerdl-environment-report", "-", "环境报告文档"))
    recs.append(record_skip("abs_api_summary", "junmoumou_abs_api_summary.md", "-", "API 总结文档"))
    recs.append(record_skip("算子调优指南", "Shareable_KADC_test", "-", "调优指南文档"))
    recs.append(record_skip("envcheck", "ysws_envcheck", "-", "环境检查文档"))
    recs.append(record_skip("demo/code-review", "demo/code-review", "-", "代码审查 demo"))
    return recs


def write_reports(recs: List[OpRecord], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "report.json")
    csv_path = os.path.join(out_dir, "report.csv")
    md_path = os.path.join(out_dir, "report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{"op": r.name, "path": r.path, "kind": r.kind, "pathway": r.pathway,
                    "status": r.status, "num_cases": r.num_cases, "num_pass": r.num_pass,
                    "num_fail": r.num_fail, "note": r.note,
                    "cases": [asdict(c) for c in r.cases]} for r in recs], f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "pathway", "case_id", "shape", "result", "max_diff", "mean_diff", "time_ms", "note"])
        for r in recs:
            if r.cases:
                for c in r.cases:
                    w.writerow([c.op, c.pathway, c.case_id, c.shape, c.result, c.max_diff, c.mean_diff, c.time_ms, c.note])
            else:
                w.writerow([r.name, r.pathway, "-", "-", r.status, "-", "-", "-", r.note])

    total = len(recs)
    tested = [r for r in recs if r.status in ("PASS", "FAIL")]
    passed = [r for r in recs if r.status == "PASS"]
    failed = [r for r in recs if r.status == "FAIL"]
    skipped = [r for r in recs if r.status == "SKIP"]
    tot_cases = sum(r.num_cases for r in recs)
    tot_pass = sum(r.num_pass for r in recs)
    tot_fail = sum(r.num_fail for r in recs)

    lines = []
    lines.append("# CANNBot 算子批量测试报告\n")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  工具: op_tester  |  设备: NPU (Atlas)\n")
    lines.append("\n## 1. 总览\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 算子项目总数 | {total} |")
    lines.append(f"| 已测试(可运行) | {len(tested)} |")
    lines.append(f"| 通过 | {len(passed)} |")
    lines.append(f"| 失败 | {len(failed)} |")
    lines.append(f"| 跳过(不可测) | {len(skipped)} |")
    lines.append(f"| 测试用例总数 | {tot_cases} |")
    lines.append(f"| 用例通过 | {tot_pass} |")
    lines.append(f"| 用例失败 | {tot_fail} |")
    lines.append(f"| 用例通过率 | {tot_pass/tot_cases*100:.1f}% |\n" if tot_cases else "| 用例通过率 | n/a |\n")

    lines.append("## 2. 算子级汇总\n")
    lines.append("| 算子 | 种类 | 通路 | 状态 | 用例数 | 通过 | 失败 | 说明 |")
    lines.append("|------|------|------|------|--------|------|------|------|")
    for r in recs:
        st = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "SKIP": "⏭ SKIP"}.get(r.status, r.status)
        lines.append(f"| {r.name} | {r.kind} | {r.pathway} | {st} | {r.num_cases} | {r.num_pass} | {r.num_fail} | {r.note} |")
    lines.append("")

    lines.append("## 3. 详细测试用例\n")
    for r in recs:
        lines.append(f"### {r.name}  ({r.pathway})  — {r.status}")
        lines.append(f"- 路径: `{r.path}`")
        lines.append(f"- 说明: {r.note}")
        if r.cases:
            lines.append("")
            lines.append("| 用例 | shape | 结果 | max_diff | mean_diff | 耗时(ms) | 备注 |")
            lines.append("|------|-------|------|----------|-----------|----------|------|")
            for c in r.cases:
                res = "✅" if c.result == "PASS" else "❌"
                lines.append(f"| {c.case_id} | {c.shape} | {res} {c.result} | {c.max_diff} | {c.mean_diff} | {c.time_ms} | {c.note} |")
        lines.append("")

    lines.append("## 4. 跳过项目原因\n")
    lines.append("| 算子 | 原因 |")
    lines.append("|------|------|")
    for r in skipped:
        lines.append(f"| {r.name} | {r.note} |")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path, csv_path, json_path


def main():
    print("=== CANNBot 算子批量测试 ===")
    recs = gather_all()
    out_dir = os.path.join(ROOT, "op_tester", "report")
    md, csvp, js = write_reports(recs, out_dir)
    print(f"\n报告已生成:")
    print(f"  Markdown: {md}")
    print(f"  CSV:      {csvp}")
    print(f"  JSON:     {js}")
    tot = sum(r.num_cases for r in recs)
    tp = sum(r.num_pass for r in recs)
    tf = sum(r.num_fail for r in recs)
    print(f"\n用例总计: {tot}  通过: {tp}  失败: {tf}")


if __name__ == "__main__":
    main()
