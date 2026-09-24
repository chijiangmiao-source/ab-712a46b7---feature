"""Compose verify 服务的单次复核入口。

依次执行：
1. 代码测试（pytest 单元测试，含随机暴力交叉验证）；
2. 构建检查（语法编译、模块导入、镜像内关键文件齐备）；
3. API/HTTP 冒烟：
   * 原入口 /audit 回归（健康路径、嵌套同优、交叉低价诱饵、空候选、
     非法引用、重复端点对、位置冲突、规模越界、未知路径）；
   * 鲁棒入口 /audit/robust（两工况各自偏好的替代解、嵌套可兼容解、
     并列归属、双残差字段校验）。

任一步失败即以非零退出码结束，全部通过退出码 0。
"""

from __future__ import annotations

import json
import os
import py_compile
import sys
import urllib.error
import urllib.request

API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
REQUIRE_FILES = os.environ.get("REQUIRE_BUILD_FILES", "").split(",")

failures: list[str] = []


def check(condition: bool, name: str, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}{(' - ' + detail) if detail and not condition else ''}")
    if not condition:
        failures.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- 1. 代码测试


def run_unit_tests() -> None:
    section("代码测试 (pytest)")
    import pytest

    rc = pytest.main(["-q", "tests"])
    check(rc == 0, "pytest 单元测试", f"退出码 {rc}")


# ---------------------------------------------------------------- 2. 构建检查


def run_build_checks() -> None:
    section("构建检查")
    for path in [
        "solver.py",
        "app.py",
        "verify.py",
        os.path.join("tests", "test_solver.py"),
        os.path.join("tests", "test_solver_robust.py"),
    ]:
        try:
            py_compile.compile(path, doraise=True)
            ok = True
        except py_compile.PyCompileError as exc:
            ok = False
            print(exc)
        check(ok, f"语法编译: {path}")

    for fname in REQUIRE_FILES:
        fname = fname.strip()
        if not fname:
            continue
        check(os.path.isfile(fname), f"镜像内文件齐备: {fname}")

    try:
        import solver  # noqa: F401
        import app  # noqa: F401

        ok = True
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(exc)
    check(ok, "模块导入 solver/app")


# ---------------------------------------------------------------- 3. API/HTTP 冒烟


def http_request(method: str, path: str, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        API_BASE + path, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def hits(n):
    return [{"id": f"h{k}", "position": k * 10} for k in range(n)]


def cand(cid, a, b, r):
    return {
        "id": cid,
        "left_endpoint": f"h{a}",
        "right_endpoint": f"h{b}",
        "residual": r,
    }


def cand2(cid, a, b, r_low, r_high):
    return {
        "id": cid,
        "left_endpoint": f"h{a}",
        "right_endpoint": f"h{b}",
        "residual_low": r_low,
        "residual_high": r_high,
    }


def run_smoke() -> None:
    section(f"API/HTTP 冒烟 ({API_BASE})")

    status, body = http_request("GET", "/health")
    check(status == 200 and body.get("status") == "ready", "健康路径 /health 报告就绪",
          f"HTTP {status} {body}")

    # 场景 1：嵌套同优 —— 顺序配对与嵌套配对同为 2 对同残差。
    payload = {
        "hits": hits(4),
        "candidates": [
            cand("a_out", 0, 3, 1),
            cand("a_in", 1, 2, 5),
            cand("b_left", 0, 1, 3),
            cand("b_right", 2, 3, 3),
        ],
    }
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "2"
        and [p["id"] for p in body["canonical_pairs"]] == ["a_out", "a_in"]
        and set(body["classification"]["optional"])
        == {"a_out", "a_in", "b_left", "b_right"}
    )
    check(ok, "嵌套同优方案并列计数与规范解", f"HTTP {status} {body}")

    # 场景 2：交叉低价诱饵 —— 0 残差的交叉弧不得胜出。
    payload = {
        "hits": hits(6),
        "candidates": [
            cand("bait", 0, 3, 0),
            cand("inner", 1, 2, 10),
            cand("tail", 4, 5, 1),
            cand("seq01", 0, 1, 1),
            cand("seq23", 2, 3, 1),
        ],
    }
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 200
        and [p["id"] for p in body["canonical_pairs"]] == ["seq01", "seq23", "tail"]
        and "bait" in body["classification"]["never"]
        and body["total_residual"] == 3
    )
    check(ok, "交叉低价诱饵不被接受", f"HTTP {status} {body}")

    # 场景 3：合法空候选 —— 唯一空方案。
    payload = {"hits": hits(4), "candidates": []}
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "1"
        and body["canonical_pairs"] == []
        and len(body["unmatched_hits"]) == 4
        and body["classification"] == {"required": [], "optional": [], "never": []}
    )
    check(ok, "合法空候选返回唯一空方案", f"HTTP {status} {body}")

    # 场景 4：非法引用 —— 错误带字段路径，且不夹带任何审计字段。
    payload = {"hits": hits(4), "candidates": [cand("bad", 0, 99, 0)]}
    # 99 不在 id 中，手工构造以模拟未知端点字符串。
    payload["candidates"][0]["right_endpoint"] = "h99"
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 400
        and isinstance(body.get("errors"), list)
        and any(e.get("field") == "/candidates/0/right_endpoint" for e in body["errors"])
        and "optimal_count" not in body
    )
    check(ok, "非法引用返回字段路径错误且无审计结果", f"HTTP {status} {body}")

    # 附加：重复端点对、位置冲突、规模越界。
    status, body = http_request(
        "POST",
        "/audit",
        {"hits": hits(4), "candidates": [cand("a", 0, 1, 0), cand("b", 0, 1, 1)]},
    )
    check(
        status == 400
        and any(e["field"] == "/candidates/1" for e in body["errors"])
        and "optimal_count" not in body,
        "重复端点对被拒绝", f"HTTP {status} {body}",
    )

    conflict = hits(4)
    conflict[2]["position"] = conflict[1]["position"]
    status, body = http_request("POST", "/audit", {"hits": conflict, "candidates": []})
    check(
        status == 400
        and any(e["field"] == "/hits/2/position" for e in body["errors"]),
        "位置冲突被拒绝", f"HTTP {status} {body}",
    )

    status, body = http_request("POST", "/audit", {"hits": hits(3), "candidates": []})
    check(
        status == 400 and any(e["field"] == "/hits" for e in body["errors"]),
        "击中规模越界被拒绝", f"HTTP {status} {body}",
    )

    # 未知路径返回 404。
    status, _ = http_request("GET", "/nope")
    check(status == 404, "未知路径返回 404", f"HTTP {status}")


def run_smoke_robust() -> None:
    """鲁棒入口 /audit/robust 的验收场景。"""
    section(f"鲁棒审计冒烟 POST /audit/robust ({API_BASE})")

    # 场景 R1：两工况各自偏好的替代解。
    # 嵌套 [out,in] = (2,100)，并列 [p01,p23] = (100,2)：分别取局部最优会
    # 得到两套冲突解释；max 目标下二者并列，计数 2，id 序列裁决规范解。
    payload = {
        "hits": hits(4),
        "candidates": [
            cand2("p01", 0, 1, 50, 1),
            cand2("p23", 2, 3, 50, 1),
            cand2("out", 0, 3, 1, 50),
            cand2("in", 1, 2, 1, 50),
        ],
    }
    status, body = http_request("POST", "/audit/robust", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "2"
        and (body["total_residual_low"], body["total_residual_high"]) == (2, 100)
        and [p["id"] for p in body["canonical_pairs"]] == ["out", "in"]
        and set(body["classification"]["optional"])
        == {"p01", "p23", "out", "in"}
        and body["classification"]["required"] == []
    )
    check(ok, "两工况各自偏好的替代解：max 并列计数与规范解", f"HTTP {status} {body}")

    # 场景 R2：嵌套可兼容解 —— 外弧工况一便宜、内弧工况二便宜且可同选，
    # 组合 (6,6) 的 max=6 严格优于并列 (20,20) 的 max=20。
    payload = {
        "hits": hits(4),
        "candidates": [
            cand2("out", 0, 3, 1, 5),
            cand2("in", 1, 2, 5, 1),
            cand2("p01", 0, 1, 10, 10),
            cand2("p23", 2, 3, 10, 10),
        ],
    }
    status, body = http_request("POST", "/audit/robust", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "1"
        and [p["id"] for p in body["canonical_pairs"]] == ["out", "in"]
        and (body["total_residual_low"], body["total_residual_high"]) == (6, 6)
        and set(body["classification"]["required"]) == {"out", "in"}
        and set(body["classification"]["never"]) == {"p01", "p23"}
    )
    check(ok, "嵌套可兼容解：一套配对同时满足两工况", f"HTTP {status} {body}")

    # 场景 R3：并列归属 —— 两最优方案共享必选弧 shared，各带一条独有弧，
    # 另有一条从不出现的弧。
    payload = {
        "hits": hits(5),
        "candidates": [
            cand2("shared", 0, 4, 2, 2),
            cand2("only_a", 1, 2, 3, 3),
            cand2("only_b", 2, 3, 3, 3),
            cand2("never_arc", 1, 3, 0, 100),
        ],
    }
    status, body = http_request("POST", "/audit/robust", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "2"
        and [p["id"] for p in body["canonical_pairs"]] == ["shared", "only_a"]
        and body["unmatched_hits"] == ["h3"]
        and body["classification"]["required"] == ["shared"]
        and set(body["classification"]["optional"]) == {"only_a", "only_b"}
        and body["classification"]["never"] == ["never_arc"]
    )
    check(ok, "并列归属：必选/可选/从不分类", f"HTTP {status} {body}")

    # 场景 R4：空候选唯一空方案。
    status, body = http_request(
        "POST", "/audit/robust", {"hits": hits(4), "candidates": []}
    )
    ok = (
        status == 200
        and body["optimal_count"] == "1"
        and body["canonical_pairs"] == []
        and len(body["unmatched_hits"]) == 4
        and body["total_residual_low"] == 0
        and body["total_residual_high"] == 0
        and body["classification"] == {"required": [], "optional": [], "never": []}
    )
    check(ok, "鲁棒入口空候选返回唯一空方案", f"HTTP {status} {body}")

    # 场景 R5：双残差字段非法（缺字段、负数、未知端点）带字段路径，
    # 且不夹带审计字段。
    status, body = http_request(
        "POST",
        "/audit/robust",
        {
            "hits": hits(4),
            "candidates": [
                {"id": "x", "left_endpoint": "h0", "right_endpoint": "h1", "residual": 0}
            ],
        },
    )
    ok = (
        status == 400
        and {e["field"] for e in body["errors"]}
        == {"/candidates/0/residual_low", "/candidates/0/residual_high"}
        and "optimal_count" not in body
    )
    check(ok, "缺双残差字段返回字段路径错误且无审计结果", f"HTTP {status} {body}")

    status, body = http_request(
        "POST",
        "/audit/robust",
        {"hits": hits(4), "candidates": [cand2("x", 0, 1, -2, 3)]},
    )
    ok = (
        status == 400
        and any(e["field"] == "/candidates/0/residual_low" for e in body["errors"])
        and "optimal_count" not in body
    )
    check(ok, "负残差被拒绝", f"HTTP {status} {body}")

    payload_bad = {"hits": hits(4), "candidates": [cand2("x", 0, 1, 0, 0)]}
    payload_bad["candidates"][0]["right_endpoint"] = "h99"
    status, body = http_request("POST", "/audit/robust", payload_bad)
    ok = (
        status == 400
        and any(e["field"] == "/candidates/0/right_endpoint" for e in body["errors"])
        and "optimal_count" not in body
    )
    check(ok, "鲁棒入口非法引用被拒绝", f"HTTP {status} {body}")


def main() -> int:
    run_unit_tests()
    run_build_checks()
    run_smoke()
    run_smoke_robust()

    print("\n=== 汇总 ===")
    if failures:
        print(f"失败 {len(failures)} 项: {failures}")
        return 1
    print("全部复核通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
