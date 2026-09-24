"""Compose verify 服务的单次复核入口。

依次执行：
1. 代码测试（pytest 单元测试，含随机暴力交叉验证）；
2. 构建检查（语法编译、模块导入、镜像内关键文件齐备）；
3. API/HTTP 冒烟：
   - 原入口 /audit 回归（嵌套同优、交叉低价诱饵、空候选、非法引用等）；
   - 鲁棒入口 /audit/robust（两工况各自偏好的替代解、均衡解压制极端解、
     嵌套可兼容解、并列归属、双残差字段非法、两入口字段语义隔离）。

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
        os.path.join("tests", "test_http.py"),
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


def rcand(cid, a, b, r1, r2):
    return {
        "id": cid,
        "left_endpoint": f"h{a}",
        "right_endpoint": f"h{b}",
        "residuals": [r1, r2],
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

    # ------------------------------------------------ 鲁棒入口 /audit/robust

    # R1：两工况各自偏好的替代解——嵌套解 (2,20) 与并列解 (20,2)
    # 在前三层目标下完全并列，由规范 id 序列裁决；四条候选均为可选。
    payload = {
        "hits": hits(4),
        "candidates": [
            rcand("r_out", 0, 3, 1, 10),
            rcand("r_in", 1, 2, 1, 10),
            rcand("r_p01", 0, 1, 10, 1),
            rcand("r_p23", 2, 3, 10, 1),
        ],
    }
    status, body = http_request("POST", "/audit/robust", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "2"
        and body["paired_hits"] == 4
        and body["max_residual_total"] == 20
        and body["residual_totals"] == [2, 20]
        and [p["id"] for p in body["canonical_pairs"]] == ["r_out", "r_in"]
        and set(body["classification"]["optional"])
        == {"r_out", "r_in", "r_p01", "r_p23"}
        and all(set(p) == {"id", "left_endpoint", "right_endpoint", "residuals"}
                for p in body["canonical_pairs"])
    )
    check(ok, "鲁棒: 两工况偏好替代解并列计数与规范裁决", f"HTTP {status} {body}")

    # R2：均衡解压制两工况极端解——(8,8) 的 max=8 严格小于极端 (2,20)。
    payload = {
        "hits": hits(4),
        "candidates": [
            rcand("e_out", 0, 3, 0, 10),
            rcand("e_in", 1, 2, 0, 10),
            rcand("e_b01", 0, 1, 4, 4),
            rcand("e_b23", 2, 3, 4, 4),
        ],
    }
    status, body = http_request("POST", "/audit/robust", payload)
    ok = (
        status == 200
        and body["residual_totals"] == [8, 8]
        and body["max_residual_total"] == 8
        and body["optimal_count"] == "1"
        and [p["id"] for p in body["canonical_pairs"]] == ["e_b01", "e_b23"]
        and body["classification"]["required"] == ["e_b01", "e_b23"]
        and set(body["classification"]["never"]) == {"e_in", "e_out"}
    )
    check(ok, "鲁棒: 均衡折中方案压制两工况局部最优", f"HTTP {status} {body}")

    # R3：嵌套可兼容解——同一套嵌套配对在两工况下都不差（弱支配并列解），
    # 工况2严格更优，故唯一最优，两条嵌套弧必选。
    payload = {
        "hits": hits(4),
        "candidates": [
            rcand("n_out", 0, 3, 3, 2),
            rcand("n_in", 1, 2, 1, 1),
            rcand("n_p01", 0, 1, 2, 5),
            rcand("n_p23", 2, 3, 2, 5),
        ],
    }
    status, body = http_request("POST", "/audit/robust", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "1"
        and body["residual_totals"] == [4, 3]
        and [p["id"] for p in body["canonical_pairs"]] == ["n_out", "n_in"]
        and body["classification"]["required"] == ["n_in", "n_out"]
        and set(body["classification"]["never"]) == {"n_p01", "n_p23"}
    )
    check(ok, "鲁棒: 嵌套可兼容解弱支配并列解", f"HTTP {status} {body}")

    # R4：并列归属——两个完美匹配共用公共弧，其余四条两两替代；
    # 公共弧必选、替代弧可选、争端点诱饵从不出现。
    payload = {
        "hits": hits(6),
        "candidates": [
            rcand("t_common", 0, 1, 1, 1),
            rcand("t_x", 2, 5, 2, 2),
            rcand("t_w", 3, 4, 0, 0),
            rcand("t_y", 2, 3, 2, 2),
            rcand("t_z", 4, 5, 0, 0),
            rcand("t_bad", 1, 2, 9, 9),
        ],
    }
    status, body = http_request("POST", "/audit/robust", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "2"
        and body["residual_totals"] == [3, 3]
        and body["classification"]["required"] == ["t_common"]
        and set(body["classification"]["optional"]) == {"t_w", "t_x", "t_y", "t_z"}
        and body["classification"]["never"] == ["t_bad"]
        and [p["id"] for p in body["canonical_pairs"]] == ["t_common", "t_x", "t_w"]
    )
    check(ok, "鲁棒: 并列归属必选/可选/从不", f"HTTP {status} {body}")

    # R5：双残差字段非法——缺项、负值给出字段路径错误且不夹带审计结果。
    payload = {
        "hits": hits(4),
        "candidates": [
            {"id": "bad", "left_endpoint": "h0", "right_endpoint": "h1",
             "residuals": [1]},
        ],
    }
    status, body = http_request("POST", "/audit/robust", payload)
    check(
        status == 400
        and any(e["field"] == "/candidates/0/residuals" for e in body["errors"])
        and "optimal_count" not in body,
        "鲁棒: 残差缺项返回字段路径错误", f"HTTP {status} {body}",
    )

    payload = {
        "hits": hits(4),
        "candidates": [
            {"id": "bad", "left_endpoint": "h0", "right_endpoint": "h9",
             "residuals": [1, -2]},
        ],
    }
    status, body = http_request("POST", "/audit/robust", payload)
    check(
        status == 400
        and any(e["field"] == "/candidates/0/residuals/1" for e in body["errors"])
        and any(e["field"] == "/candidates/0/right_endpoint" for e in body["errors"])
        and set(body.keys()) == {"errors"},
        "鲁棒: 负残差与非法端点并列报错且无审计字段", f"HTTP {status} {body}",
    )

    # R6：两入口字段语义隔离——residual/residuals 互不兼容。
    payload = {
        "hits": hits(4),
        "candidates": [
            {"id": "p", "left_endpoint": "h0", "right_endpoint": "h1",
             "residuals": [1, 2]},
        ],
    }
    status, body = http_request("POST", "/audit", payload)
    check(
        status == 400 and any(e["field"] == "/candidates/0/residual" for e in body["errors"]),
        "原入口拒绝 residuals 字段（语义不变）", f"HTTP {status} {body}",
    )
    payload = {
        "hits": hits(4),
        "candidates": [
            {"id": "p", "left_endpoint": "h0", "right_endpoint": "h1",
             "residual": 3},
        ],
    }
    status, body = http_request("POST", "/audit/robust", payload)
    check(
        status == 400 and any(e["field"] == "/candidates/0/residuals" for e in body["errors"]),
        "鲁棒入口拒绝单 residual 字段", f"HTTP {status} {body}",
    )

    # 未知路径返回 404。
    status, _ = http_request("GET", "/nope")
    check(status == 404, "未知路径返回 404", f"HTTP {status}")


def main() -> int:
    run_unit_tests()
    run_build_checks()
    run_smoke()

    print("\n=== 汇总 ===")
    if failures:
        print(f"失败 {len(failures)} 项: {failures}")
        return 1
    print("全部复核通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
