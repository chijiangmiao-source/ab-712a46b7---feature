"""audit_robust（双增益工况鲁棒审计）测试。

固定场景覆盖：
* 两工况各自偏好的替代解（max 目标拒绝分别取局部最优）；
* 嵌套可兼容解（一套配对同时容纳两工况偏好）；
* 并列归属（两个对称最优点时的方案计数与必选/可选/从不分类）；
* 空候选、层级目标、字典序规范解、校验错误、任意精度与规模性能。
随机测试用暴力枚举交叉验证方案数、规范解、分类与 outside 计数。
"""

from __future__ import annotations

import itertools
import random

import pytest

from solver import MAX_CANDIDATES, MAX_HITS, MIN_HITS, ValidationError, audit_robust

from tests.brute import brute_solve_robust


def make_hits(n, prefix="h"):
    return [{"id": f"{prefix}{k}", "position": k * 10} for k in range(n)]


def cand(cid, left, right, r_low, r_high):
    return {
        "id": cid,
        "left_endpoint": left,
        "right_endpoint": right,
        "residual_low": r_low,
        "residual_high": r_high,
    }


def to_arc_records(candidates, hit_ids):
    pos = {h: k for k, h in enumerate(hit_ids)}
    return [
        (
            c["id"],
            pos[c["left_endpoint"]],
            pos[c["right_endpoint"]],
            c["residual_low"],
            c["residual_high"],
        )
        for c in candidates
    ]


# ---------------------------------------------------------------- 固定场景


def test_empty_candidates_single_empty_solution():
    res = audit_robust({"hits": make_hits(4), "candidates": []})
    assert res["optimal_count"] == "1"
    assert res["canonical_pairs"] == []
    assert res["unmatched_hits"] == ["h0", "h1", "h2", "h3"]
    assert res["classification"] == {"required": [], "optional": [], "never": []}
    assert res["paired_hits"] == 0
    assert res["total_residual_low"] == 0
    assert res["total_residual_high"] == 0


def test_two_conditions_prefer_different_alternatives():
    # 4 个击中上两个完美匹配（同对数、非交叉）：
    #   嵌套方案 [out, in]：工况一残差 2、工况二残差 100；
    #   并列方案 [p01, p23]：工况一残差 100、工况二残差 2。
    # 分别按工况取局部最优会得到两套互相冲突的解释；鲁棒目标最小化
    # max 后两者并列（max=100），再比 sum 仍并列，由 id 序列裁决。
    hits = make_hits(4)
    candidates = [
        cand("p01", "h0", "h1", 50, 1),
        cand("p23", "h2", "h3", 50, 1),
        cand("out", "h0", "h3", 1, 50),
        cand("in", "h1", "h2", 1, 50),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["paired_hits"] == 4
    # 两个方案 (2,100) 与 (100,2)：max=100、sum=102，二者相同。
    assert res["total_residual_low"] == 2
    assert res["total_residual_high"] == 100
    assert res["optimal_count"] == "2"
    # 左端顺序 id 序列：['in'... ] 嵌套首弧 id 'out' < 'p01'，嵌套方案胜出。
    assert [p["id"] for p in res["canonical_pairs"]] == ["out", "in"]
    assert res["unmatched_hits"] == []
    # 两条解释上的四条弧各出现一次 => 全部可选；没有必选/从不。
    assert res["classification"]["required"] == []
    assert set(res["classification"]["optional"]) == {"p01", "p23", "out", "in"}
    assert res["classification"]["never"] == []


def test_max_objective_rejects_cheating_single_condition():
    # 交叉低价诱饵的双工况版：bait 在工况一为 0 残差但与 inner 交叉，
    # 3 对方案只能 bait+inner+tail，工况二代价极高；真正鲁棒最优是
    # 顺序三对。
    hits = make_hits(6)
    candidates = [
        cand("bait", "h0", "h3", 0, 1000),
        cand("inner", "h1", "h2", 10, 1000),
        cand("tail", "h4", "h5", 1, 1),
        cand("seq01", "h0", "h1", 1, 1),
        cand("seq23", "h2", "h3", 1, 1),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["paired_hits"] == 6
    assert [p["id"] for p in res["canonical_pairs"]] == ["seq01", "seq23", "tail"]
    assert res["total_residual_low"] == 3
    assert res["total_residual_high"] == 3
    assert set(res["classification"]["never"]) == {"bait", "inner"}
    assert set(res["classification"]["required"]) == {"seq01", "seq23", "tail"}


def test_nested_compatible_solution_serves_both_conditions():
    # 嵌套可兼容解：外层弧在工况一便宜、内层弧在工况二便宜，二者可同选
    # （嵌套允许），组合方案在两工况残差都低，严格优于任何拆台方案。
    hits = make_hits(4)
    candidates = [
        cand("out", "h0", "h3", 1, 60),
        cand("in", "h1", "h2", 60, 1),
        cand("p01", "h0", "h1", 30, 30),
        cand("p23", "h2", "h3", 30, 30),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    # 嵌套 (61,61) => max 61；并列 (60,60) => max 60，并列方案更优。
    assert res["optimal_count"] == "1"
    assert [p["id"] for p in res["canonical_pairs"]] == ["p01", "p23"]
    assert res["total_residual_low"] == 60
    assert res["total_residual_high"] == 60
    assert set(res["classification"]["required"]) == {"p01", "p23"}
    assert set(res["classification"]["never"]) == {"out", "in"}

    # 反过来让嵌套组合严格占优：(1,1)+(...) 组合两工况都最低。
    candidates = [
        cand("out", "h0", "h3", 1, 5),
        cand("in", "h1", "h2", 5, 1),
        cand("p01", "h0", "h1", 10, 10),
        cand("p23", "h2", "h3", 10, 10),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["optimal_count"] == "1"
    assert [p["id"] for p in res["canonical_pairs"]] == ["out", "in"]
    assert (res["total_residual_low"], res["total_residual_high"]) == (6, 6)
    assert set(res["classification"]["required"]) == {"out", "in"}
    assert set(res["classification"]["never"]) == {"p01", "p23"}


def test_pairs_objective_outranks_residuals():
    # 1 对方案残差 (0,0)，2 对方案残差 (100,100)：配对数优先。
    hits = make_hits(4)
    candidates = [
        cand("cheap", "h0", "h1", 0, 0),
        cand("a", "h0", "h3", 50, 50),
        cand("b", "h1", "h2", 50, 50),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["paired_hits"] == 4
    assert [p["id"] for p in res["canonical_pairs"]] == ["a", "b"]
    assert (res["total_residual_low"], res["total_residual_high"]) == (100, 100)
    assert set(res["classification"]["required"]) == {"a", "b"}
    assert res["classification"]["never"] == ["cheap"]


def test_sum_breaks_max_tie():
    # 两方案 max 相同（都由工况二决定），但 sum 不同 => 选 (9,10) 而非 (10,10)。
    hits = make_hits(4)
    candidates = [
        # 方案 A（并列两弧）：(9,10)
        cand("a1", "h0", "h1", 4, 5),
        cand("a2", "h2", "h3", 5, 5),
        # 方案 B（嵌套两弧）：(10,10)
        cand("b1", "h0", "h3", 5, 5),
        cand("b2", "h1", "h2", 5, 5),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["optimal_count"] == "1"
    assert [p["id"] for p in res["canonical_pairs"]] == ["a1", "a2"]
    assert (res["total_residual_low"], res["total_residual_high"]) == (9, 10)
    assert set(res["classification"]["required"]) == {"a1", "a2"}
    assert set(res["classification"]["never"]) == {"b1", "b2"}


def test_tie_attribution_classification_with_shared_arc():
    # 两个并列最优方案共享一条必选弧，各带一条独有弧：
    #   方案 A=[shared, only_a]：(5,5)；方案 B=[shared, only_b]：(5,5)。
    # 5 个击中：shared=(0,4) 嵌套内部 (1,2)/(2,3) 互斥的两条。
    hits = make_hits(5)
    candidates = [
        cand("shared", "h0", "h4", 2, 2),
        cand("only_a", "h1", "h2", 3, 3),
        cand("only_b", "h2", "h3", 3, 3),
        cand("never_arc", "h1", "h3", 0, 100),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["optimal_count"] == "2"
    # 规范解 id 序列：方案 A 为 ['only_a','shared']... 按左端顺序是
    # ['shared','only_a']；与 ['shared','only_b'] 比较 only_a < only_b。
    assert [p["id"] for p in res["canonical_pairs"]] == ["shared", "only_a"]
    # shared 覆盖 0、4；only_a 覆盖 1、2；仅击中 3 未配对。
    assert res["unmatched_hits"] == ["h3"]
    assert res["classification"]["required"] == ["shared"]
    assert set(res["classification"]["optional"]) == {"only_a", "only_b"}
    assert res["classification"]["never"] == ["never_arc"]


def test_arbitrary_precision_count():
    # 与单残差版同构：每 3 个击中一组，组内三条候选弧两工况残差均相同，
    # 都为最优 1 对，组间独立 => 3^(n/3) 个方案。
    n = 180
    hits = make_hits(n)
    candidates = []
    for a in range(0, n, 3):
        candidates.append(cand(f"span{a}", f"h{a}", f"h{a+2}", 0, 0))
        candidates.append(cand(f"adj1{a}", f"h{a}", f"h{a+1}", 0, 0))
        candidates.append(cand(f"adj2{a}", f"h{a+1}", f"h{a+2}", 0, 0))
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["optimal_count"] == str(3 ** (n // 3))
    assert [p["id"] for p in res["canonical_pairs"]] == [
        f"adj1{a}" for a in range(0, n, 3)
    ]
    assert set(res["classification"]["optional"]) == {c["id"] for c in candidates}
    # 规范配对同时回显两项残差。
    assert all(
        p["residual_low"] == 0 and p["residual_high"] == 0
        for p in res["canonical_pairs"]
    )


def test_canonical_pairs_carry_both_residuals():
    hits = make_hits(4)
    candidates = [
        cand("q", "h0", "h3", 7, 9),
        cand("p", "h1", "h2", 3, 4),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    pair = {p["id"]: p for p in res["canonical_pairs"]}
    assert pair["q"]["residual_low"] == 7
    assert pair["q"]["residual_high"] == 9
    assert pair["p"]["residual_low"] == 3
    assert pair["p"]["residual_high"] == 4
    assert "residual" not in pair["q"]


# ---------------------------------------------------------------- 校验错误


def invalid_payload(payload):
    with pytest.raises(ValidationError) as exc:
        audit_robust(payload)
    return exc.value.errors


def test_missing_residual_fields():
    # 两项残差都缺失。
    errors = invalid_payload(
        {
            "hits": make_hits(4),
            "candidates": [{"id": "c", "left_endpoint": "h0", "right_endpoint": "h1"}],
        }
    )
    fields = {e["field"] for e in errors}
    assert "/candidates/0/residual_low" in fields
    assert "/candidates/0/residual_high" in fields


def test_one_residual_field_invalid():
    def candidates(**kw):
        base = {"id": "c", "left_endpoint": "h0", "right_endpoint": "h1"}
        base.update(kw)
        return [base]

    # 缺一项、另一项合法。
    errors = invalid_payload(
        {"hits": make_hits(4), "candidates": candidates(residual_low=1)}
    )
    assert any(e["field"] == "/candidates/0/residual_high" for e in errors)
    assert not any(e["field"] == "/candidates/0/residual_low" for e in errors)

    # 负数 / 布尔 / 浮点 / 字符串 均拒绝，路径指向具体字段。
    for field, val in [
        ("residual_low", -1),
        ("residual_high", True),
        ("residual_low", 1.5),
        ("residual_high", "3"),
    ]:
        kw = {"residual_low": 0, "residual_high": 0, field: val}
        errors = invalid_payload({"hits": make_hits(4), "candidates": candidates(**kw)})
        assert any(e["field"] == f"/candidates/0/{field}" for e in errors), (field, val)


def test_endpoint_and_scale_errors_with_field_paths():
    errors = invalid_payload(
        {"hits": make_hits(4), "candidates": [cand("c", "h0", "ghost", 0, 0)]}
    )
    assert any(e["field"] == "/candidates/0/right_endpoint" for e in errors)

    errors = invalid_payload(
        {"hits": make_hits(4), "candidates": [cand("c", "h2", "h1", 0, 0)]}
    )
    assert any(e["field"] == "/candidates/0/right_endpoint" for e in errors)

    errors = invalid_payload({"hits": make_hits(3), "candidates": []})
    assert any(e["field"] == "/hits" for e in errors)

    errors = invalid_payload(
        {
            "hits": make_hits(4),
            "candidates": [
                cand("a", "h0", "h1", 0, 0),
                cand("b", "h0", "h1", 1, 1),
            ],
        }
    )
    assert any(e["field"] == "/candidates/1" for e in errors)


def test_error_response_carries_no_audit_fields():
    errors = invalid_payload("not-an-object")
    assert errors[0]["field"] == ""
    assert set(errors[0].keys()) == {"field", "message"}


def test_legacy_residual_field_not_accepted_by_robust_entry():
    # 鲁棒入口只认两项新字段：仅给旧字段 residual 时两项新字段均非法。
    errors = invalid_payload(
        {
            "hits": make_hits(4),
            "candidates": [
                {
                    "id": "c",
                    "left_endpoint": "h0",
                    "right_endpoint": "h1",
                    "residual": 5,
                }
            ],
        }
    )
    fields = {e["field"] for e in errors}
    assert "/candidates/0/residual_low" in fields
    assert "/candidates/0/residual_high" in fields


# ---------------------------------------------------------------- 随机暴力对照


@pytest.mark.parametrize("seed", range(120))
def test_matches_bruteforce(seed):
    rng = random.Random(seed)
    n = rng.randint(MIN_HITS, 9)
    ids = [f"h{k}" for k in range(n)]

    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    candidates = []
    counter = itertools.count()
    for a, b in possible:
        if rng.random() < 0.4:
            candidates.append(
                cand(
                    f"cid{next(counter):03d}",
                    ids[a],
                    ids[b],
                    rng.choice([0, 0, 1, 2, 5]),
                    rng.choice([0, 0, 1, 2, 5]),
                )
            )

    res = audit_robust({"hits": make_hits(n), "candidates": candidates})
    ref = brute_solve_robust(n, to_arc_records(candidates, ids))

    assert int(res["optimal_count"]) == ref["optimal_count"]
    assert res["paired_hits"] == 2 * ref["max_pairs"]
    assert res["total_residual_low"] == ref["canonical_point"][0]
    assert res["total_residual_high"] == ref["canonical_point"][1]
    assert max(res["total_residual_low"], res["total_residual_high"]) == ref["min_max"]
    assert res["total_residual_low"] + res["total_residual_high"] == ref["min_sum"]
    assert [p["id"] for p in res["canonical_pairs"]] == ref["canonical"]

    hit_ids = [f"h{k}" for k in range(n)]
    assert res["unmatched_hits"] == [hit_ids[k] for k in ref["canonical_unmatched"]]

    cls = res["classification"]
    assert cls == ref["classification"]

    for cid, _a, _b, _r1, _r2 in to_arc_records(candidates, ids):
        used = ref["usage"][cid]
        if used == 0:
            assert cid in cls["never"]
        elif used == ref["optimal_count"]:
            assert cid in cls["required"]
        else:
            assert cid in cls["optional"]

    # 规范配对回显的两项残差必须与输入一致。
    input_res = {
        c["id"]: (c["residual_low"], c["residual_high"]) for c in candidates
    }
    for p in res["canonical_pairs"]:
        assert (p["residual_low"], p["residual_high"]) == input_res[p["id"]]


# ---------------------------------------------------------------- 性能


def test_max_scale_performance():
    n = MAX_HITS
    hits = make_hits(n)
    rng = random.Random(42)
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    candidates = [
        cand(f"c{k:04d}", f"h{a}", f"h{b}", rng.randrange(1000), rng.randrange(1000))
        for k, (a, b) in enumerate(possible[:MAX_CANDIDATES])
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert int(res["optimal_count"]) >= 1
    assert len(res["canonical_pairs"]) * 2 == res["paired_hits"]
