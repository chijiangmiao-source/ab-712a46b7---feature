"""audit_robust（双增益工况鲁棒审计）的单元测试。

覆盖：
* 两工况各自偏好不同替代解时，鲁棒目标选择折中方案；
* 嵌套可兼容解（同一套配对同时服务两工况）；
* 并列归属（required / optional / never）与任意精度计数；
* 双残差字段、端点引用、规模非法时的字段路径错误（且不混入审计结果）；
* 随机暴力枚举交叉验证与最大规模性能。
"""

from __future__ import annotations

import itertools
import random

import pytest

from solver import MAX_CANDIDATES, MAX_HITS, MIN_HITS, ValidationError, audit_robust

from tests.brute import brute_solve_robust


def make_hits(n, prefix="h"):
    return [{"id": f"{prefix}{k}", "position": k * 10} for k in range(n)]


def rcand(cid, left, right, r1, r2):
    return {
        "id": cid,
        "left_endpoint": left,
        "right_endpoint": right,
        "residuals": [r1, r2],
    }


def to_arc_records(candidates, hit_ids):
    pos = {h: k for k, h in enumerate(hit_ids)}
    return [
        (
            c["id"],
            pos[c["left_endpoint"]],
            pos[c["right_endpoint"]],
            c["residuals"][0],
            c["residuals"][1],
        )
        for c in candidates
    ]


# ---------------------------------------------------------------- 固定场景


def test_empty_candidates_single_empty_solution():
    n = MIN_HITS
    res = audit_robust({"hits": make_hits(n), "candidates": []})
    assert res["optimal_count"] == "1"
    assert res["paired_hits"] == 0
    assert res["residual_totals"] == [0, 0]
    assert res["max_residual_total"] == 0
    assert res["canonical_pairs"] == []
    assert res["unmatched_hits"] == [f"h{k}" for k in range(n)]
    assert res["classification"] == {"required": [], "optional": [], "never": []}


def test_two_conditions_prefer_different_alternatives():
    # 两个完美匹配各服务一个工况：
    #   嵌套 (out,in)：工况1便宜(2)、工况2昂贵(20)；
    #   并列 (p01,p23)：工况1昂贵(20)、工况2便宜(2)。
    # 单工况会各取所好；鲁棒目标下两者 max 均为 20、sum 均为 22，
    # 任一单工况局部最优都不支配对方，只能由同一套配对并列承担。
    hits = make_hits(4)
    candidates = [
        rcand("out", "h0", "h3", 1, 10),
        rcand("in", "h1", "h2", 1, 10),
        rcand("p01", "h0", "h1", 10, 1),
        rcand("p23", "h2", "h3", 10, 1),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["paired_hits"] == 4
    # 两方案都是 max=20、sum=22，前三层并列。
    assert res["max_residual_total"] == 20
    assert res["optimal_count"] == "2"
    # 规范解由 id 序列裁决：左端顺序序列 [out, in] 与 [p01, p23]
    # 比较，'out' < 'p01'，故选嵌套方案 (2,20)。
    assert [p["id"] for p in res["canonical_pairs"]] == ["out", "in"]
    assert res["residual_totals"] == [2, 20]
    assert res["classification"]["required"] == []
    assert set(res["classification"]["optional"]) == {"out", "in", "p01", "p23"}


def test_balanced_solution_beats_two_extremes():
    # 6 点上三个完美匹配（候选端点对均不重复，公共弧只定义一次）：
    #   A = u+v+w：(2,20)，工况1偏好；
    #   B = p+q+t：(20,2)，工况2偏好；
    #   C = m+q+w：(8,8)，两工况均衡，max=8 严格胜出。
    hits = make_hits(6)
    candidates = [
        rcand("u", "h0", "h1", 1, 7),
        rcand("v", "h2", "h3", 0, 7),
        rcand("w", "h4", "h5", 1, 6),
        rcand("p", "h0", "h5", 7, 1),
        rcand("q", "h1", "h2", 6, 1),
        rcand("t", "h3", "h4", 7, 0),
        rcand("m", "h0", "h3", 1, 1),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["paired_hits"] == 6
    assert res["residual_totals"] == [8, 8]
    assert res["max_residual_total"] == 8
    assert res["optimal_count"] == "1"
    assert [p["id"] for p in res["canonical_pairs"]] == ["m", "q", "w"]
    assert res["classification"]["required"] == ["m", "q", "w"]
    assert set(res["classification"]["never"]) == {"u", "v", "p", "t"}


def test_nested_compatible_solution():
    # 嵌套解与并列解都是 2 对；嵌套解在两工况下都不差（弱支配），
    # 且工况2严格更优 => 唯一最优为嵌套。
    hits = make_hits(4)
    candidates = [
        rcand("out", "h0", "h3", 3, 2),
        rcand("in", "h1", "h2", 1, 1),
        rcand("p01", "h0", "h1", 2, 5),
        rcand("p23", "h2", "h3", 2, 5),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["optimal_count"] == "1"
    assert res["residual_totals"] == [4, 3]
    assert res["max_residual_total"] == 4
    assert [p["id"] for p in res["canonical_pairs"]] == ["out", "in"]
    assert res["classification"]["required"] == ["in", "out"]
    assert set(res["classification"]["never"]) == {"p01", "p23"}


def test_max_pair_count_dominates_residual_balance():
    # 2 对的偏斜方案胜过 1 对的完美均衡方案。
    hits = make_hits(4)
    candidates = [
        rcand("two_outer", "h0", "h3", 0, 9),
        rcand("two_inner", "h1", "h2", 0, 9),
        rcand("one_balanced", "h0", "h1", 1, 1),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["paired_hits"] == 4
    assert res["residual_totals"] == [0, 18]
    assert [p["id"] for p in res["canonical_pairs"]] == ["two_outer", "two_inner"]
    assert "one_balanced" in res["classification"]["never"]


def test_sum_breaks_max_tie():
    # 两方案 max 并列为 5：(5,5) 与 (5,0)——sum 更小者胜出。
    hits = make_hits(4)
    candidates = [
        # 方案 A：(5,0)
        rcand("a_out", "h0", "h3", 4, 0),
        rcand("a_in", "h1", "h2", 1, 0),
        # 方案 B：(3,2) -> (5,5)
        rcand("b01", "h0", "h1", 3, 3),
        rcand("b23", "h2", "h3", 2, 2),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["max_residual_total"] == 5
    assert res["residual_totals"] == [5, 0]
    assert res["optimal_count"] == "1"
    assert [p["id"] for p in res["canonical_pairs"]] == ["a_out", "a_in"]
    assert set(res["classification"]["never"]) == {"b01", "b23"}


def test_arbitrary_precision_count():
    # 每 3 个击中一组，组内三条候选弧两工况残差完全相同 => 3 个最优方案，
    # 组间独立 => 3^(n/3)。
    n = 180
    hits = make_hits(n)
    candidates = []
    for a in range(0, n, 3):
        candidates.append(rcand(f"span{a}", f"h{a}", f"h{a+2}", 0, 0))
        candidates.append(rcand(f"adj1{a}", f"h{a}", f"h{a+1}", 0, 0))
        candidates.append(rcand(f"adj2{a}", f"h{a+1}", f"h{a+2}", 0, 0))
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["optimal_count"] == str(3 ** (n // 3))
    assert res["residual_totals"] == [0, 0]
    assert [p["id"] for p in res["canonical_pairs"]] == [
        f"adj1{a}" for a in range(0, n, 3)
    ]
    unmatched = [f"h{k}" for k in range(n) if k % 3 == 2]
    assert res["unmatched_hits"] == unmatched
    assert set(res["classification"]["optional"]) == {c["id"] for c in candidates}


def test_required_and_never_attribution():
    # 6 击中上两个并列最优完美匹配，共用公共弧 common：
    #   A: common(0,1) + x(2,5) + w(3,4)
    #   B: common(0,1) + y(2,3) + z(4,5)
    # 两方案均为 3 对、残差 (3,3)；bad 与 common 争端点，至多 2 对 => 从不出现。
    hits = make_hits(6)
    candidates = [
        rcand("common", "h0", "h1", 1, 1),
        rcand("x", "h2", "h5", 2, 2),
        rcand("w", "h3", "h4", 0, 0),
        rcand("y", "h2", "h3", 2, 2),
        rcand("z", "h4", "h5", 0, 0),
        rcand("bad", "h1", "h2", 9, 9),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert res["optimal_count"] == "2"
    assert res["residual_totals"] == [3, 3]
    assert res["max_residual_total"] == 3
    assert res["classification"]["required"] == ["common"]
    assert set(res["classification"]["optional"]) == {"w", "x", "y", "z"}
    assert res["classification"]["never"] == ["bad"]
    # 规范序列第二项 'x' < 'y'，故取方案 A。
    assert [p["id"] for p in res["canonical_pairs"]] == ["common", "x", "w"]
    assert res["unmatched_hits"] == []


def test_canonical_pairs_carry_both_residuals():
    hits = make_hits(4)
    candidates = [
        rcand("p", "h0", "h3", 2, 7),
        rcand("q", "h1", "h2", 3, 1),
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    pairs = {p["id"]: p for p in res["canonical_pairs"]}
    assert pairs["p"]["residuals"] == [2, 7]
    assert pairs["q"]["residuals"] == [3, 1]
    assert "residual" not in pairs["p"]
    assert res["residual_totals"] == [5, 8]


# ---------------------------------------------------------------- 校验错误


def invalid_payload(payload):
    with pytest.raises(ValidationError) as exc:
        audit_robust(payload)
    return exc.value.errors


def test_residuals_field_errors():
    errors = invalid_payload(
        {
            "hits": make_hits(4),
            "candidates": [
                {
                    "id": "c",
                    "left_endpoint": "h0",
                    "right_endpoint": "h1",
                    "residuals": [1],
                }
            ],
        }
    )
    assert any(e["field"] == "/candidates/0/residuals" for e in errors)

    errors = invalid_payload(
        {
            "hits": make_hits(4),
            "candidates": [
                {
                    "id": "c",
                    "left_endpoint": "h0",
                    "right_endpoint": "h1",
                    "residuals": [1, 2, 3],
                }
            ],
        }
    )
    assert any(e["field"] == "/candidates/0/residuals" for e in errors)

    errors = invalid_payload(
        {
            "hits": make_hits(4),
            "candidates": [
                {
                    "id": "c",
                    "left_endpoint": "h0",
                    "right_endpoint": "h1",
                    "residuals": [1, -2],
                }
            ],
        }
    )
    assert any(e["field"] == "/candidates/0/residuals/1" for e in errors)

    errors = invalid_payload(
        {
            "hits": make_hits(4),
            "candidates": [
                {
                    "id": "c",
                    "left_endpoint": "h0",
                    "right_endpoint": "h1",
                    "residuals": [True, 2],
                }
            ],
        }
    )
    assert any(e["field"] == "/candidates/0/residuals/0" for e in errors)

    errors = invalid_payload(
        {
            "hits": make_hits(4),
            "candidates": [
                {
                    "id": "c",
                    "left_endpoint": "h0",
                    "right_endpoint": "h1",
                    "residuals": "1,2",
                }
            ],
        }
    )
    assert any(e["field"] == "/candidates/0/residuals" for e in errors)


def test_endpoint_and_scale_errors_shared_rules():
    errors = invalid_payload(
        {
            "hits": make_hits(4),
            "candidates": [rcand("c", "h0", "ghost", 0, 0)],
        }
    )
    assert any(e["field"] == "/candidates/0/right_endpoint" for e in errors)

    errors = invalid_payload(
        {"hits": make_hits(4), "candidates": [rcand("c", "h2", "h1", 0, 0)]}
    )
    assert any(e["field"] == "/candidates/0/right_endpoint" for e in errors)

    errors = invalid_payload({"hits": make_hits(MIN_HITS - 1), "candidates": []})
    assert any(e["field"] == "/hits" for e in errors)

    big_hits = make_hits(MAX_HITS)
    rng = random.Random(0)
    many = []
    seen = set()
    while len(many) < MAX_CANDIDATES + 1:
        a = rng.randrange(MAX_HITS - 1)
        b = rng.randrange(a + 1, MAX_HITS)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        many.append(rcand(f"c{len(many)}", f"h{a}", f"h{b}", 0, 0))
    errors = invalid_payload({"hits": big_hits, "candidates": many})
    assert any(e["field"] == "/candidates" for e in errors)

    errors = invalid_payload({"hits": make_hits(4), "candidates": "nope"})
    assert any(e["field"] == "/candidates" for e in errors)

    errors = invalid_payload("not-an-object")
    assert errors[0]["field"] == ""


def test_error_response_carries_no_audit_fields():
    # solver 层：校验失败抛错且不返回任何结果字典。
    with pytest.raises(ValidationError):
        audit_robust(
            {
                "hits": make_hits(4),
                "candidates": [rcand("c", "h0", "h1", 0, -1)],
            }
        )


# ---------------------------------------------------------------- 随机暴力对照


@pytest.mark.parametrize("seed", range(80))
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
                rcand(
                    f"cid{next(counter):03d}",
                    ids[a],
                    ids[b],
                    rng.choice([0, 0, 1, 2, 5, 9]),
                    rng.choice([0, 0, 1, 2, 5, 9]),
                )
            )

    res = audit_robust({"hits": make_hits(n), "candidates": candidates})
    ref = brute_solve_robust(n, to_arc_records(candidates, ids))

    assert int(res["optimal_count"]) == ref["optimal_count"]
    assert res["paired_hits"] == 2 * ref["max_pairs"]
    assert res["max_residual_total"] == ref["min_max"]
    assert res["residual_totals"] == [ref["s1"], ref["s2"]]
    assert res["residual_totals"][0] + res["residual_totals"][1] == ref["min_sum"]
    assert [p["id"] for p in res["canonical_pairs"]] == ref["canonical"]

    hit_ids = [f"h{k}" for k in range(n)]
    assert res["unmatched_hits"] == [hit_ids[k] for k in ref["canonical_unmatched"]]

    cls = res["classification"]
    assert cls == ref["classification"]

    for cid, *_rest in to_arc_records(candidates, ids):
        used = ref["usage"][cid]
        if used == 0:
            assert cid in cls["never"]
        elif used == ref["optimal_count"]:
            assert cid in cls["required"]
        else:
            assert cid in cls["optional"]


# ---------------------------------------------------------------- 性能


def test_max_scale_performance():
    n = MAX_HITS
    hits = make_hits(n)
    rng = random.Random(42)
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    candidates = [
        rcand(f"c{k:04d}", f"h{a}", f"h{b}", rng.randrange(1000), rng.randrange(1000))
        for k, (a, b) in enumerate(possible[:MAX_CANDIDATES])
    ]
    res = audit_robust({"hits": hits, "candidates": candidates})
    assert int(res["optimal_count"]) >= 1
    assert len(res["canonical_pairs"]) * 2 == res["paired_hits"]
