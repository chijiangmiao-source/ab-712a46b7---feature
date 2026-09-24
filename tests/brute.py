"""暴力枚举参考实现：枚举全部非交叉匹配，供小规模随机交叉验证。

入参 arcs 为 (cid, a, b, residual) 列表，端点对 (a,b) 不重复。
"""

from __future__ import annotations


def _enumerate(n, by_left):
    # sols[i][j] -> set[frozenset[cid]]：区间 [i,j) 上全部非交叉匹配。
    sols = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        sols[i][i] = {frozenset()}

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            out = set(sols[i + 1][j])  # i 未配对
            for b, _w, cid in by_left[i]:
                if b >= j:
                    continue
                for inner in sols[i + 1][b]:
                    for outer in sols[b + 1][j]:
                        out.add(frozenset({cid}) | inner | outer)
            sols[i][j] = out
    return sols


def _classify(arcs, optimal, total_count):
    left_of = {cid: a for cid, a, *_ in arcs}
    usage = {cid: 0 for cid, *_ in arcs}
    for matching in optimal:
        for cid in matching:
            usage[cid] += 1

    classification = {"required": [], "optional": [], "never": []}
    for cid, *_ in arcs:
        used = usage[cid]
        if used == 0:
            classification["never"].append(cid)
        elif used == total_count:
            classification["required"].append(cid)
        else:
            classification["optional"].append(cid)
    classification = {k: sorted(v) for k, v in classification.items()}
    return classification, usage, left_of


def _canonical(optimal, left_of):
    def id_sequence(matching):
        return sorted(matching, key=lambda c: (left_of[c], c))

    return min((id_sequence(m) for m in optimal), key=list)


def brute_solve(n, arcs):
    by_left = [[] for _ in range(n)]
    for cid, a, b, r in arcs:
        by_left[a].append((b, r, cid))

    sols = _enumerate(n, by_left)

    cost = {cid: r for cid, _a, _b, r in arcs}
    scored = []
    for matching in sols[0][n]:
        scored.append((matching, len(matching), sum(cost[c] for c in matching)))

    max_pairs = max(p for _m, p, _c in scored)
    min_cost = min(c for _m, p, c in scored if p == max_pairs)
    optimal = [m for m, p, c in scored if p == max_pairs and c == min_cost]

    canonical = _canonical(optimal, {cid: a for cid, a, _b, _r in arcs})

    classification, usage, _left = _classify(arcs, optimal, len(optimal))

    endpoint_of = {cid: (a, b) for cid, a, b, _r in arcs}
    unmatched = set(range(n))
    for cid in canonical:
        a, b = endpoint_of[cid]
        unmatched.discard(a)
        unmatched.discard(b)

    return {
        "optimal_count": len(optimal),
        "max_pairs": max_pairs,
        "min_cost": min_cost,
        "canonical": canonical,
        "canonical_unmatched": sorted(unmatched),
        "classification": classification,
        "usage": usage,
    }


def brute_solve_robust(n, arcs):
    """双工况暴力参考：arcs 为 (cid, a, b, r1, r2)。

    目标层级：最大对数 -> min(max(S1,S2)) -> min(S1+S2)；
    规范解按左端位置顺序的 id 序列字典序最小。
    """
    by_left = [[] for _ in range(n)]
    for cid, a, b, r1, r2 in arcs:
        by_left[a].append((b, (r1, r2), cid))

    sols = _enumerate(n, by_left)

    cost = {cid: (r1, r2) for cid, _a, _b, r1, r2 in arcs}
    scored = []
    for matching in sols[0][n]:
        s1 = sum(cost[c][0] for c in matching)
        s2 = sum(cost[c][1] for c in matching)
        scored.append((matching, len(matching), s1, s2))

    max_pairs = max(p for _m, p, _s1, _s2 in scored)
    pool = [row for row in scored if row[1] == max_pairs]
    min_max = min(max(s1, s2) for _m, _p, s1, s2 in pool)
    pool = [row for row in pool if max(row[2], row[3]) == min_max]
    min_sum = min(s1 + s2 for _m, _p, s1, s2 in pool)
    optimal = [m for m, _p, s1, s2 in pool if s1 + s2 == min_sum]

    left_of = {cid: a for cid, a, _b, _r1, _r2 in arcs}
    canonical = _canonical(optimal, left_of)
    classification, usage, _left = _classify(arcs, optimal, len(optimal))

    endpoint_of = {cid: (a, b) for cid, a, b, _r1, _r2 in arcs}
    totals = {"s1": 0, "s2": 0}
    unmatched = set(range(n))
    for cid in canonical:
        totals["s1"] += cost[cid][0]
        totals["s2"] += cost[cid][1]
        a, b = endpoint_of[cid]
        unmatched.discard(a)
        unmatched.discard(b)

    return {
        "optimal_count": len(optimal),
        "max_pairs": max_pairs,
        "min_max": min_max,
        "min_sum": min_sum,
        "s1": totals["s1"],
        "s2": totals["s2"],
        "canonical": canonical,
        "canonical_unmatched": sorted(unmatched),
        "classification": classification,
        "usage": usage,
    }
