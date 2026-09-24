"""暴力枚举参考实现：枚举全部非交叉匹配，供小规模随机交叉验证。

入参 arcs 为 (cid, a, b, residual) 列表，端点对 (a,b) 不重复。
"""

from __future__ import annotations


def brute_solve(n, arcs):
    by_left = [[] for _ in range(n)]
    for cid, a, b, r in arcs:
        by_left[a].append((b, r, cid))

    # sols[i][j] -> list[frozenset[cid]]：区间 [i,j) 上全部非交叉匹配。
    sols = [[[] for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        sols[i][i] = [frozenset()]

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            out = set(sols[i + 1][j])  # i 未配对
            for b, _r, cid in by_left[i]:
                if b >= j:
                    continue
                for inner in sols[i + 1][b]:
                    for outer in sols[b + 1][j]:
                        out.add(frozenset({cid}) | inner | outer)
            sols[i][j] = list(out)

    cost = {cid: r for cid, _a, _b, r in arcs}
    left_of = {cid: a for cid, a, _b, _r in arcs}
    scored = []
    for matching in sols[0][n]:
        scored.append((matching, len(matching), sum(cost[c] for c in matching)))

    max_pairs = max(p for _m, p, _c in scored)
    min_cost = min(c for _m, p, c in scored if p == max_pairs)
    optimal = [m for m, p, c in scored if p == max_pairs and c == min_cost]

    def id_sequence(matching):
        return sorted(matching, key=lambda c: (left_of[c], c))

    canonical = min((id_sequence(m) for m in optimal), key=list)

    usage = {cid: 0 for cid, _a, _b, _r in arcs}
    for matching in optimal:
        for cid in matching:
            usage[cid] += 1

    total_count = len(optimal)
    classification = {"required": [], "optional": [], "never": []}
    for cid, _a, _b, _r in arcs:
        used = usage[cid]
        if used == 0:
            classification["never"].append(cid)
        elif used == total_count:
            classification["required"].append(cid)
        else:
            classification["optional"].append(cid)
    classification = {k: sorted(v) for k, v in classification.items()}

    endpoint_of = {cid: (a, b) for cid, a, b, _r in arcs}
    unmatched = set(range(n))
    for cid in canonical:
        a, b = endpoint_of[cid]
        unmatched.discard(a)
        unmatched.discard(b)

    return {
        "optimal_count": total_count,
        "max_pairs": max_pairs,
        "min_cost": min_cost,
        "canonical": canonical,
        "canonical_unmatched": sorted(unmatched),
        "classification": classification,
        "usage": usage,
    }


def brute_solve_robust(n, arcs):
    """双增益工况暴力参考：arcs 为 (cid, a, b, r1, r2) 列表。

    字典序目标：(配对数, max(x,y), x+y, 左端位置顺序 id 序列)，
    其中 x、y 分别为两工况残差总和。分类/计数基于达到前三层目标的方案。
    """
    by_left = [[] for _ in range(n)]
    for cid, a, b, r1, r2 in arcs:
        by_left[a].append((b, r1, r2, cid))

    sols = [[[] for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        sols[i][i] = [frozenset()]

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            out = set(sols[i + 1][j])
            for b, _r1, _r2, cid in by_left[i]:
                if b >= j:
                    continue
                for inner in sols[i + 1][b]:
                    for outer in sols[b + 1][j]:
                        out.add(frozenset({cid}) | inner | outer)
            sols[i][j] = list(out)

    w1 = {cid: r1 for cid, _a, _b, r1, _r2 in arcs}
    w2 = {cid: r2 for cid, _a, _b, _r1, r2 in arcs}
    left_of = {cid: a for cid, a, _b, _r1, _r2 in arcs}

    scored = []
    for matching in sols[0][n]:
        x = sum(w1[c] for c in matching)
        y = sum(w2[c] for c in matching)
        scored.append((matching, len(matching), x, y))

    max_pairs = max(p for _m, p, _x, _y in scored)
    pool = [t for t in scored if t[1] == max_pairs]
    min_max = min(max(x, y) for _m, _p, x, y in pool)
    pool = [t for t in pool if max(t[2], t[3]) == min_max]
    min_sum = min(x + y for _m, _p, x, y in pool)
    optimal = [t for t in pool if t[2] + t[3] == min_sum]

    def id_sequence(matching):
        return sorted(matching, key=lambda c: (left_of[c], c))

    canonical = min((id_sequence(m) for m, *_ in optimal), key=list)
    canonical_point = next((x, y) for m, _p, x, y in optimal if id_sequence(m) == canonical)

    usage = {cid: 0 for cid, _a, _b, _r1, _r2 in arcs}
    for matching, *_ in optimal:
        for cid in matching:
            usage[cid] += 1

    total_count = len(optimal)
    classification = {"required": [], "optional": [], "never": []}
    for cid, _a, _b, _r1, _r2 in arcs:
        used = usage[cid]
        if used == 0:
            classification["never"].append(cid)
        elif used == total_count:
            classification["required"].append(cid)
        else:
            classification["optional"].append(cid)
    classification = {k: sorted(v) for k, v in classification.items()}

    endpoint_of = {cid: (a, b) for cid, a, b, _r1, _r2 in arcs}
    unmatched = set(range(n))
    for cid in canonical:
        a, b = endpoint_of[cid]
        unmatched.discard(a)
        unmatched.discard(b)

    return {
        "optimal_count": total_count,
        "max_pairs": max_pairs,
        "min_max": min_max,
        "min_sum": min_sum,
        "canonical": canonical,
        "canonical_point": canonical_point,
        "canonical_unmatched": sorted(unmatched),
        "classification": classification,
        "usage": usage,
    }
