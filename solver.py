"""硅微条击中配对审计求解器。

输入校验通过后，求解点列上的带权非交叉匹配问题（允许弧段相互嵌套）：

* 击中沿位置严格递增排列，候选对连接其中两个击中（左端点位置 < 右端点位置）；
* 选中的对端点互异，按位置绘制后两两不交叉（允许嵌套与并列）；
* 目标依次为：最大化已配对击中数（等价于最大化对数 * 2）、最小化残差总和；
* 在所有达到前两级目标的方案上统计任意精度方案数，输出按左端位置顺序下
  配对标识序列字典序最小的规范方案，并把每条候选对判为 required / optional / never。

算法采用区间 inside DP（求最优目标、方案数、规范序列）与 outside DP
（inside-outside，统计每条候选对出现在多少个最优方案中）。
所有计数使用 Python 任意精度整数；区间 DP 复杂度 O(n*|C| + n^3)，
n <= 180、|C| <= 4000。

此外提供双增益工况的鲁棒审计 ``audit_robust``：每条候选对同时携带两项
非负整数残差（两种增益标定下各一项），配对必须是同一套、不随工况改变的
非交叉匹配。目标依次为：

1. 最大化已配对击中数；
2. 最小化 ``max(工况一残差总和, 工况二残差总和)``；
3. 最小化两项残差总和之和；
4. 以左端位置顺序的配对 id 序列字典序最小者为规范方案。

``max`` 不可加，因此每个区间在固定最大配对数下维护 (x, y) 的 Pareto
最小点集（x、y 分别为两工况残差总和）：区间 DP 合并“跳过首点/首点配对”
两类规则后剔除被支配点；根前沿按 (max(x,y), x+y, id 序列) 裁决。
方案计数与候选分类基于达到前三层目标的根点（至多两个关于 x=y 对称的点），
由逐点 outside DP 统计每条候选弧的最优方案出现次数。
"""

from __future__ import annotations

from typing import Any, Optional

MIN_HITS = 4
MAX_HITS = 180
MAX_CANDIDATES = 4000


class ValidationError(Exception):
    """携带字段路径的请求校验错误。"""

    def __init__(self, errors: list[dict[str, str]]):
        super().__init__("; ".join(e["message"] for e in errors))
        self.errors = errors


def _err(errors: list[dict[str, str]], field: str, message: str) -> None:
    errors.append({"field": field, "message": message})


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate(
    payload: Any,
) -> tuple[list[dict[str, Any]], list[tuple[str, int, int, int]]]:
    """原审计入口校验：单残差候选。"""
    hits, records = _parse_payload(payload, ("residual",))
    return hits, [tuple(r) for r in records]  # type: ignore[misc]


def _validate_robust(
    payload: Any,
) -> tuple[list[dict[str, Any]], list[tuple[str, int, int, int, int]]]:
    """鲁棒审计入口校验：每条候选携带 residual_low / residual_high 两项残差。"""
    hits, records = _parse_payload(payload, ("residual_low", "residual_high"))
    return hits, [tuple(r) for r in records]  # type: ignore[misc]


def _parse_payload(
    payload: Any,
    residual_fields: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[tuple[Any, ...]]]:
    """两个审计入口共用的请求校验与候选解析。

    ``residual_fields`` 给出每条候选必须携带的非负整数残差字段名（原入口
    为单个 ``residual``，鲁棒入口为两工况各自的残差）；返回候选记录
    ``(id, 左端点下标, 右端点下标, *残差)``。
    """
    errors: list[dict[str, str]] = []

    if not isinstance(payload, dict):
        raise ValidationError([{"field": "", "message": "请求体必须是 JSON 对象"}])

    if "hits" not in payload:
        _err(errors, "/hits", "缺少 hits 字段")
    if "candidates" not in payload:
        _err(errors, "/candidates", "缺少 candidates 字段")
    if errors:
        raise ValidationError(errors)

    raw_hits = payload["hits"]
    raw_candidates = payload["candidates"]

    if not isinstance(raw_hits, list):
        _err(errors, "/hits", "hits 必须是数组")
        raw_hits = []
    if not isinstance(raw_candidates, list):
        _err(errors, "/candidates", "candidates 必须是数组")
        raw_candidates = []

    if isinstance(raw_hits, list) and not (MIN_HITS <= len(raw_hits) <= MAX_HITS):
        _err(
            errors,
            "/hits",
            f"击中数量必须在 {MIN_HITS} 到 {MAX_HITS} 之间，收到 {len(raw_hits)}",
        )
    if isinstance(raw_candidates, list) and len(raw_candidates) > MAX_CANDIDATES:
        _err(
            errors,
            "/candidates",
            f"候选配对数量不能超过 {MAX_CANDIDATES}，收到 {len(raw_candidates)}",
        )

    hits: list[dict[str, Any]] = []
    seen_hit_ids: set[str] = set()

    for k, hit in enumerate(raw_hits if isinstance(raw_hits, list) else []):
        base = f"/hits/{k}"
        if not isinstance(hit, dict):
            _err(errors, base, "击中必须是对象")
            continue
        hid = hit.get("id")
        pos = hit.get("position")
        if not isinstance(hid, str) or not hid:
            _err(errors, f"{base}/id", "击中 id 必须是非空字符串")
        elif hid in seen_hit_ids:
            _err(errors, f"{base}/id", f"击中 id 重复: {hid}")
        else:
            seen_hit_ids.add(hid)
        if not _is_int(pos):
            _err(errors, f"{base}/position", "position 必须是整数")
            pos = None
        hits.append({"id": hid, "position": pos})

    if not errors:
        for k in range(1, len(hits)):
            if hits[k]["position"] <= hits[k - 1]["position"]:
                _err(
                    errors,
                    f"/hits/{k}/position",
                    f"位置必须严格递增: {hits[k - 1]['position']} 之后出现 "
                    f"{hits[k]['position']}",
                )
                break

    candidate_records: list[tuple[str, int, int, int]] = []
    if not any(e["field"].startswith("/hits") for e in errors):
        index_by_id = {h["id"]: k for k, h in enumerate(hits)}
        seen_pair_ids: set[str] = set()
        seen_endpoint_pairs: set[tuple[int, int]] = set()

        for k, cand in enumerate(raw_candidates if isinstance(raw_candidates, list) else []):
            base = f"/candidates/{k}"
            if not isinstance(cand, dict):
                _err(errors, base, "候选配对必须是对象")
                continue
            cid = cand.get("id")
            left = cand.get("left_endpoint")
            right = cand.get("right_endpoint")

            if not isinstance(cid, str) or not cid:
                _err(errors, f"{base}/id", "候选 id 必须是非空字符串")
            elif cid in seen_pair_ids:
                _err(errors, f"{base}/id", f"候选 id 重复: {cid}")
            else:
                seen_pair_ids.add(cid)

            residuals: list[Any] = []
            residuals_ok = True
            for field in residual_fields:
                value = cand.get(field)
                if not _is_int(value):
                    _err(errors, f"{base}/{field}", f"{field} 必须是非负整数")
                    residuals_ok = False
                elif value < 0:
                    _err(errors, f"{base}/{field}", f"{field} 不能为负，收到 {value}")
                    residuals_ok = False
                residuals.append(value)

            if not isinstance(left, str):
                _err(errors, f"{base}/left_endpoint", "left_endpoint 必须是字符串标识")
            elif left not in index_by_id:
                _err(errors, f"{base}/left_endpoint", f"未知端点标识: {left}")

            if not isinstance(right, str):
                _err(errors, f"{base}/right_endpoint", "right_endpoint 必须是字符串标识")
            elif right not in index_by_id:
                _err(errors, f"{base}/right_endpoint", f"未知端点标识: {right}")

            if (
                isinstance(left, str)
                and isinstance(right, str)
                and left in index_by_id
                and right in index_by_id
            ):
                a = index_by_id[left]
                b = index_by_id[right]
                if a >= b:
                    _err(
                        errors,
                        f"{base}/right_endpoint",
                        "右端点位置必须严格大于左端点位置，且两端点必须不同",
                    )
                elif (a, b) in seen_endpoint_pairs:
                    _err(errors, base, f"重复端点对: ({left}, {right})")
                else:
                    # 端点对登记与字段合法性无关，保持重复检测行为一致。
                    seen_endpoint_pairs.add((a, b))
                    if isinstance(cid, str) and bool(cid) and residuals_ok:
                        candidate_records.append((cid, a, b, *residuals))

    if errors:
        raise ValidationError(errors)

    return hits, candidate_records


def audit(payload: Any) -> dict[str, Any]:
    """执行完整审计，返回可直接 JSON 序列化的结果。"""

    hits, candidates = _validate(payload)
    n = len(hits)

    # arcs[i]: 以位置 i 为左端点的候选 (右端点, 残差, id)。
    arcs: list[list[tuple[int, int, str]]] = [[] for _ in range(n)]
    for cid, a, b, r in candidates:
        arcs[a].append((b, r, cid))

    # ---------- inside 区间 DP ----------
    # P[i][j]/C[i][j]/W[i][j]：区间 [i,j) 上的最大对数、最小残差、最优方案数。
    P = [[0] * (n + 1) for _ in range(n + 1)]
    C = [[0] * (n + 1) for _ in range(n + 1)]
    W = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        W[i][i] = 1

    def take(pairs: int, cost: int, ways: int, sequence: Optional[list[str]]) -> None:
        """把一条规则的结果并入当前区间的最优值。"""
        if ways == 0:
            return
        if best[0] is None or pairs > best[0] or (pairs == best[0] and cost < best[1]):
            best[0] = pairs
            best[1] = cost
            best[2] = ways
            best[3] = sequence
        elif pairs == best[0] and cost == best[1]:
            best[2] += ways
            if sequence is not None and (best[3] is None or sequence < best[3]):
                best[3] = sequence

    # seq[i][j]：区间 [i,j) 最优方案中按左端位置顺序的最小 id 序列；
    # choice 记录对应首步：('s',) 跳过 i，或 ('p', k, cid) 以弧 (i,k) 配对。
    seq: list[list[Optional[list[str]]]] = [[None] * (n + 1) for _ in range(n + 1)]
    choice: list[list[Optional[tuple[Any, ...]]]] = [
        [None] * (n + 1) for _ in range(n + 1)
    ]
    for i in range(n + 1):
        seq[i][i] = []

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            best: list[Any] = [None, None, 0, None]  # pairs, cost, ways, 最小序列
            # 规则 1：i 未配对。
            take(P[i + 1][j], C[i + 1][j], W[i + 1][j], seq[i + 1][j])
            skip_ties = (P[i + 1][j], C[i + 1][j])
            # 规则 2：i 与 k 配对，内部 [i+1,k) 与外部 [k+1,j) 独立。
            for k, r, cid in arcs[i]:
                if k >= j:
                    continue
                take(
                    1 + P[i + 1][k] + P[k + 1][j],
                    r + C[i + 1][k] + C[k + 1][j],
                    W[i + 1][k] * W[k + 1][j],
                    [cid] + seq[i + 1][k] + seq[k + 1][j],
                )

            P[i][j] = best[0]
            C[i][j] = best[1]
            W[i][j] = best[2]
            seq[i][j] = best[3]

            # 确定取得最小 id 序列的首步规则。
            chosen: Optional[tuple[Any, ...]] = None
            if skip_ties == (P[i][j], C[i][j]) and seq[i + 1][j] == best[3]:
                chosen = ("s",)
            else:
                for k, r, cid in arcs[i]:
                    if k >= j:
                        continue
                    if (
                        1 + P[i + 1][k] + P[k + 1][j] == P[i][j]
                        and r + C[i + 1][k] + C[k + 1][j] == C[i][j]
                    ):
                        candidate = [cid] + seq[i + 1][k] + seq[k + 1][j]  # type: ignore[operator]
                        if candidate == best[3]:
                            chosen = ("p", k, cid)
                            break
            choice[i][j] = chosen

    total = W[0][n]

    # ---------- outside DP ----------
    # O[i][j]：根区间 [0,n) 的最优方案中，[i,j) 作为一个内部最优子区间出现的
    # 方案数（外部上下文数）。按父区间向其两个子区间下发贡献。
    Out = [[0] * (n + 1) for _ in range(n + 1)]
    Out[0][n] = 1
    for length in range(n, 0, -1):
        for h in range(0, n - length + 1):
            m = h + length
            outside = Out[h][m]
            if outside == 0:
                continue
            # 跳过规则：父 [h,m) -> 子 [h+1,m)。
            if P[h + 1][m] == P[h][m] and C[h + 1][m] == C[h][m]:
                Out[h + 1][m] += outside
            # 配对规则：父 [h,m) 经弧 (h,k) -> 左子 [h+1,k)、右子 [k+1,m)。
            for k, r, _cid in arcs[h]:
                if k >= m:
                    continue
                if (
                    1 + P[h + 1][k] + P[k + 1][m] == P[h][m]
                    and r + C[h + 1][k] + C[k + 1][m] == C[h][m]
                ):
                    Out[h + 1][k] += outside * W[k + 1][m]
                    Out[k + 1][m] += outside * W[h + 1][k]

    # ---------- 候选对出现次数与分类 ----------
    # 弧 (a,b) 作为某父区间 [a,m) 的首步规则出现：
    # 出现方案数 = W[a+1][b] * Σ_m O[a][m] * W[b+1][m]（仅计最优规则）。
    used_count: dict[str, int] = {}
    for cid, a, b, r in candidates:
        count = 0
        interior_ways = W[a + 1][b]
        for m in range(b + 1, n + 1):
            if (
                1 + P[a + 1][b] + P[b + 1][m] == P[a][m]
                and r + C[a + 1][b] + C[b + 1][m] == C[a][m]
            ):
                count += Out[a][m] * interior_ways * W[b + 1][m]
        used_count[cid] = count

    required: list[str] = []
    optional: list[str] = []
    never: list[str] = []
    for cid, a, b, _r in candidates:
        c = used_count[cid]
        if c == 0:
            never.append(cid)
        elif c == total:
            required.append(cid)
        else:
            optional.append(cid)

    # ---------- 规范方案回溯 ----------
    canonical_ids: list[str] = []
    unmatched_idx: list[int] = []

    def build(i: int, j: int) -> None:
        while i < j:
            step = choice[i][j]
            assert step is not None
            if step[0] == "s":
                unmatched_idx.append(i)
                i += 1
            else:
                k, cid = step[1], step[2]
                canonical_ids.append(cid)
                build(i + 1, k)
                i = k + 1

    build(0, n)

    info = {cid: (a, b, r) for cid, a, b, r in candidates}
    canonical_pairs = [
        {
            "id": cid,
            "left_endpoint": hits[info[cid][0]]["id"],
            "right_endpoint": hits[info[cid][1]]["id"],
            "residual": info[cid][2],
        }
        for cid in canonical_ids
    ]

    return {
        # 以字符串承载任意精度十进制整数，避免客户端 JSON 大整数精度损失。
        "optimal_count": str(total),
        "paired_hits": 2 * P[0][n],
        "total_residual": C[0][n],
        "canonical_pairs": canonical_pairs,
        "unmatched_hits": [hits[k]["id"] for k in unmatched_idx],
        "classification": {
            "required": sorted(required),
            "optional": sorted(optional),
            "never": sorted(never),
        },
    }


# ---------------------------------------------------------------------------
# 双增益工况鲁棒审计
# ---------------------------------------------------------------------------

# 区间状态：点 (x, y) = （工况一残差总和, 工况二残差总和）；
# 每个区间在固定最大配对数下保留 (x, y) 的 Pareto 最小点集（两坐标均越小越好，
# 一点 (x',y') 支配 (x,y) 当且仅当 x'<=x 且 y'<=y 且不全等）。
Point = tuple[int, int]
PointState = tuple[int, tuple[str, ...]]  # (该点方案数, 最小左端顺序 id 序列)


def _pareto_minima(mp: dict[Point, PointState]) -> dict[Point, PointState]:
    """剔除被支配点：按 x 升序（同 x 按 y 升序）扫描，仅保留 y 严格刷新
    最小值的点。存活点按 x 升序排列时 y 严格递减。"""
    kept: dict[Point, PointState] = {}
    best_y: Optional[int] = None
    for point, state in sorted(mp.items()):
        y = point[1]
        if best_y is None or y < best_y:
            kept[point] = state
            best_y = y
    return kept


def audit_robust(payload: Any) -> dict[str, Any]:
    """双增益工况鲁棒审计，返回可直接 JSON 序列化的结果。

    每条候选携带 ``residual_low`` / ``residual_high`` 两项非负整数残差；
    优化顺序为最大化配对数、最小化 max(两工况残差总和)、最小化两者之和，
    再以左端位置顺序的配对 id 序列字典序裁决规范方案。方案计数与候选分类
    基于达到前三层目标的全部方案。
    """

    hits, candidates = _validate_robust(payload)
    n = len(hits)

    # arcs[i]: 以位置 i 为左端点的候选 (右端点, 工况一残差, 工况二残差, id)。
    arcs: list[list[tuple[int, int, int, str]]] = [[] for _ in range(n)]
    for cid, a, b, r1, r2 in candidates:
        arcs[a].append((b, r1, r2, cid))

    # ---------- inside 区间 DP（Pareto 前沿） ----------
    # P[i][j]：区间 [i,j) 的最大配对对数。
    # F[i][j]：达到 P[i][j] 的方案中 (x,y) 的 Pareto 最小点集，
    #          点 -> (任意精度方案数, 最小 id 序列)。
    P = [[0] * (n + 1) for _ in range(n + 1)]
    F: list[list[dict[Point, PointState]]] = [
        [dict() for _ in range(n + 1)] for _ in range(n + 1)
    ]
    for i in range(n + 1):
        F[i][i] = {(0, 0): (1, ())}

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            pmax = P[i + 1][j]
            for k, _r1, _r2, _cid in arcs[i]:
                if k < j:
                    pairs = 1 + P[i + 1][k] + P[k + 1][j]
                    if pairs > pmax:
                        pmax = pairs
            P[i][j] = pmax

            merged: dict[Point, PointState] = {}

            def absorb(point: Point, ways: int, sequence: tuple[str, ...]) -> None:
                old = merged.get(point)
                if old is None:
                    merged[point] = (ways, sequence)
                else:
                    old_ways, old_seq = old
                    merged[point] = (
                        old_ways + ways,
                        sequence if sequence < old_seq else old_seq,
                    )

            # 规则 1：i 未配对。
            if P[i + 1][j] == pmax:
                for point, (ways, seq) in F[i + 1][j].items():
                    absorb(point, ways, seq)
            # 规则 2：i 与 k 配对，内部 [i+1,k) 与外部 [k+1,j) 独立，
            # 点为两子区间点的 Minkowski 和再叠加该弧两项残差。
            for k, r1, r2, cid in arcs[i]:
                if k >= j or 1 + P[i + 1][k] + P[k + 1][j] != pmax:
                    continue
                head = (cid,)
                inner = F[i + 1][k]
                outer = F[k + 1][j]
                for (xi, yi), (wi, si) in inner.items():
                    for (xo, yo), (wo, so) in outer.items():
                        absorb(
                            (xi + xo + r1, yi + yo + r2),
                            wi * wo,
                            head + si + so,
                        )

            F[i][j] = _pareto_minima(merged)

    # ---------- 根前沿裁决：(max, sum, id 序列) 字典序 ----------
    def rank_key(item: tuple[Point, PointState]) -> tuple[int, int, tuple[str, ...]]:
        (x, y), (_ways, seq) = item
        return (max(x, y), x + y, seq)

    (best_x, best_y), (_best_ways, canonical_seq) = min(F[0][n].items(), key=rank_key)
    best_max = max(best_x, best_y)
    best_sum = best_x + best_y
    # 达到前三层目标的根点（同 max、同 sum）；Pareto 前沿上至多两个
    # （关于 x=y 对称的）点。
    global_points = {
        point
        for point in F[0][n]
        if max(point[0], point[1]) == best_max and point[0] + point[1] == best_sum
    }
    total = sum(F[0][n][point][0] for point in global_points)

    # ---------- 逐点 outside DP ----------
    # O[i][j][p]：根区间达到前三层目标的方案中，子区间 [i,j) 以其 Pareto
    # 点 p 作为最优子结构出现的方案数（外部上下文数）。下发两类首步规则时
    # 顺带统计每条候选弧出现在多少个最优方案中。
    O: list[list[dict[Point, int]]] = [
        [dict() for _ in range(n + 1)] for _ in range(n + 1)
    ]
    for point in global_points:
        O[0][n][point] = 1

    used_count: dict[str, int] = {cid: 0 for cid, _a, _b, _r1, _r2 in candidates}

    for length in range(n, 0, -1):
        for h in range(0, n - length + 1):
            m = h + length
            outside = O[h][m]
            if not outside:
                continue
            # 跳过规则：父点不变地下发到 [h+1,m)。
            if P[h + 1][m] == P[h][m]:
                child = F[h + 1][m]
                out_child = O[h + 1][m]
                for point, count in outside.items():
                    if point in child:
                        out_child[point] = out_child.get(point, 0) + count
            # 配对规则：父点 p = pi + po + (r1,r2)，枚举内部点 pi 并在
            # 外部点表中直接查补点 po。
            for k, r1, r2, cid in arcs[h]:
                if k >= m or 1 + P[h + 1][k] + P[k + 1][m] != P[h][m]:
                    continue
                inner = F[h + 1][k]
                outer = F[k + 1][m]
                out_inner = O[h + 1][k]
                out_outer = O[k + 1][m]
                for (x, y), count in outside.items():
                    tx, ty = x - r1, y - r2
                    for (xi, yi), (wi, _si) in inner.items():
                        po = (tx - xi, ty - yi)
                        outer_state = outer.get(po)
                        if outer_state is None:
                            continue
                        wo = outer_state[0]
                        # 该分解贡献 count * wi * wo 个最优方案，弧 cid
                        # 全部出现一次；向两个子点下发上下文数。
                        used_count[cid] += count * wi * wo
                        out_inner[(xi, yi)] = out_inner.get((xi, yi), 0) + count * wo
                        out_outer[po] = out_outer.get(po, 0) + count * wi

    required: list[str] = []
    optional: list[str] = []
    never: list[str] = []
    for cid, _a, _b, _r1, _r2 in candidates:
        count = used_count[cid]
        if count == 0:
            never.append(cid)
        elif count == total:
            required.append(cid)
        else:
            optional.append(cid)

    # ---------- 规范方案输出 ----------
    info = {cid: (a, b, r1, r2) for cid, a, b, r1, r2 in candidates}
    matched_idx: set[int] = set()
    canonical_pairs = []
    for cid in canonical_seq:
        a, b, r1, r2 = info[cid]
        matched_idx.add(a)
        matched_idx.add(b)
        canonical_pairs.append(
            {
                "id": cid,
                "left_endpoint": hits[a]["id"],
                "right_endpoint": hits[b]["id"],
                "residual_low": r1,
                "residual_high": r2,
            }
        )

    return {
        # 以字符串承载任意精度十进制整数，避免客户端 JSON 大整数精度损失。
        "optimal_count": str(total),
        "paired_hits": 2 * P[0][n],
        "total_residual_low": best_x,
        "total_residual_high": best_y,
        "canonical_pairs": canonical_pairs,
        "unmatched_hits": [hits[k]["id"] for k in range(n) if k not in matched_idx],
        "classification": {
            "required": sorted(required),
            "optional": sorted(optional),
            "never": sorted(never),
        },
    }
