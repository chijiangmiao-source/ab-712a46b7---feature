"""硅微条击中配对审计求解器。

提供两种审计入口：

* ``audit``（单工况）：候选带一个 ``residual``，目标依次为最大化已配对击中数、
  最小化残差总和；
* ``audit_robust``（双增益工况鲁棒审计）：每条候选同时提交两项非负整数残差
  ``residuals``（两套增益标定），选中的配对集不允许随工况改变，目标依次为：
  1. 最大化已配对击中数（等价于最大化对数 * 2）；
  2. 最小化两工况残差总和的较大者 ``max(S1, S2)``；
  3. 最小化两工况残差总和之和 ``S1 + S2``；
  4. 以左端位置顺序下配对标识序列字典序最小者为规范方案。

两种入口共用击中与端点规则：击中沿位置严格递增排列，候选对连接其中两个击中
（左端点位置 < 右端点位置），选中的对端点互异、两两不交叉（允许嵌套与并列）。

单工况算法采用区间 inside DP + outside DP。鲁棒入口在区间状态上维护
``(S1, S2)`` 的 Pareto 前沿（固定最大对数下，两方向都不被支配的残差向量），
再以同样的 inside-outside 方式统计每条候选出现在多少个达到前三层目标的方案中，
据此分类 required / optional / never。所有计数使用 Python 任意精度整数。
n <= 180、|C| <= 4000。
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


# ---------------------------------------------------------------- 校验


def _validate_hits(payload: dict[str, Any], errors: list[dict[str, str]]) -> list[dict[str, Any]]:
    """校验 hits（两种入口共用），返回 [{id, position}]；错误写入 errors。"""
    raw_hits = payload["hits"]
    if not isinstance(raw_hits, list):
        _err(errors, "/hits", "hits 必须是数组")
        return []

    if not (MIN_HITS <= len(raw_hits) <= MAX_HITS):
        _err(
            errors,
            "/hits",
            f"击中数量必须在 {MIN_HITS} 到 {MAX_HITS} 之间，收到 {len(raw_hits)}",
        )

    hits: list[dict[str, Any]] = []
    seen_hit_ids: set[str] = set()

    for k, hit in enumerate(raw_hits):
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

    if not any(e["field"].startswith("/hits") for e in errors):
        for k in range(1, len(hits)):
            if hits[k]["position"] <= hits[k - 1]["position"]:
                _err(
                    errors,
                    f"/hits/{k}/position",
                    f"位置必须严格递增: {hits[k - 1]['position']} 之后出现 "
                    f"{hits[k]['position']}",
                )
                break

    return hits


def _validate_endpoints(
    left: Any,
    right: Any,
    index_by_id: dict[str, int],
    base: str,
    errors: list[dict[str, str]],
) -> tuple[Optional[int], Optional[int], bool]:
    """校验一条候选的左右端点，返回 (左下标, 右下标, 是否合法有序)。"""
    if not isinstance(left, str):
        _err(errors, f"{base}/left_endpoint", "left_endpoint 必须是字符串标识")
    elif left not in index_by_id:
        _err(errors, f"{base}/left_endpoint", f"未知端点标识: {left}")

    if not isinstance(right, str):
        _err(errors, f"{base}/right_endpoint", "right_endpoint 必须是字符串标识")
    elif right not in index_by_id:
        _err(errors, f"{base}/right_endpoint", f"未知端点标识: {right}")

    if not (
        isinstance(left, str)
        and isinstance(right, str)
        and left in index_by_id
        and right in index_by_id
    ):
        return None, None, False

    a = index_by_id[left]
    b = index_by_id[right]
    if a >= b:
        _err(
            errors,
            f"{base}/right_endpoint",
            "右端点位置必须严格大于左端点位置，且两端点必须不同",
        )
        return a, b, False
    return a, b, True


def _check_top_level(payload: Any) -> None:
    errors: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        raise ValidationError([{"field": "", "message": "请求体必须是 JSON 对象"}])
    if "hits" not in payload:
        _err(errors, "/hits", "缺少 hits 字段")
    if "candidates" not in payload:
        _err(errors, "/candidates", "缺少 candidates 字段")
    if errors:
        raise ValidationError(errors)


def _validate(
    payload: Any,
) -> tuple[list[dict[str, Any]], list[tuple[str, int, int, int]]]:
    _check_top_level(payload)
    assert isinstance(payload, dict)
    errors: list[dict[str, str]] = []

    hits = _validate_hits(payload, errors)

    raw_candidates = payload["candidates"]
    if not isinstance(raw_candidates, list):
        _err(errors, "/candidates", "candidates 必须是数组")
        raw_candidates = []
    if isinstance(raw_candidates, list) and len(raw_candidates) > MAX_CANDIDATES:
        _err(
            errors,
            "/candidates",
            f"候选配对数量不能超过 {MAX_CANDIDATES}，收到 {len(raw_candidates)}",
        )

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
            residual = cand.get("residual")

            if not isinstance(cid, str) or not cid:
                _err(errors, f"{base}/id", "候选 id 必须是非空字符串")
            elif cid in seen_pair_ids:
                _err(errors, f"{base}/id", f"候选 id 重复: {cid}")
            else:
                seen_pair_ids.add(cid)

            if not _is_int(residual):
                _err(errors, f"{base}/residual", "residual 必须是非负整数")
            elif residual < 0:
                _err(errors, f"{base}/residual", f"residual 不能为负，收到 {residual}")

            a, b, endpoint_ok = _validate_endpoints(
                left, right, index_by_id, base, errors
            )
            if endpoint_ok:
                if (a, b) in seen_endpoint_pairs:
                    _err(errors, base, f"重复端点对: ({left}, {right})")
                else:
                    seen_endpoint_pairs.add((a, b))
                    if isinstance(cid, str) and cid and _is_int(residual) and residual >= 0:
                        candidate_records.append((cid, a, b, residual))

    if errors:
        raise ValidationError(errors)

    return hits, candidate_records


def _validate_robust(
    payload: Any,
) -> tuple[list[dict[str, Any]], list[tuple[str, int, int, int, int]]]:
    """鲁棒入口校验：候选以 residuals: [非负整数, 非负整数] 同时提交两工况残差。"""
    _check_top_level(payload)
    assert isinstance(payload, dict)
    errors: list[dict[str, str]] = []

    hits = _validate_hits(payload, errors)

    raw_candidates = payload["candidates"]
    if not isinstance(raw_candidates, list):
        _err(errors, "/candidates", "candidates 必须是数组")
        raw_candidates = []
    if isinstance(raw_candidates, list) and len(raw_candidates) > MAX_CANDIDATES:
        _err(
            errors,
            "/candidates",
            f"候选配对数量不能超过 {MAX_CANDIDATES}，收到 {len(raw_candidates)}",
        )

    candidate_records: list[tuple[str, int, int, int, int]] = []
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
            residuals = cand.get("residuals")

            if not isinstance(cid, str) or not cid:
                _err(errors, f"{base}/id", "候选 id 必须是非空字符串")
            elif cid in seen_pair_ids:
                _err(errors, f"{base}/id", f"候选 id 重复: {cid}")
            else:
                seen_pair_ids.add(cid)

            residuals_ok = False
            r0 = r1 = 0
            if not isinstance(residuals, list):
                _err(errors, f"{base}/residuals", "residuals 必须是包含两个非负整数的数组")
            elif len(residuals) != 2:
                _err(
                    errors,
                    f"{base}/residuals",
                    f"residuals 必须恰好包含两项残差，收到 {len(residuals)} 项",
                )
            else:
                bad = False
                values: list[int] = []
                for t, rv in enumerate(residuals):
                    if not _is_int(rv):
                        _err(
                            errors,
                            f"{base}/residuals/{t}",
                            "residuals 每一项都必须是非负整数",
                        )
                        bad = True
                    elif rv < 0:
                        _err(
                            errors,
                            f"{base}/residuals/{t}",
                            f"残差不能为负，收到 {rv}",
                        )
                        bad = True
                    else:
                        values.append(rv)
                if not bad:
                    residuals_ok = True
                    r0, r1 = values[0], values[1]

            a, b, endpoint_ok = _validate_endpoints(
                left, right, index_by_id, base, errors
            )
            if endpoint_ok:
                if (a, b) in seen_endpoint_pairs:
                    _err(errors, base, f"重复端点对: ({left}, {right})")
                else:
                    seen_endpoint_pairs.add((a, b))
                    if isinstance(cid, str) and cid and residuals_ok:
                        candidate_records.append((cid, a, b, r0, r1))

    if errors:
        raise ValidationError(errors)

    return hits, candidate_records


# ---------------------------------------------------------------- 单工况审计


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


# ---------------------------------------------------------------- 双工况鲁棒审计


class _Frontier:
    """固定区间、固定最大配对数下的 (s1, s2) Pareto 前沿。

    点按 s1 严格递增存储，则 s2 严格递减；每个点附带：
      ways  —— 取得该残差向量的（任意精度）方案数；
      seq   —— 这些方案中按左端位置顺序最小的配对 id 序列。
    """

    __slots__ = ("pairs", "pts")

    def __init__(self, pairs: int, pts: list[list[Any]]):
        self.pairs = pairs
        # pts 元素: [s1, s2, ways, seq]
        self.pts = pts

    @staticmethod
    def empty() -> "_Frontier":
        return _Frontier(0, [[0, 0, 1, []]])


def _build_frontier(pairs: int, pts_map: dict[tuple[int, int], list[Any]]) -> _Frontier:
    """对同一配对数下聚合好的向量做支配过滤，保留 Pareto 前沿。"""
    if not pts_map:
        return _Frontier(pairs, [])
    raw = sorted(pts_map.values(), key=lambda p: (p[0], p[1]))
    pts: list[list[Any]] = []
    best_s2: Optional[int] = None
    for p in raw:
        s1, s2 = p[0], p[1]
        # 同 s1：raw 已按 s2 升序，仅第一个（s2 最小）可能存活；其后的被支配。
        if pts and pts[-1][0] == s1:
            continue
        if best_s2 is None or s2 < best_s2:
            pts.append(p)
            best_s2 = s2
    return _Frontier(pairs, pts)


def audit_robust(payload: Any) -> dict[str, Any]:
    """双增益工况鲁棒审计，返回可直接 JSON 序列化的结果。"""

    hits, candidates = _validate_robust(payload)
    n = len(hits)

    # arcs[i]: 以位置 i 为左端点的候选 (右端点, 工况1残差, 工况2残差, id)。
    arcs: list[list[tuple[int, int, int, str]]] = [[] for _ in range(n)]
    for cid, a, b, r1, r2 in candidates:
        arcs[a].append((b, r1, r2, cid))

    # ---------- inside 区间 DP（Pareto 前沿） ----------
    # F[i][j]：区间 [i,j) 上“只最大化配对数”后的 (s1,s2) 前沿。
    F: list[list[Optional[_Frontier]]] = [
        [None] * (n + 1) for _ in range(n + 1)
    ]
    for i in range(n + 1):
        F[i][i] = _Frontier.empty()

    def combine(i: int, j: int) -> _Frontier:
        """计算区间 [i,j) 的前沿：跳过 i 或 i 经某条弧与 k 配对。

        每种“首步规则”各自产生一个按残差向量聚合的映射（同向量的不同
        内/外子方案计数相乘后相加，序列取最小），再在规则间按最大配对数
        合并。配对数只取决于规则结构（1 + 两子区间对数），与向量无关。
        """
        # 规则 1：i 未配对。
        f_skip = F[i + 1][j]
        assert f_skip is not None
        best_pairs = f_skip.pairs
        best_map: dict[tuple[int, int], list[Any]] = {
            (s1, s2): [s1, s2, ways, seq] for s1, s2, ways, seq in f_skip.pts
        }

        # 规则 2：i 与 k 配对，内部 [i+1,k) 与外部 [k+1,j) 独立，向量相加。
        for k, r1, r2, cid in arcs[i]:
            if k >= j:
                continue
            fi = F[i + 1][k]
            fo = F[k + 1][j]
            assert fi is not None and fo is not None
            pairs = 1 + fi.pairs + fo.pairs
            if pairs < best_pairs:
                continue
            mp: dict[tuple[int, int], list[Any]] = {}
            for x1, x2, w1, s_in in fi.pts:
                b1, b2 = r1 + x1, r2 + x2
                for y1, y2, w2, s_out in fo.pts:
                    key = (b1 + y1, b2 + y2)
                    ways = w1 * w2
                    seq = [cid] + s_in + s_out
                    p = mp.get(key)
                    if p is None:
                        mp[key] = [key[0], key[1], ways, seq]
                    else:
                        p[2] += ways
                        if seq < p[3]:
                            p[3] = seq
            if pairs > best_pairs:
                best_pairs = pairs
                best_map = mp
            else:
                for key, p in mp.items():
                    q = best_map.get(key)
                    if q is None:
                        best_map[key] = p
                    else:
                        q[2] += p[2]
                        if p[3] < q[3]:
                            q[3] = p[3]

        return _build_frontier(best_pairs, best_map)

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            F[i][j] = combine(i, j)

    root = F[0][n]
    assert root is not None

    # ---------- 前三层目标选点：最大对数（前沿已固定）→ min max → min sum ----------
    def rank(p: list[Any]) -> tuple[int, int]:
        return (max(p[0], p[1]), p[0] + p[1])

    optimal_pts = [p for p in root.pts if rank(p) == min(rank(q) for q in root.pts)]
    total = sum(p[2] for p in optimal_pts)  # 达到前三层目标的全部方案数

    # ---------- outside DP（在前沿向量上传播） ----------
    # Out[i][j]：[i,j) 作为内部子区间出现时，映射 out_vec -> 外部上下文方案数。
    # out_vec = 根方案总残差向量 - 本子区间内部残差向量（恒为非负整数对）。
    Out: list[list[dict[tuple[int, int], int]]] = [
        [{} for _ in range(n + 1)] for _ in range(n + 1)
    ]
    Out[0][n][(0, 0)] = 1

    for length in range(n, 0, -1):
        for h in range(0, n - length + 1):
            m = h + length
            contexts = Out[h][m]
            if not contexts:
                continue
            fr = F[h][m]
            assert fr is not None

            # 跳过规则：父 [h,m) -> 子 [h+1,m)，残差贡献不变。
            fs = F[h + 1][m]
            assert fs is not None
            if fs.pairs == fr.pairs:
                child = Out[h + 1][m]
                for vec, ways in contexts.items():
                    child[vec] = child.get(vec, 0) + ways

            # 配对规则：父 [h,m) 经弧 (h,k) -> 左子 [h+1,k)、右子 [k+1,m)。
            for k, r1, r2, _cid in arcs[h]:
                if k >= m:
                    continue
                fi = F[h + 1][k]
                fo = F[k + 1][m]
                assert fi is not None and fo is not None
                if 1 + fi.pairs + fo.pairs != fr.pairs:
                    continue
                for (v1, v2), ctx_ways in contexts.items():
                    # 下发左子：外部贡献 = 弧 + 右子内部 + 原上下文。
                    left_map = Out[h + 1][k]
                    for y1, y2, w2, _s in fo.pts:
                        vec = (v1 + r1 + y1, v2 + r2 + y2)
                        left_map[vec] = left_map.get(vec, 0) + ctx_ways * w2
                    # 下发右子：外部贡献 = 弧 + 左子内部 + 原上下文。
                    right_map = Out[k + 1][m]
                    for x1, x2, w1, _s in fi.pts:
                        vec = (v1 + r1 + x1, v2 + r2 + x2)
                        right_map[vec] = right_map.get(vec, 0) + ctx_ways * w1

    # ---------- 候选出现次数：仅统计达到前三层目标的方案 ----------
    target_rank = rank(optimal_pts[0])

    def is_target(s1: int, s2: int) -> bool:
        return (max(s1, s2), s1 + s2) == target_rank

    used_count: dict[str, int] = {}
    for cid, a, b, r1, r2 in candidates:
        fi = F[a + 1][b]
        assert fi is not None
        count = 0
        # 弧 (a,b) 作为父区间 [a,m) 的首步；(g1,g2) 为 [a,m) 的外部上下文。
        for m in range(b + 1, n + 1):
            fo = F[b + 1][m]
            fm = F[a][m]
            assert fo is not None and fm is not None
            # 弧分解必须在 [a,m) 上取得最大对数；对数不足的分解即便残差
            # 巧合达标也不属于最优方案（其子点也必被前沿支配）。
            if 1 + fi.pairs + fo.pairs != fm.pairs:
                continue
            for (g1, g2), ctx_ways in Out[a][m].items():
                for x1, x2, w_in in ((p[0], p[1], p[2]) for p in fi.pts):
                    for y1, y2, w_out in ((p[0], p[1], p[2]) for p in fo.pts):
                        s1 = g1 + r1 + x1 + y1
                        s2 = g2 + r2 + x2 + y2
                        if is_target(s1, s2):
                            count += ctx_ways * w_in * w_out
        used_count[cid] = count

    required: list[str] = []
    optional: list[str] = []
    never: list[str] = []
    for cid, _a, _b, _r1, _r2 in candidates:
        c = used_count[cid]
        if c == 0:
            never.append(cid)
        elif c == total:
            required.append(cid)
        else:
            optional.append(cid)

    # ---------- 规范方案回溯 ----------
    # 规范序列：所有达到前三层目标的前沿点中最小的 id 序列；
    # 一个确定的匹配对应确定的残差向量，据此定位规范解的目标向量。
    canonical_target_seq = min(p[3] for p in optimal_pts)
    target_s1 = target_s2 = 0
    for p in optimal_pts:
        if p[3] == canonical_target_seq:
            target_s1, target_s2 = p[0], p[1]
            break

    canonical_ids: list[str] = []
    unmatched_idx: list[int] = []

    def choose_step(
        i: int, j: int, need1: int, need2: int
    ) -> tuple[Any, ...]:
        """选择使区间 [i,j) 内部残差恰为 (need1,need2) 的最小 id 序列首步。

        前沿点上已缓存取得该确定向量的最小序列，故只需比较各规则的组合序列；
        配对规则同时给出左右子区间各自必须达到的向量 (x1,x2)、(y1,y2)。
        """
        fr = F[i][j]
        assert fr is not None
        best_seq: Optional[list[str]] = None
        best_step: Optional[tuple[Any, ...]] = None

        def offer(seq: list[str], step: tuple[Any, ...]) -> None:
            nonlocal best_seq, best_step
            if best_seq is None or seq < best_seq:
                best_seq, best_step = seq, step

        # 规则 1：i 未配对，子区间 [i+1,j) 须独立达到同一向量。
        fs = F[i + 1][j]
        assert fs is not None
        if fs.pairs == fr.pairs:
            for s1, s2, _w, seq in fs.pts:
                if s1 == need1 and s2 == need2:
                    offer(seq, ("s",))
                    break

        # 规则 2：i 经弧 (i,k) 配对；弧 + 左子向量 + 右子向量 = 目标向量。
        for k, r1, r2, cid in arcs[i]:
            if k >= j:
                continue
            fi = F[i + 1][k]
            fo = F[k + 1][j]
            assert fi is not None and fo is not None
            if 1 + fi.pairs + fo.pairs != fr.pairs:
                continue
            d1, d2 = need1 - r1, need2 - r2
            for x1, x2, _w1, s_in in fi.pts:
                y1, y2 = d1 - x1, d2 - x2
                if y1 < 0 or y2 < 0:
                    continue
                for q1, q2, _w2, s_out in fo.pts:
                    if q1 == y1 and q2 == y2:
                        offer(
                            [cid] + s_in + s_out,
                            ("p", k, cid, x1, x2, y1, y2),
                        )
                        break

        assert best_step is not None, "规范回溯找不到达到目标向量的规则"
        return best_step

    # 栈帧：(i, j, 该区间内部必须达到的残差向量)；先压右子再处理左子。
    stack: list[tuple[int, int, int, int]] = [(0, n, target_s1, target_s2)]
    while stack:
        i, j, need1, need2 = stack.pop()
        while i < j:
            step = choose_step(i, j, need1, need2)
            if step[0] == "s":
                unmatched_idx.append(i)
                i += 1
                # 需要达到的向量不变。
            else:
                k, cid, x1, x2, y1, y2 = step[1:]
                canonical_ids.append(cid)
                # 右子 [k+1,j) 须达到 (y1,y2)，先压栈；
                # 左子 [i+1,k) 须达到 (x1,x2)，在当前循环内继续。
                stack.append((k + 1, j, y1, y2))
                i, j = i + 1, k
                need1, need2 = x1, x2

    info = {cid: (a, b, r1, r2) for cid, a, b, r1, r2 in candidates}
    canonical_pairs = [
        {
            "id": cid,
            "left_endpoint": hits[info[cid][0]]["id"],
            "right_endpoint": hits[info[cid][1]]["id"],
            "residuals": [info[cid][2], info[cid][3]],
        }
        for cid in canonical_ids
    ]

    total_s1 = sum(info[cid][2] for cid in canonical_ids)
    total_s2 = sum(info[cid][3] for cid in canonical_ids)

    return {
        # 以字符串承载任意精度十进制整数，避免客户端 JSON 大整数精度损失。
        "optimal_count": str(total),
        "paired_hits": 2 * root.pairs,
        "residual_totals": [total_s1, total_s2],
        "max_residual_total": max(total_s1, total_s2),
        "canonical_pairs": canonical_pairs,
        "unmatched_hits": [hits[k]["id"] for k in unmatched_idx],
        "classification": {
            "required": sorted(required),
            "optional": sorted(optional),
            "never": sorted(never),
        },
    }
