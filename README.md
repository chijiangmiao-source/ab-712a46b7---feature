# 硅微条击中配对审计服务

把一列按位置严格递增的硅微条击中还原为互不交叉的粒子径迹配对，并在全部
最优方案上做审计：方案计数、规范解、未配对击中与每条候选对的必选/可选/从不分类。

服务提供两个审计入口：

| 入口 | 候选残差字段 | 适用场景 |
| --- | --- | --- |
| `POST /audit` | `residual`（单个非负整数） | 单一增益标定 |
| `POST /audit/robust` | `residuals: [r1, r2]`（两个非负整数） | 两种增益标定下选出**同一套**非交叉配对 |

## 问题与目标

- 输入：4–180 个击中（`id` 唯一、`position` 严格递增），至多 4000 条候选配对
  （`id` 唯一、端点对不重复；`/audit` 用 `residual`，`/audit/robust` 用
  恰好两项的 `residuals`，均为非负整数）。
- 选中的配对必须端点互异，按位置绘制后两两不交叉（允许嵌套）。
- 优化顺序（字典序）：
  1. 最大化已配对击中数（即最大化配对对数）；
  2. `/audit`：最小化残差总和；
     `/audit/robust`：先最小化两工况残差总和的较大者 `max(S1, S2)`，
     再最小化两者之和 `S1 + S2`；
  3. 以“左端位置顺序下的配对 id 序列”字典序最小者为规范解。
- 审计输出：
  - `optimal_count`：任意精度十进制（字符串承载），达到优化目标前若干层的方案数；
  - `canonical_pairs`：规范配对（按左端位置顺序；`/audit` 含 `residual`，
    `/audit/robust` 含 `residuals: [r1, r2]`）；
  - `unmatched_hits`：规范解中的未配对击中；
  - `/audit/robust` 额外返回 `residual_totals: [S1, S2]` 与
    `max_residual_total`（规范方案两工况残差总和及其较大者）；
  - `classification`：依据达到优化目标前若干层的全部方案给出
    `required` / `optional` / `never`（鲁棒入口依据达到“最大配对数 +
    min max + min sum”前三层目标的全部方案判定）。
- 空候选列表合法，返回计数 1 的唯一空方案。
- 重复端点对、未知端点、位置不严格递增/冲突、标识重复、规模越界、残差字段非法等
  均返回 `400 {"errors": [{"field": "/路径", "message": "..."}]}`，
  错误响应不夹带任何审计字段。两个入口的字段语义互不兼容：`/audit` 拒绝
  `residuals`，`/audit/robust` 拒绝单个 `residual`。

## 算法

区间非交叉匹配（允许嵌套），状态为区间 `[i,j)`：

- inside DP：`跳过 i` 或 `i 与 k 配对`（内部 `[i+1,k)` 与外部 `[k+1,j)` 独立），
  维护最大对数与任意精度方案数，以及字典序最小 id 序列；
- 单工况维护最小残差标量；鲁棒入口在每个区间维护 `(S1, S2)` 的 Pareto 前沿
  （固定最大对数下两方向均不被支配的残差向量，按向量聚合方案数），
  在根区间按 `max → sum` 选点；
- outside DP（inside-outside）：统计每条候选弧出现在多少个最优方案中，
  据此分类 required / optional / never。

复杂度 O(n·|C| + n³)（单工况），鲁棒入口再乘以区间 Pareto 前沿规模
（n ≤ 180，|C| ≤ 4000，最大规模实测数秒）。

## 文件

| 文件 | 说明 |
| --- | --- |
| `solver.py` | 校验 + 区间 DP 求解（`audit` / `audit_robust`，仅标准库） |
| `app.py` | HTTP 服务：`GET /health`、`POST /audit`、`POST /audit/robust`（仅标准库） |
| `verify.py` | 单次复核：pytest、构建检查、API/HTTP 冒烟（含两入口） |
| `tests/` | 单元测试、进程内 HTTP 测试、随机暴力枚举交叉验证（单/双工况） |
| `Dockerfile` | API 镜像定义 |
| `docker-compose.yml` | `api` 服务 + `verify` 复核服务 |

## 运行

```bash
# 默认宿主机端口 8080，可用 HOST_PORT 覆盖
HOST_PORT=9090 docker compose up --build -d api

curl -s http://localhost:9090/health
# {"status":"ready","service":"track-pair-audit"}
```

单工况审计请求示例（`POST /audit`）：

```bash
curl -s -X POST http://localhost:9090/audit \
  -H 'Content-Type: application/json' \
  -d '{
    "hits": [
      {"id":"h0","position":0},{"id":"h1","position":10},
      {"id":"h2","position":20},{"id":"h3","position":30}
    ],
    "candidates": [
      {"id":"a_out","left_endpoint":"h0","right_endpoint":"h3","residual":1},
      {"id":"a_in","left_endpoint":"h1","right_endpoint":"h2","residual":5},
      {"id":"b_left","left_endpoint":"h0","right_endpoint":"h1","residual":3},
      {"id":"b_right","left_endpoint":"h2","right_endpoint":"h3","residual":3}
    ]
  }'
```

鲁棒审计请求示例（`POST /audit/robust`，每条候选同时提交两种增益标定残差）：

```bash
curl -s -X POST http://localhost:9090/audit/robust \
  -H 'Content-Type: application/json' \
  -d '{
    "hits": [
      {"id":"h0","position":0},{"id":"h1","position":10},
      {"id":"h2","position":20},{"id":"h3","position":30}
    ],
    "candidates": [
      {"id":"a_out","left_endpoint":"h0","right_endpoint":"h3","residuals":[1,10]},
      {"id":"a_in","left_endpoint":"h1","right_endpoint":"h2","residuals":[1,10]},
      {"id":"b_left","left_endpoint":"h0","right_endpoint":"h1","residuals":[10,1]},
      {"id":"b_right","left_endpoint":"h2","right_endpoint":"h3","residuals":[10,1]}
    ]
  }'
```

## 复核（verify 单次服务）

```bash
docker compose build && docker compose run --rm verify
```

`verify` 服务等待 `api` 健康后依次执行：

1. 代码测试（含随机输入与暴力枚举的方案数/规范解/分类对照，覆盖单、双工况）；
2. 构建检查（语法编译、模块导入、镜像内关键文件齐备）；
3. API/HTTP 冒烟：
   - 原入口回归：健康路径、嵌套同优、交叉低价诱饵、空候选、非法引用、
     重复端点对、位置冲突、规模越界、未知路径；
   - 鲁棒入口：两工况各自偏好的替代解、均衡解压制极端解、嵌套可兼容解、
     并列归属（必选/可选/从不）、双残差字段非法、两入口字段语义隔离。

全部通过退出码 0，任一失败非零。

## 本地开发

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
PORT=8080 python app.py
```
