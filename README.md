# 硅微条击中配对审计服务

把一列按位置严格递增的硅微条击中还原为互不交叉的粒子径迹配对，并在全部
最优方案上做审计：方案计数、规范解、未配对击中与每条候选对的必选/可选/从不分类。

提供两个审计入口：

* `POST /audit`：单增益工况，每条候选一个 `residual`；
* `POST /audit/robust`：双增益工况鲁棒审计，每条候选同时提交
  `residual_low` / `residual_high` 两项非负整数残差，求解器必须选出
  **同一套**不随工况改变的非交叉配对，而不能对两种标定分别采用各自的
  局部最优解释。

## 问题与目标（单工况 `/audit`）

- 输入：4–180 个击中（`id` 唯一、`position` 严格递增），至多 4000 条候选配对
  （`id` 唯一、端点对不重复、`residual` 为非负整数）。
- 选中的配对必须端点互异，按位置绘制后两两不交叉（允许嵌套）。
- 优化顺序（字典序）：
  1. 最大化已配对击中数（即最大化配对对数）；
  2. 最小化残差总和；
  3. 以“左端位置顺序下的配对 id 序列”字典序最小者为规范解。
- 审计输出：
  - `optimal_count`：任意精度十进制（字符串承载），达到前两级目标的方案数；
  - `canonical_pairs`：规范配对（含端点与残差，按左端位置顺序）；
  - `unmatched_hits`：规范解中的未配对击中；
  - `classification`：依据全部最优方案给出 `required` / `optional` / `never`。
- 空候选列表合法，返回计数 1 的唯一空方案。
- 重复端点对、未知端点、位置不严格递增/冲突、标识重复、规模越界等均返回
  `400 {"errors": [{"field": "/路径", "message": "..."}]}`，错误响应不夹带任何审计字段。

## 鲁棒审计 `/audit/robust`

探测器在两种增益标定下对同一候选径迹给出不同残差。候选字段为
`residual_low`、`residual_high`（均为非负整数），其余击中/候选端点规则与
单工况入口完全一致。

优化顺序（字典序，前三层决定方案计数与候选分类）：

1. 最大化已配对击中数；
2. 最小化 `max(工况一残差总和, 工况二残差总和)`；
3. 最小化两种工况残差总和之和；
4. 以“左端位置顺序下的配对 id 序列”字典序最小者为规范方案。

响应字段：

- `optimal_count`：任意精度十进制（字符串承载），达到前三层目标的方案数；
- `paired_hits`：已配对击中数；
- `total_residual_low` / `total_residual_high`：规范方案的两项残差总和；
- `canonical_pairs`：规范配对，每条含端点与 `residual_low` / `residual_high`；
- `unmatched_hits`、`classification`：语义同单工况入口（分类基于达到
  前三层目标的全部方案）。

`max` 目标不可加，求解器在每个区间、固定最大配对数下维护
`(工况一残差总和, 工况二残差总和)` 的 Pareto 最小点集，根前沿按
`(max, sum, id 序列)` 裁决，再由逐点 outside DP 统计每条候选弧的
最优方案出现次数。

双残差字段缺失/非法（含布尔、浮点、负数）、端点引用非法、规模越界等均
返回字段路径错误，且不混入任何审计结果。原 `/audit` 入口的请求与响应
语义保持不变（只接受单个 `residual`）。

## 算法

区间非交叉匹配（允许嵌套），状态为区间 `[i,j)`：

- inside DP：`跳过 i` 或 `i 与 k 配对`（内部 `[i+1,k)` 与外部 `[k+1,j)` 独立），
  维护最大对数、最小残差、任意精度方案数，以及字典序最小 id 序列；
- outside DP（inside-outside）：统计每条候选弧出现在多少个最优方案中，
  据此分类 required / optional / never。
- 双工况鲁棒审计把单值最小残差替换为 Pareto 最小点集，并做逐点 outside 下发。

复杂度 O(n·|C| + n³)（鲁棒入口再乘以小区间 Pareto 前沿宽度，实测很薄）；
n ≤ 180，|C| ≤ 4000，两种入口最大规模实测均在秒级以内。

## 文件

| 文件 | 说明 |
| --- | --- |
| `solver.py` | 校验 + 区间 DP 求解（单工况 `audit`、双工况 `audit_robust`，仅标准库） |
| `app.py` | HTTP 服务：`GET /health`、`POST /audit`、`POST /audit/robust`（仅标准库） |
| `verify.py` | 单次复核：pytest、构建检查、两个入口的 API/HTTP 冒烟 |
| `tests/` | 单元测试、进程内 HTTP 测试、随机暴力枚举交叉验证 |
| `Dockerfile` | API 镜像定义 |
| `docker-compose.yml` | `api` 服务 + `verify` 复核服务 |

## 运行

```bash
# 默认宿主机端口 8080，可用 HOST_PORT 覆盖
HOST_PORT=9090 docker compose up --build -d api

curl -s http://localhost:9090/health
# {"status":"ready","service":"track-pair-audit"}
```

审计请求示例：

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

鲁棒审计请求示例（每条候选给出两种增益标定下的残差）：

```bash
curl -s -X POST http://localhost:9090/audit/robust \
  -H 'Content-Type: application/json' \
  -d '{
    "hits": [
      {"id":"h0","position":0},{"id":"h1","position":10},
      {"id":"h2","position":20},{"id":"h3","position":30}
    ],
    "candidates": [
      {"id":"p01","left_endpoint":"h0","right_endpoint":"h1","residual_low":50,"residual_high":1},
      {"id":"p23","left_endpoint":"h2","right_endpoint":"h3","residual_low":50,"residual_high":1},
      {"id":"out","left_endpoint":"h0","right_endpoint":"h3","residual_low":1,"residual_high":50},
      {"id":"in","left_endpoint":"h1","right_endpoint":"h2","residual_low":1,"residual_high":50}
    ]
  }'
# 嵌套 [out,in]=(2,100) 与并列 [p01,p23]=(100,2) 的 max 与 sum 均相同，
# 由左端位置顺序 id 序列裁决为 [out, in]，optimal_count 为 2。
```

## 复核（verify 单次服务）

```bash
docker compose build && docker compose run --rm verify
```

`verify` 服务等待 `api` 健康后依次执行：

1. 代码测试（216 项：原 77 项回归 + 鲁棒入口 135 项，后者含 120 组随机
   输入与暴力枚举的方案数/规范解/分类对照）；
2. 构建检查（语法编译、模块导入、镜像内关键文件齐备）；
3. API/HTTP 冒烟：
   - `/audit` 原入口回归（健康路径、嵌套同优、交叉低价诱饵、空候选、
     非法引用、重复端点对、位置冲突、规模越界、未知路径）；
   - `/audit/robust` 覆盖两工况各自偏好的替代解、嵌套可兼容解、并列归属
     与双残差字段/端点/空候选校验。

全部通过退出码 0，任一失败非零。

## 本地开发

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
PORT=8080 python app.py
```
