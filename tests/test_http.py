"""HTTP 层测试：在随机端口上线程内启动真实 app，走 urllib 调用。"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from app import AuditHandler


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), AuditHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def request(base, method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health(server):
    status, body = request(server, "GET", "/health")
    assert status == 200
    assert body == {"status": "ready", "service": "track-pair-audit"}


def test_audit_ok(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "p", "left_endpoint": "h0", "right_endpoint": "h3", "residual": 2},
            {"id": "q", "left_endpoint": "h1", "right_endpoint": "h2", "residual": 1},
        ],
    }
    status, body = request(server, "POST", "/audit", payload)
    assert status == 200
    assert body["optimal_count"] == "1"
    assert [p["id"] for p in body["canonical_pairs"]] == ["p", "q"]
    assert body["unmatched_hits"] == []


def test_audit_error_shape_has_no_audit_fields(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "x", "left_endpoint": "h0", "right_endpoint": "nope", "residual": 0}
        ],
    }
    status, body = request(server, "POST", "/audit", payload)
    assert status == 400
    assert set(body.keys()) == {"errors"}
    assert body["errors"][0]["field"] == "/candidates/0/right_endpoint"


def test_bad_json(server):
    req = urllib.request.Request(
        server + "/audit", data=b"{not json", method="POST"
    )
    req.add_header("Content-Type", "application/json")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 400


def test_unknown_route(server):
    status, _ = request(server, "GET", "/")
    assert status == 404


# ---------------------------------------------------------- 鲁棒入口


def test_robust_audit_ok(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {
                "id": "p",
                "left_endpoint": "h0",
                "right_endpoint": "h3",
                "residuals": [2, 7],
            },
            {
                "id": "q",
                "left_endpoint": "h1",
                "right_endpoint": "h2",
                "residuals": [3, 1],
            },
        ],
    }
    status, body = request(server, "POST", "/audit/robust", payload)
    assert status == 200
    assert body["optimal_count"] == "1"
    assert body["residual_totals"] == [5, 8]
    assert body["max_residual_total"] == 8
    assert [p["id"] for p in body["canonical_pairs"]] == ["p", "q"]
    assert body["canonical_pairs"][0]["residuals"] == [2, 7]
    assert body["unmatched_hits"] == []
    assert set(body["classification"]["required"]) == {"p", "q"}


def test_robust_audit_balanced_beats_extremes(server):
    # 两个完美匹配：嵌套极端解 (0,10) 与并列均衡解 (8,8)；
    # 单工况2会偏好极端解，鲁棒目标按 max 必须选择均衡解。
    hits = [{"id": f"h{k}", "position": k} for k in range(4)]
    candidates = [
        {"id": "out", "left_endpoint": "h0", "right_endpoint": "h3", "residuals": [0, 10]},
        {"id": "in", "left_endpoint": "h1", "right_endpoint": "h2", "residuals": [0, 10]},
        {"id": "b01", "left_endpoint": "h0", "right_endpoint": "h1", "residuals": [4, 4]},
        {"id": "b23", "left_endpoint": "h2", "right_endpoint": "h3", "residuals": [4, 4]},
    ]
    status, body = request(
        server, "POST", "/audit/robust", {"hits": hits, "candidates": candidates}
    )
    assert status == 200
    assert body["residual_totals"] == [8, 8]
    assert body["max_residual_total"] == 8
    assert [p["id"] for p in body["canonical_pairs"]] == ["b01", "b23"]
    assert set(body["classification"]["never"]) == {"in", "out"}


def test_robust_audit_error_shape_has_no_audit_fields(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {
                "id": "x",
                "left_endpoint": "h0",
                "right_endpoint": "nope",
                "residuals": [0, 1],
            }
        ],
    }
    status, body = request(server, "POST", "/audit/robust", payload)
    assert status == 400
    assert set(body.keys()) == {"errors"}
    assert body["errors"][0]["field"] == "/candidates/0/right_endpoint"

    # 双残差字段非法（缺项、负值）。
    payload["candidates"] = [
        {"id": "x", "left_endpoint": "h0", "right_endpoint": "h1", "residuals": [1]}
    ]
    status, body = request(server, "POST", "/audit/robust", payload)
    assert status == 400
    assert any(e["field"] == "/candidates/0/residuals" for e in body["errors"])
    assert "optimal_count" not in body

    payload["candidates"] = [
        {"id": "x", "left_endpoint": "h0", "right_endpoint": "h1", "residuals": [1, -2]}
    ]
    status, body = request(server, "POST", "/audit/robust", payload)
    assert status == 400
    assert any(e["field"] == "/candidates/0/residuals/1" for e in body["errors"])


def test_robust_route_distinct_from_audit(server):
    # 同一请求体走单工况入口：缺 residual 时按原语义判非法；
    # 鲁棒入口要求 residuals。两入口字段语义互不串扰。
    hits = [{"id": f"h{k}", "position": k} for k in range(4)]
    robust_only = [
        {
            "id": "p",
            "left_endpoint": "h0",
            "right_endpoint": "h1",
            "residuals": [1, 2],
        }
    ]
    status, body = request(server, "POST", "/audit", {"hits": hits, "candidates": robust_only})
    assert status == 400
    assert any(e["field"] == "/candidates/0/residual" for e in body["errors"])

    single_only = [
        {
            "id": "p",
            "left_endpoint": "h0",
            "right_endpoint": "h1",
            "residual": 3,
        }
    ]
    status, body = request(
        server, "POST", "/audit/robust", {"hits": hits, "candidates": single_only}
    )
    assert status == 400
    assert any(e["field"] == "/candidates/0/residuals" for e in body["errors"])
