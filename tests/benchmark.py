"""
tests/benchmark.py
压测与性能基线采集 —— 基于 Locust 的负载测试脚本。

使用方式:
    # 1. 确保服务已启动
    conda activate rec_agent
    uvicorn app.main:app --host 0.0.0.0 --port 8000

    # 2. Web UI 模式（可视化）
    cd <project_root>
    locust -f tests/benchmark.py --host http://localhost:8000

    # 3. 无头模式（CI 场景）—— 10 并发用户，每秒增加 2 个，持续 60 秒
    locust -f tests/benchmark.py --host http://localhost:8000 \
        --headless -u 10 -r 2 -t 60s \
        --csv=logs/benchmark

    # 4. 多 worker 模式（分布式压测）
    locust -f tests/benchmark.py --host http://localhost:8000 --master
    locust -f tests/benchmark.py --host http://localhost:8000 --worker

    # 5. 执行后自动分析节点耗时瓶颈
    python tests/benchmark.py --analyze-logs

功能:
    - 模拟 60% 导购、20% 问答、20% 闲聊三种用户行为
    - 采集 QPS、P50/P95/P99 延迟、错误率
    - 解析应用日志中 MonitorAgent 的 TRACE_REPORT，识别各节点耗时瓶颈
    - 输出结构化性能报告
"""

import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from locust import HttpUser, between, events, task

# ─────────────────── 测试用例集 ───────────────────

SHOPPING_QUERIES = [
    {"query": "推荐一款3000元左右的手机，拍照好的", "user_id": "bench_user_01"},
    {"query": "想买个5000元以内的微单相机，主要拍风景", "user_id": "bench_user_02"},
    {"query": "帮我找一双适合跑步的运动鞋，预算500以内", "user_id": "bench_user_03"},
    {"query": "推荐一台性价比高的笔记本电脑，8000预算", "user_id": "bench_user_04"},
    {"query": "想给女朋友买个包，预算2000左右", "user_id": "bench_user_05"},
    {"query": "家用投影仪推荐，3000元以内", "user_id": "bench_user_06"},
    {"query": "推荐一款降噪耳机，通勤用", "user_id": "bench_user_07"},
    {"query": "想买个扫地机器人，预算2000", "user_id": "bench_user_08"},
    {"query": "推荐适合学生的平板电脑", "user_id": "bench_user_09"},
    {"query": "想买套护肤品，干性皮肤适合的", "user_id": "bench_user_10"},
]

QA_QUERIES = [
    {
        "query": "Sony A7M4 的夜拍效果怎么样？",
        "user_id": "bench_user_11",
        "selected_product_id": "P001",
    },
    {"query": "iPhone 16 和 Pixel 9 哪个拍照更好？", "user_id": "bench_user_12"},
    {"query": "机械键盘和薄膜键盘哪个更适合办公？", "user_id": "bench_user_13"},
    {"query": "4K 显示器对眼睛好还是 2K 好？", "user_id": "bench_user_14"},
    {"query": "无线充电会不会伤电池？", "user_id": "bench_user_15"},
]

CHAT_QUERIES = [
    {"query": "你好，今天天气不错", "user_id": "bench_user_16"},
    {"query": "你能做些什么？", "user_id": "bench_user_17"},
    {"query": "哈哈谢谢你的推荐", "user_id": "bench_user_18"},
    {"query": "没什么想买的，就随便逛逛", "user_id": "bench_user_19"},
    {"query": "你是谁开发的？", "user_id": "bench_user_20"},
]


# ─────────────────── Locust 用户行为定义 ───────────────────


class RecAgentUser(HttpUser):
    """
    模拟电商导购用户的负载测试行为。

    行为分布: 60% 导购推荐 / 20% 商品问答 / 20% 闲聊
    通过 @task 装饰器的权重参数控制各行为的调用比例。
    """

    wait_time = between(1, 3)

    def _post_chat(self, payload: dict, request_name: str):
        """发送 /api/chat 请求并校验响应。"""
        with self.client.post(
            "/api/chat",
            json=payload,
            name=request_name,
            catch_response=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return

            try:
                body = resp.json()
            except Exception:  # noqa: BLE001
                resp.failure("响应体非 JSON")
                return

            if not body.get("response"):
                resp.failure("response 字段为空")
                return

            resp.success()

    @task(6)
    def shopping_query(self):
        """导购推荐请求（权重 6 → 60%）。"""
        import random

        case = random.choice(SHOPPING_QUERIES)
        self._post_chat(
            {"query": case["query"], "user_id": case["user_id"]},
            "/api/chat [导购]",
        )

    @task(2)
    def qa_query(self):
        """商品问答请求（权重 2 → 20%）。"""
        import random

        case = random.choice(QA_QUERIES)
        payload = {"query": case["query"], "user_id": case["user_id"]}
        if "selected_product_id" in case:
            payload["selected_product_id"] = case["selected_product_id"]
        self._post_chat(payload, "/api/chat [问答]")

    @task(2)
    def chat_query(self):
        """闲聊请求（权重 2 → 20%）。"""
        import random

        case = random.choice(CHAT_QUERIES)
        self._post_chat(
            {"query": case["query"], "user_id": case["user_id"]},
            "/api/chat [闲聊]",
        )


class RecAgentStreamUser(HttpUser):
    """
    流式接口压测用户 —— 验证 SSE 端点的吞吐与稳定性。

    独立定义是因为流式请求的响应处理逻辑不同（需逐行解析 SSE 事件），
    可通过 locust 的 --tags 参数单独运行：
        locust -f tests/benchmark.py --host http://localhost:8000 --tags stream
    """

    wait_time = between(2, 5)

    def _post_stream(self, payload: dict, request_name: str):
        """发送 /api/chat/stream 请求，解析 SSE 事件。"""
        start_ts = time.time()
        first_token_ts = None
        token_count = 0
        got_done = False

        try:
            with self.client.post(
                "/api/chat/stream",
                json=payload,
                name=request_name,
                stream=True,
                catch_response=True,
                timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"HTTP {resp.status_code}")
                    return

                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8") if isinstance(line, bytes) else line

                    if decoded.startswith("data:"):
                        if first_token_ts is None:
                            first_token_ts = time.time()
                        token_count += 1

                    if decoded.startswith("event: done") or '"event":"done"' in decoded:
                        got_done = True

                if got_done:
                    resp.success()
                else:
                    resp.failure("未收到 done 事件")

        except Exception as e:
            elapsed_ms = (time.time() - start_ts) * 1000
            events.request.fire(
                request_type="POST",
                name=request_name,
                response_time=elapsed_ms,
                response_length=0,
                exception=e,
            )

        ttft = round((first_token_ts - start_ts) * 1000, 1) if first_token_ts else 0
        if ttft > 0:
            events.request.fire(
                request_type="METRIC",
                name=f"{request_name} [TTFT]",
                response_time=ttft,
                response_length=token_count,
                exception=None,
            )

    @task
    def stream_shopping(self):
        import random

        case = random.choice(SHOPPING_QUERIES)
        self._post_stream(
            {"query": case["query"], "user_id": case["user_id"]},
            "/api/chat/stream [导购]",
        )


# ─────────────────── 日志分析：提取 node_latency_breakdown ───────────────────


def analyze_trace_reports(log_path: str | Path | None = None) -> dict:
    """
    解析应用日志中的 TRACE_REPORT，汇总各节点耗时分布。

    返回格式:
        {
            "total_traces": int,
            "avg_total_latency_ms": float,
            "p50_total_latency_ms": float,
            "p95_total_latency_ms": float,
            "p99_total_latency_ms": float,
            "node_breakdown": {
                "<node_name>": {
                    "count": int,
                    "avg_ms": float,
                    "p50_ms": float,
                    "p95_ms": float,
                    "p99_ms": float,
                    "max_ms": float,
                    "error_rate": float,
                },
                ...
            },
            "bottleneck": "<最慢节点名称>",
            "intent_distribution": {"search": N, "qa": N, "chat": N, ...},
        }
    """
    if log_path is None:
        project_root = Path(__file__).parent.parent
        log_path = project_root / "logs" / "app.log"
    else:
        log_path = Path(log_path)

    if not log_path.exists():
        print(f"[WARN] 日志文件不存在: {log_path}")
        return {}

    trace_pattern = re.compile(r"TRACE_REPORT \| (.+)$")

    total_latencies: list[float] = []
    node_latencies: dict[str, list[float]] = defaultdict(list)
    node_errors: dict[str, int] = defaultdict(int)
    node_counts: dict[str, int] = defaultdict(int)
    intent_dist: dict[str, int] = defaultdict(int)

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = trace_pattern.search(line)
            if not m:
                continue

            try:
                report = json.loads(m.group(1))
            except json.JSONDecodeError:  # noqa: BLE001
                continue

            total_latencies.append(report.get("total_latency_ms", 0))

            intent = report.get("user_intent", "unknown")
            intent_dist[intent] += 1

            for node_info in report.get("node_latency_breakdown", []):
                name = node_info.get("node", "unknown")
                latency = node_info.get("latency_ms", 0)
                success = node_info.get("success", True)

                node_latencies[name].append(latency)
                node_counts[name] += 1
                if not success:
                    node_errors[name] += 1

    if not total_latencies:
        print("[WARN] 未找到任何 TRACE_REPORT 记录")
        return {}

    def _percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    node_breakdown = {}
    for name, latencies in node_latencies.items():
        count = node_counts[name]
        errors = node_errors.get(name, 0)
        node_breakdown[name] = {
            "count": count,
            "avg_ms": round(statistics.mean(latencies), 1),
            "p50_ms": round(_percentile(latencies, 50), 1),
            "p95_ms": round(_percentile(latencies, 95), 1),
            "p99_ms": round(_percentile(latencies, 99), 1),
            "max_ms": round(max(latencies), 1),
            "error_rate": round(errors / count, 4) if count else 0.0,
        }

    bottleneck = max(node_breakdown, key=lambda n: node_breakdown[n]["p95_ms"]) if node_breakdown else "-"

    return {
        "total_traces": len(total_latencies),
        "avg_total_latency_ms": round(statistics.mean(total_latencies), 1),
        "p50_total_latency_ms": round(_percentile(total_latencies, 50), 1),
        "p95_total_latency_ms": round(_percentile(total_latencies, 95), 1),
        "p99_total_latency_ms": round(_percentile(total_latencies, 99), 1),
        "node_breakdown": node_breakdown,
        "bottleneck": bottleneck,
        "intent_distribution": dict(intent_dist),
    }


def print_benchmark_report(analysis: dict) -> None:
    """输出可读的性能基线报告。"""
    if not analysis:
        print("[ERROR] 无分析数据可展示")
        return

    print("\n" + "=" * 90)
    print("  性能基线报告 (基于 MonitorAgent TRACE_REPORT)")
    print("=" * 90)

    print(f"\n  采样请求数: {analysis['total_traces']}")
    print(f"  意图分布: {json.dumps(analysis.get('intent_distribution', {}), ensure_ascii=False)}")

    print("\n  ── 全链路延迟 ──")
    print(f"    平均: {analysis['avg_total_latency_ms']:.1f} ms")
    print(f"    P50:  {analysis['p50_total_latency_ms']:.1f} ms")
    print(f"    P95:  {analysis['p95_total_latency_ms']:.1f} ms")
    print(f"    P99:  {analysis['p99_total_latency_ms']:.1f} ms")

    print("\n  ── 各节点耗时细分 ──")
    print(f"  {'节点':<25} {'调用数':>6} {'平均':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'最大':>8} {'错误率':>8}")
    print("  " + "-" * 85)

    breakdown = analysis.get("node_breakdown", {})
    sorted_nodes = sorted(breakdown.items(), key=lambda x: x[1]["p95_ms"], reverse=True)

    for name, stats in sorted_nodes:
        is_bottleneck = " ◀" if name == analysis.get("bottleneck") else ""
        print(
            f"  {name:<25} {stats['count']:>6} "
            f"{stats['avg_ms']:>7.1f} {stats['p50_ms']:>7.1f} "
            f"{stats['p95_ms']:>7.1f} {stats['p99_ms']:>7.1f} "
            f"{stats['max_ms']:>7.1f} {stats['error_rate']:>7.2%}"
            f"{is_bottleneck}"
        )

    print("\n  " + "-" * 85)
    print(f"  绝对瓶颈节点: {analysis.get('bottleneck', '-')}")
    bottleneck_name = analysis.get("bottleneck", "")
    if bottleneck_name in breakdown:
        bn = breakdown[bottleneck_name]
        print(f"    → P95 = {bn['p95_ms']:.1f} ms, 平均 = {bn['avg_ms']:.1f} ms")

    print("\n" + "=" * 90)


def save_benchmark_report(analysis: dict, output_path: str | Path | None = None) -> Path:
    """将性能报告保存为 JSON 文件。"""
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "logs" / "benchmark_report.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "benchmark_baseline",
        **analysis,
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  报告已保存至: {output_path}")
    return output_path


# ─────────────────── Locust 事件钩子：测试结束后自动分析 ───────────────────


@events.quitting.add_listener
def on_quitting(environment, **_kwargs):
    """Locust 测试结束后，自动分析应用日志并输出节点耗时报告。"""
    print("\n\n" + "=" * 90)
    print("  Locust 压测完毕，开始分析 MonitorAgent 日志...")
    print("=" * 90)

    analysis = analyze_trace_reports()
    if analysis:
        print_benchmark_report(analysis)
        save_benchmark_report(analysis)

    stats = environment.runner.stats
    if stats.total:
        total = stats.total
        print("\n  ── Locust 汇总 ──")
        print(f"    总请求数:   {total.num_requests}")
        print(f"    失败请求数: {total.num_failures}")
        print(f"    错误率:     {total.fail_ratio:.2%}")
        print(f"    平均 RPS:   {total.total_rps:.2f}")
        print(f"    平均延迟:   {total.avg_response_time:.1f} ms")
        if total.get_response_time_percentile(0.5):
            print(f"    P50 延迟:   {total.get_response_time_percentile(0.5):.1f} ms")
        if total.get_response_time_percentile(0.95):
            print(f"    P95 延迟:   {total.get_response_time_percentile(0.95):.1f} ms")
        if total.get_response_time_percentile(0.99):
            print(f"    P99 延迟:   {total.get_response_time_percentile(0.99):.1f} ms")


# ─────────────────── 独立运行：仅做日志分析 ───────────────────


def main():
    """独立运行模式：解析已有日志，输出性能报告。"""
    import argparse

    parser = argparse.ArgumentParser(description="压测性能基线分析")
    parser.add_argument(
        "--analyze-logs",
        action="store_true",
        help="仅分析现有日志，不启动 Locust 压测",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="指定日志文件路径（默认: logs/app.log）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="性能报告输出路径（默认: logs/benchmark_report.json）",
    )
    args = parser.parse_args()

    if args.analyze_logs:
        analysis = analyze_trace_reports(args.log_file)
        if analysis:
            print_benchmark_report(analysis)
            save_benchmark_report(analysis, args.output)
        else:
            print("[ERROR] 无法分析日志，请确保服务已运行并产生过请求日志")
            sys.exit(1)
    else:
        print("用法:")
        print("  1. 启动 Locust 压测:")
        print("     locust -f tests/benchmark.py --host http://localhost:8000")
        print()
        print("  2. 无头模式 (CI):")
        print("     locust -f tests/benchmark.py --host http://localhost:8000 "
              "--headless -u 10 -r 2 -t 60s --csv=logs/benchmark")
        print()
        print("  3. 仅分析已有日志:")
        print("     python tests/benchmark.py --analyze-logs")
        print()
        print("  4. 多 worker 分布式压测:")
        print("     locust -f tests/benchmark.py --host http://localhost:8000 --master")
        print("     locust -f tests/benchmark.py --host http://localhost:8000 --worker")


if __name__ == "__main__":
    main()
