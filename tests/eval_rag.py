"""
tests/eval_rag.py
RAG 质量评测 —— 使用 Ragas 计算多维度指标。

使用方式:
    conda activate rec_agent
    python tests/eval_rag.py
    python tests/eval_rag.py --cross-judge   # 启用非同源 Judge 模型

评测流程:
    1. 定义测试集（query + contexts + reference answer）
    2. 调用 RAGAgent 获取实际回答
    3. 使用 Ragas 计算四项核心指标:
        - Faithfulness: 回答是否忠实于检索上下文
        - ResponseRelevancy: 回答与问题的相关度
        - ContextPrecision: 检索上下文中相关片段的精确度
        - LLMContextRecall: 参考答案的信息能被检索上下文覆盖的程度
    4. 支持非同源 Judge（primary 生成回答，fallback 做评测）
    5. 输出评测报告
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logger import get_logger

_log = get_logger(agent_name="EvalRAG", trace_id="eval-rag")


# ─────────────────── 评测数据集 ───────────────────

EVAL_DATASET = [
    {
        "user_input": "Sony A7M4 的夜拍效果怎么样？",
        "retrieved_contexts": [
            "[用户评论] A7M4的夜拍表现出色，ISO 12800下噪点控制很好，暗部细节保留丰富。",
            "[用户评论] 夜拍对焦速度很快，弱光下也能精准锁定人眼，拍人像特别好用。",
            "[FAQ] A7M4支持15级动态范围，配合BIONZ XR处理器，暗光拍摄能力显著提升。",
        ],
        "response": "",
        "reference": "Sony A7M4夜拍效果出色，ISO 12800下噪点控制好，暗部细节丰富，弱光对焦快速精准。",
    },
    {
        "user_input": "这款耳机的续航怎么样？",
        "retrieved_contexts": [
            "[用户评论] 续航大概5小时左右，配合充电盒能用到20小时，日常通勤够用。",
            "[用户评论] 开降噪的话续航掉得快一些，大概4小时，关掉降噪能到6小时。",
            "[说明书] 电池容量55mAh（单耳），充电盒400mAh，支持快充，充电10分钟可用1小时。",
        ],
        "response": "",
        "reference": "耳机单次续航约5-6小时，开降噪约4小时，配合充电盒总续航约20小时，支持快充。",
    },
    {
        "user_input": "这个手机壳防摔吗？",
        "retrieved_contexts": [
            "[用户评论] 摔了好几次了，手机壳有点磕碰痕迹但手机完好无损，防摔效果不错。",
            "[用户评论] 四角加厚设计，有气囊缓冲，1.5米跌落测试通过，很放心。",
            "[FAQ] 采用TPU+PC双层材质，通过SGS 1.2米跌落测试认证。",
        ],
        "response": "",
        "reference": "这款手机壳防摔性能好，四角加厚气囊设计，双层材质，通过SGS跌落测试认证。",
    },
    {
        "user_input": "这台洗衣机的噪音大吗？",
        "retrieved_contexts": [
            "[用户评论] 洗涤时噪音很小，基本听不到，脱水时有点声音但可以接受。",
            "[用户评论] DD直驱电机确实安静很多，对比之前的皮带驱动，差别明显。",
            "[说明书] 洗涤噪音≤46dB，脱水噪音≤72dB，采用DD直驱变频电机。",
        ],
        "response": "",
        "reference": "洗衣机噪音控制好，DD直驱电机洗涤噪音≤46dB，脱水≤72dB，日常使用安静。",
    },
    {
        "user_input": "这个背包适合装笔记本电脑吗？",
        "retrieved_contexts": [
            "[用户评论] 有专门的电脑隔层，放15.6寸笔记本完全没问题，还有减震垫。",
            "[用户评论] 我放的14寸MacBook Pro，空间绰绰有余，旁边还能放平板。",
            "[FAQ] 内设独立笔记本隔层，最大可容纳16英寸笔记本，配有EVA防震内衬。",
        ],
        "response": "",
        "reference": "背包有独立电脑隔层，最大容纳16英寸笔记本，带EVA防震内衬，适合装笔记本。",
    },
]


def _generate_rag_responses():
    """调用 RAG 流程为每个测试样本生成回答。"""
    from langchain_core.messages import SystemMessage
    from app.core.llm import get_llm
    from app.prompts.rag import RAG_SYSTEM_PROMPT

    llm = get_llm("primary")

    for sample in EVAL_DATASET:
        chunks_text = "\n---\n".join(sample["retrieved_contexts"])
        system_prompt = RAG_SYSTEM_PROMPT.format(
            query=sample["user_input"],
            product_info="（评测用商品）",
            knowledge_chunks=chunks_text,
        )

        try:
            response = llm.invoke([SystemMessage(content=system_prompt)])
            sample["response"] = response.content
            _log.info("生成回答 | query={} | response_len={}", sample["user_input"][:30], len(sample["response"]))
        except Exception as e:
            _log.error("生成回答失败 | query={} | error={}", sample["user_input"][:30], str(e))
            sample["response"] = "抱歉，无法生成回答。"


def _get_evaluator_llm(model_type: str = "primary"):
    """获取评测用的 LLM，支持指定模型类型以实现非同源 Judge。"""
    from app.core.llm import get_llm
    from ragas.llms import LangchainLLMWrapper

    llm = get_llm(model_type)
    return LangchainLLMWrapper(llm)


def _load_metrics(evaluator_llm):
    """加载 Ragas 评测指标（Faithfulness + ResponseRelevancy + ContextPrecision + LLMContextRecall）。"""
    try:
        from ragas.metrics.collections import (
            Faithfulness, ResponseRelevancy,
            ContextPrecision, LLMContextRecall,
        )
    except ImportError:
        from ragas.metrics import (
            Faithfulness, ResponseRelevancy,
            ContextPrecision, LLMContextRecall,
        )

    return [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]


def _run_ragas_evaluation(judge_model: str = "primary"):
    """使用 Ragas 进行四维度评测。

    Args:
        judge_model: 评测用 LLM 模型类型，"primary" 为同源，"fallback" 为非同源。
    """
    try:
        from ragas import SingleTurnSample, EvaluationDataset, evaluate
    except ImportError:
        _log.error("ragas 未安装，请运行: pip install ragas")
        return None

    samples = []
    for item in EVAL_DATASET:
        samples.append(
            SingleTurnSample(
                user_input=item["user_input"],
                retrieved_contexts=item["retrieved_contexts"],
                response=item["response"],
                reference=item["reference"],
            )
        )

    eval_dataset = EvaluationDataset(samples=samples)

    _log.info("开始 Ragas 评测 | samples={} | judge_model={}", len(samples), judge_model)

    try:
        evaluator_llm = _get_evaluator_llm(judge_model)
        metrics = _load_metrics(evaluator_llm)

        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
        )
        return results
    except Exception as e:
        _log.error("Ragas 评测执行失败 | judge={} | error={}", judge_model, str(e))

        if judge_model != "fallback":
            _log.info("尝试使用 Fallback LLM 进行评测...")
            try:
                fallback_llm = _get_evaluator_llm("fallback")
                metrics = _load_metrics(fallback_llm)
                results = evaluate(
                    dataset=eval_dataset,
                    metrics=metrics,
                )
                return results
            except Exception as e2:
                _log.error("Ragas 评测完全失败 | error={}", str(e2))
                return None
        else:
            _log.error("Ragas 评测完全失败 | error={}", str(e))
            return None


def _print_report(results, label: str = ""):
    """输出评测报告。"""
    title_suffix = f" [{label}]" if label else ""
    print("\n" + "=" * 70)
    print(f"  RAG 质量评测报告 (Ragas){title_suffix}")
    print("=" * 70)

    if results is None:
        print("  [ERROR] 评测失败，请检查依赖和配置")
        print("=" * 70)
        return False, {}

    print(f"\n  评测样本数: {len(EVAL_DATASET)}")
    print(f"  评测指标: Faithfulness, ResponseRelevancy, ContextPrecision, LLMContextRecall")
    print()

    scores = results.scores if hasattr(results, "scores") else results
    avg_scores = {}
    if isinstance(scores, dict):
        avg_scores = scores
        for metric_name, score in scores.items():
            status = "[OK]" if score >= 0.7 else "[WARN]"
            print(f"  {status} {metric_name}: {score:.4f}")
    else:
        print(f"  评测结果: {results!r}")

    if hasattr(results, "to_pandas"):
        print("\n  逐样本得分:")
        print("  " + "-" * 66)
        df = results.to_pandas()
        for idx, row in df.iterrows():
            query = EVAL_DATASET[idx]["user_input"][:35]
            score_parts = []
            for col in df.columns:
                if col not in ("user_input", "response", "retrieved_contexts", "reference"):
                    score_parts.append(f"{col}={row[col]:.3f}" if isinstance(row[col], float) else f"{col}={row[col]}")
            print(f"  [{idx+1}] {query:<35} | {' | '.join(score_parts)}")

    print("\n" + "=" * 70)

    overall_ok = all(v >= 0.6 for v in avg_scores.values()) if avg_scores else True

    if overall_ok:
        print("  [PASS] RAG 评测通过")
    else:
        print("  [FAIL] 部分指标低于阈值 (0.6)，需要优化")

    print("=" * 70)

    return overall_ok, avg_scores


def _save_report(all_scores: dict, cross_scores: dict | None, overall_ok: bool):
    """保存评测报告到 JSON 文件。"""
    report_path = Path(__file__).parent.parent / "logs" / "eval_rag_report.json"
    report_path.parent.mkdir(exist_ok=True)

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": len(EVAL_DATASET),
        "metrics": ["faithfulness", "answer_relevancy", "context_precision", "llm_context_recall"],
        "scores": {k: round(v, 4) for k, v in all_scores.items()} if all_scores else {},
        "passed": overall_ok,
    }

    if cross_scores:
        report_data["cross_judge_scores"] = {k: round(v, 4) for k, v in cross_scores.items()}

    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告已保存至: {report_path}")


def main():
    """运行 RAG 评测流程。"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG 质量评测")
    parser.add_argument("--cross-judge", action="store_true",
                        help="启用非同源 Judge 模式（primary 生成回答，fallback 做评测）")
    args = parser.parse_args()

    cross_judge = args.cross_judge

    metrics_desc = "Faithfulness, ResponseRelevancy, ContextPrecision, LLMContextRecall"
    print("=" * 70)
    print(f"  RAG 质量评测 (Ragas)")
    print(f"  评测指标: {metrics_desc}")
    if cross_judge:
        print("  Judge 模式: 非同源 (primary 生成, fallback 评测)")
    print("=" * 70)

    # Step 1: 生成 RAG 回答（始终用 primary）
    steps = 3 if cross_judge else 2
    print(f"\n[1/{steps}] 生成 RAG 回答...")
    t0 = time.time()
    _generate_rag_responses()
    gen_time = time.time() - t0
    print(f"      完成 ({gen_time:.1f}s)")

    for i, sample in enumerate(EVAL_DATASET, 1):
        print(f"\n  样本 {i}: {sample['user_input']}")
        print(f"  回答:   {sample['response'][:100]}...")

    # Step 2: Ragas 评测（primary judge）
    print(f"\n[2/{steps}] 运行 Ragas 评测 (primary judge)...")
    t1 = time.time()
    results = _run_ragas_evaluation(judge_model="primary")
    eval_time = time.time() - t1
    print(f"      完成 ({eval_time:.1f}s)")

    passed, primary_scores = _print_report(results, label="Primary Judge")

    # Step 3: 非同源 Judge 评测（cross_judge 模式）
    cross_scores = None
    if cross_judge:
        print(f"\n[3/{steps}] 运行 Ragas 评测 (cross judge / fallback)...")
        t2 = time.time()
        cross_results = _run_ragas_evaluation(judge_model="fallback")
        cross_time = time.time() - t2
        print(f"      完成 ({cross_time:.1f}s)")

        _, cross_scores = _print_report(cross_results, label="Cross Judge (fallback)")

        if primary_scores and cross_scores:
            _print_cross_comparison(primary_scores, cross_scores)

    _save_report(primary_scores, cross_scores, passed)

    return passed


def _print_cross_comparison(primary_scores: dict, cross_scores: dict):
    """输出 Primary vs Cross Judge 的对比分析。"""
    print("\n" + "=" * 70)
    print("  Primary vs Cross Judge 对比分析")
    print("=" * 70)
    print(f"\n  {'指标':<30} {'Primary':>10} {'Cross':>10} {'差异':>10}")
    print("  " + "-" * 62)

    diffs = []
    for metric in primary_scores:
        p = primary_scores.get(metric, 0)
        c = cross_scores.get(metric, 0)
        diff = p - c
        diffs.append(abs(diff))
        diff_str = f"{diff:+.4f}"
        print(f"  {metric:<30} {p:>10.4f} {c:>10.4f} {diff_str:>10}")

    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    print(f"\n  平均绝对偏差: {avg_diff:.4f}")
    if avg_diff < 0.05:
        print("  [GOOD] 两个 Judge 评分高度一致，评测结果可信")
    elif avg_diff < 0.15:
        print("  [OK] 两个 Judge 评分基本一致，差异在可接受范围")
    else:
        print("  [WARN] 两个 Judge 评分差异较大，建议检查评测数据质量")

    print("=" * 70)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
