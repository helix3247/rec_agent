# E-Commerce Recommendation Agent (AI 导购智能体)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-orange)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-INCLUDED-blue)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

> 生产级 AI 导购智能体。基于 LangGraph 11 节点编排、多模型智能路由、混合检索 RAG、完整可靠性保障，打造"金牌导购员"对话体验。

---

## 📖 项目简介

Rec Agent 是一个**工程化的、可观测的、有生产思维的** AI 应用参考实现。不仅演示如何用 LLM 做电商推荐，更重要的是展示**完整的系统工程设计**：

### 用户故事
用户模糊地说"推荐几款适合拍夜景的复古相机，预算 5000 块"，系统能够：
1. **自动理解意图**：意图识别 → 意图=search，槽位={预算:5000,类别:相机,用途:拍夜景,风格:复古}
2. **澄清缺失信息**：如果预算不明确，自动提问"您的预算范围是多少?"
3. **精准搜索推荐**：ES 混合检索（BM25 + 向量）+ 用户画像重排 → 返回 3-5 个候选
4. **实时反思修正**：如果推荐结果超出预算或不符合需求，Reflector 触发重试或给出预算调整建议
5. **深度问答支持**：用户问"A7M4 夜拍效果怎样?"，从 Milvus 知识库检索评论/FAQ，RAG 生成基于证据的回答
6. **多轮记忆**：长期记忆自动迁移（Redis → Milvus），下次对话自动加载用户历史偏好

### 工程亮点

| 特性 | 实现 | 价值 |
|------|------|------|
| **LangGraph 编排** | 11 个节点（IntentParser → Dispatcher → 5 个专家 Agent → Reflector → ResponseFormatter → Monitor） | 清晰的执行流，条件边路由，可循环反思 |
| **多模型智能路由** | SmartModelRouter 按复杂度（LIGHT/MEDIUM/HEAVY）选择 primary/fallback 模型 | 降低成本，性能自适应 |
| **混合检索** | ES（BM25 + KNN）+ Milvus（语义）+ 结构化过滤 + 用户画像重排 | 兼顾准确率与召回，个性化推荐 |
| **生产级可靠性** | 熔断器、重试退避、幂等保护、7 层降级链 | 99%+ 可用性，故障自愈 |
| **完整可观测** | trace_id 全链路，节点级指标，Langfuse 集成，JSON 结构化日志 | 快速故障定位，性能优化有据可查 |
| **评测闭环** | E2E 模拟 + LLM Judge + Ragas（Faithfulness/ResponseRelevancy） | 持续质量保证 |

---

## 🛠️ 技术栈

### AI & LLM
- **主模型**：DeepSeek（成本优化）/ OpenAI GPT-4o（fallback，更强推理）
- **框架**：LangChain 0.1+，LangGraph（多智能体编排）
- **Embedding**：OpenAI text-embedding-3-large（3072维）

### 后端与数据
- **API 框架**：FastAPI（异步）
- **向量数据库**：Milvus 2.x（Hybrid Search）
- **搜索引擎**：Elasticsearch 8.x（倒排索引 + 向量检索）
- **关系数据库**：MySQL 8.0（用户/订单/画像）
- **缓存**：Redis（会话历史、长期记忆迁移）

### 可观测性与评测
- **链路追踪**：Langfuse
- **日志**：JSON 结构化日志，可通过 trace_id 串联
- **评测**：pytest + Ragas（RAG 质量评测）+ E2E 模拟

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-username/rec_agent.git
cd rec_agent

# 创建 conda 环境（推荐）
conda create -n rec_agent python=3.10
conda activate rec_agent

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env，填入以下关键配置：
# - LLM_API_KEY（DeepSeek 或 OpenAI）
# - FALLBACK_LLM_API_KEY（备用模型）
# - EMBEDDING_API_KEY（Embedding 模型）
# - Langfuse 相关配置（可选，用于可观测性）
```

### 3. 启动基础设施

```bash
# 一键启动 MySQL、ES、Milvus、Redis
docker-compose up -d

# 验证服务就绪
docker-compose ps
```

### 4. 数据初始化

```bash
# 1. 生成 Mock 数据（相机 + 男装品类，80+ 商品 + 用户评论）
python scripts/generate_mock_data.py

# 2. 初始化 MySQL 表结构并插入数据
python scripts/init_mysql.py

# 3. 向量化并同步到 ES 和 Milvus
# ⚠️ 此步会调用 Embedding API，消耗 token
python scripts/sync_to_vector_db.py

# 可选：仅同步 ES 或 Milvus
python scripts/sync_to_vector_db.py --target es
python scripts/sync_to_vector_db.py --target milvus
```

### 5. 启动服务

```bash
# 启动 FastAPI 服务
uvicorn app.main:app --reload --port 8000

# 或使用 gunicorn 生产启动（4 个 worker）
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 6. 测试 API

```bash
# 简单导购查询
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "推荐几款适合拍夜景的复古相机，预算 5000 块",
    "user_id": "user_123",
    "thread_id": "conv_001"
  }'

# 响应示例：
# {
#   "response": "为您推荐以下相机...",
#   "trace_id": "trace-abc123def456",
#   "thread_id": "conv_001",
#   "candidates": [
#     {"product_id": "P001", "title": "Sony A7M4", "price": 4999, "reason": "..."},
#     ...
#   ],
#   "suggested_questions": ["A7M4 的夜拍效果怎样?", "有没有其他价位的选择?"]
# }

# 多轮对话（带 thread_id）
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "A7M4 的夜拍效果怎样?",
    "user_id": "user_123",
    "thread_id": "conv_001",
    "selected_product_id": "P001"
  }'

# 结束会话并迁移长期记忆
curl -X POST "http://localhost:8000/api/chat/end-session?thread_id=conv_001&user_id=user_123"
```

### 7. 运行测试与评测

```bash
# 单元测试（82 个测试用例）
pytest tests/ -v

# 查看覆盖率
pytest tests/ --cov=app --cov-report=html

# E2E 模拟测试
python tests/e2e_simulator.py

# RAG 质量评测（Ragas）
python tests/eval_rag.py
```

---

## 📂 目录结构

```
rec_agent/
├── app/
│   ├── main.py                       # FastAPI 应用入口 + lifespan
│   ├── graph.py                      # LangGraph 工作流编排
│   ├── state.py                      # AgentState 全局状态定义
│   ├── api/
│   │   └── endpoints/
│   │       └── chat.py               # POST /api/chat 接口
│   ├── agents/                       # 11 个 Agent 节点
│   │   ├── intent_parser.py          # 意图识别 + 槽位抽取
│   │   ├── dispatcher.py             # 路由决策
│   │   ├── shopping.py               # 导购推荐
│   │   ├── outfit.py                 # 穿搭推荐
│   │   ├── rag.py                    # 商品知识问答
│   │   ├── dialog.py                 # 多轮对话/澄清
│   │   ├── tool_call.py              # 工具调用（订单查询等）
│   │   ├── planner.py                # 任务规划与拆解
│   │   ├── reflector.py              # 反思与质量校验
│   │   ├── response_formatter.py     # 响应润色 + 推荐问题
│   │   ├── monitor.py                # 全链路指标汇总
│   │   └── fallback.py               # SmartModelRouter 模型路由
│   ├── core/                         # 核心基础设施
│   │   ├── config.py                 # 配置管理 + 校验
│   │   ├── llm.py                    # LLM 客户端工厂
│   │   ├── reliability.py            # 重试、熔断、幂等、超时
│   │   ├── metrics.py                # 节点指标采集
│   │   ├── logger.py                 # 结构化日志
│   │   ├── langfuse_integration.py   # Langfuse 集成
│   │   └── db_pool.py                # MySQL 连接池
│   ├── tools/                        # 工具层（检索、查询、重排）
│   │   ├── search.py                 # ES Hybrid Search
│   │   ├── knowledge.py              # Milvus 知识库检索
│   │   ├── db.py                     # MySQL 用户画像/订单
│   │   ├── personalization.py        # 用户画像重排
│   │   └── memory.py                 # 长期记忆（Redis → Milvus）
│   ├── prompts/                      # 各 Agent 的 System Prompt
│   ├── models/                       # 数据模型（Intent、State 等）
│   └── utils/                        # 工具函数
├── tests/
│   ├── e2e_simulator.py              # 端到端模拟 + LLM Judge
│   ├── eval_rag.py                   # RAG 质量评测（Ragas）
│   ├── test_*.py                     # 单元测试（82 个用例）
│   ├── verify_metrics.py             # Monitor 指标验证
│   ├── pytest.ini                    # pytest 配置
│   └── conftest.py                   # pytest fixtures
├── scripts/
│   ├── config.py                     # 脚本配置
│   ├── generate_mock_data.py         # 生成 Mock 数据
│   ├── init_mysql.py                 # MySQL 初始化
│   ├── sync_to_vector_db.py          # 数据向量化同步
│   └── synonyms.txt                  # ES 同义词字典（可选）
├── .docs/
│   ├── 技术方案.md                   # 总体技术方案
│   ├── 需求.md                       # 功能需求文档
│   ├── 差距分析与补充开发计划.md    # 阶段 8.1-8.5 完成情况
│   ├── 面试问题和回答.md             # 大厂面试模拟（50 题）
│   └── 面试补强开发计划.md           # 提升分数的定向改进（9.1-9.7）
├── docker-compose.yml                # 基础设施编排
├── requirements.txt                  # Python 依赖
├── .env.example                      # 环境变量示例
├── .gitignore
└── README.md
```

---

## 🔗 API 文档

### POST `/api/chat` - 对话接口

**请求**：
```json
{
  "query": "推荐几款适合拍夜景的相机",          // 必需
  "user_id": "user_123",                        // 可选，用于画像和日志
  "thread_id": "conv_001",                      // 可选，多轮对话会话 ID
  "selected_product_id": "P001"                 // 可选，指定商品 ID 用于 RAG
}
```

**响应**：
```json
{
  "response": "为您推荐以下相机...",
  "trace_id": "trace-abc123",
  "thread_id": "conv_001",
  "candidates": [
    {
      "product_id": "P001",
      "title": "Sony A7M4",
      "price": 4999,
      "reason": "4500 像素、ISO 12800 弱光对焦，适合夜拍"
    }
  ],
  "suggested_questions": [
    "A7M4 的夜拍效果怎样?",
    "有没有其他价位的选择?"
  ]
}
```

### POST `/api/chat/end-session` - 结束会话

结束会话并将对话历史迁移到长期记忆（Milvus）。

**请求**：`POST /api/chat/end-session?thread_id=conv_001&user_id=user_123`

**响应**：
```json
{
  "success": true,
  "message": "会话记忆已迁移到长期记忆"
}
```

### GET `/health` - 健康检查

查看各服务连通性和模型路由健康状态。

**响应示例**：
```json
{
  "status": "ok",
  "app": "Rec Agent",
  "models": {
    "primary": {
      "healthy": true,
      "error_rate": 0.01,
      "avg_latency_ms": 1200
    },
    "fallback": {
      "healthy": true,
      "error_rate": 0.05,
      "avg_latency_ms": 2100
    }
  },
  "mysql_pool": {
    "size": 8,
    "idle": 5,
    "active": 3
  }
}
```

---

## 📊 系统指标与可观测性

### 关键指标

| 指标 | 来源 | 含义 |
|------|------|------|
| `total_latency_ms` | MonitorAgent | 单次请求端到端延迟 |
| `node_latency_breakdown` | MonitorAgent | 各节点耗时细分（用于定位慢节点） |
| `token_usage` | MonitorAgent | 本次请求消耗的 token（prompt/completion 分开） |
| `tool_call_stats` | MonitorAgent | 工具调用成功率及失败列表 |
| `model_routing` | MonitorAgent | 使用的模型及其健康状态 |
| `error_rate` | SmartModelRouter | 模型的近期错误率（滑动窗口） |
| `consecutive_failures` | SmartModelRouter | 连续失败次数（触发熔断） |

### Langfuse 集成

所有请求自动上报到 Langfuse，包含：
- 完整的 trace（请求 → 各节点 → 响应）
- Token 消耗（用于成本分析）
- 模型路由决策（用于优化）
- 异常和降级事件

启用方式：在 `.env` 中配置 `LANGFUSE_ENABLED=true` 和密钥。

---

## 🧪 测试与评测

### 单元测试

```bash
# 运行全部 82 个测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_tools.py -v
pytest tests/test_reliability.py -v

# 查看覆盖率
pytest tests/ --cov=app --cov-report=term-missing
```

### E2E 模拟

```bash
# 运行 8 个预定义场景（导购/穿搭/RAG/对比/规划/闲聊/澄清/多轮）
python tests/e2e_simulator.py

# 输出：用例通过率、各维度评分（clarity_first/enough_candidates/evidence_based/has_suggestions）
```

### RAG 质量评测

```bash
# 使用 Ragas 框架评测 RAG 指标
python tests/eval_rag.py

# 输出：Faithfulness（忠实度）、ResponseRelevancy（相关性）
```

---

## 🎓 学习路径

如果你想深入理解这个项目的设计思想，建议按以下顺序阅读：

1. **`.docs/技术方案.md`** — 整体架构和设计思路
2. **`.docs/需求.md`** — 功能需求文档
3. **`app/graph.py`** — LangGraph 工作流编排（代码的骨架）
4. **`app/agents/`** — 各 Agent 的具体实现
5. **`app/core/reliability.py`** — 生产级可靠性的实现
6. **`.docs/面试问题和回答.md`** — 大厂面试场景下的深度理解

进阶阅读：
- **`.docs/差距分析与补充开发计划.md`** — 从 0 到 1 的完整开发过程
- **`.docs/面试补强开发计划.md`** — 提升系统的针对性改进方向
- **`tests/`** — 测试套件展示了各模块的使用方式

---

## 🌟 项目亮点

### 从"Demo"到"生产"的工程化设计

| 方面 | Demo | 本项目 |
|------|------|--------|
| 工作流 | 顺序调用 | LangGraph 条件边路由 + 3 层反思 + 规划拆解 |
| LLM | 单一模型 | SmartModelRouter 智能路由（LIGHT/MEDIUM/HEAVY） |
| 错误处理 | try-catch | 熔断器 + 重试退避 + 7 层降级链 |
| 检索 | 单一索引 | ES + Milvus + 结构化过滤 + 用户画像重排 |
| 会话管理 | 内存 | Redis 短期 + Milvus 长期迁移 |
| 观测 | print 日志 | Langfuse + trace_id 全链路 + 节点级指标 |
| 评测 | 手工验证 | E2E 模拟 + LLM Judge + Ragas |

### 可论文化的设计模式

- **SmartModelRouter**：基于任务复杂度的多模型路由，简单任务用便宜模型，复杂任务用强模型
- **Reflector 反思机制**：规则前置检查 + LLM 深度检查 + 策略递进（relax_filter → rewrite_query → clarify）
- **Planner 规划拆解**：COT 思维链拆解复杂需求为子任务，支持循环执行和结果汇总
- **Hybrid Retrieval**：ES（精准） + Milvus（语义） + 用户画像重排，兼顾召回和精准

---

## 📈 性能基准（本地单机）

> 基于 Mock 数据在单机环境的典型表现。实际值会因模型和网络而变化。

| 场景 | QPS | P95 延迟 | P99 延迟 | 平均 Token |
|------|-----|---------|---------|-----------|
| 简单澄清 | 5-8 | 800ms | 1200ms | 500 |
| 导购推荐 | 2-4 | 3000ms | 5000ms | 2000 |
| RAG 问答 | 2-3 | 3500ms | 6000ms | 2500 |
| 穿搭推荐 | 1-2 | 5000ms | 8000ms | 3000 |

**瓶颈分析**：LLM 推理（占 70%）> Embedding API（15%）> 数据库查询（10%）> 日志上报（5%）

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发流程

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 开发并测试：`pytest tests/ -v`
4. 提交变更：`git commit -am "描述你的改进"`
5. 推送到 GitHub：`git push origin feature/your-feature`
6. 提交 Pull Request

### 代码规范

- 遵循 PEP 8（Python）
- 为新功能补充单元测试（目标覆盖率 ≥ 80%）
- 更新 `.docs/` 中的相关文档

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。

---

## 🙋 常见问题

### Q: 这个项目能用于生产吗？
**A**: 架构和可靠性机制（熔断、重试、降级）设计为生产级别，但需要根据实际业务做以下补强：
- 真实数据替代 Mock 数据
- 按实际 QPS 调优连接池和缓存参数
- 接入正式的内容安全审核
- 补充安全防护（输入 sanitize、prompt injection 防护）

详见 [`.docs/面试补强开发计划.md`](.docs/面试补强开发计划.md)。

### Q: 支持中文吗？
**A**: 完全支持。使用的 embedding 模型原生支持中文，ES 可选配置 ik_analyzer 中文分词。

### Q: 如何调试单个 Agent？
**A**: 每个 Agent 都有独立的单元测试。例如，运行 `pytest tests/test_tools.py::test_search_products -v` 只测试搜索模块。

### Q: 成本大概多少？
**A**: 取决于 QPS 和 LLM 选择。以 DeepSeek 为主模型：
- 1000 请求 ≈ ¥10-20（含 embedding）
- 10000 请求/天 ≈ ¥100-200/天

使用 SmartModelRouter 可以进一步降低成本约 30%（轻量任务用更便宜的模型）。

---

**最后更新**：2026-03-08  
**项目完成度**：✅ 100%（核心功能 + 工程质量）  
**适用场景**：电商导购、客服对话、知识问答等实时对话场景
