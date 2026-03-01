# E-Commerce Recommendation Agent (AI 导购智能体)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-orange)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

> 一个基于大语言模型（LLM）的智能电商导购 Agent，具备多轮对话、混合检索（Hybrid Search）、RAG 问答和意图识别能力。

## 📖 项目简介

本项目旨在模拟真实的“金牌导购员”体验。不同于传统的关键词搜索，Rec Agent 能够理解用户模糊的自然语言需求（如“推荐几款适合拍夜景的复古相机”），通过多轮对话澄清需求，并结合**Elasticsearch**（精准检索）和**Milvus**（语义检索）提供精准的商品推荐。

核心能力：
- **🕵️ 意图识别 & 槽位填充**: 自动提取预算、品牌、用途等关键参数。
- **🔍 混合检索 (Hybrid RAG)**: 结合 ES 的倒排索引和 Milvus 的向量索引，兼顾准确率与召回率。
- **🧠 深度商品问答**: 基于 RAG 技术，回答关于商品参数、说明书、用户评论的细节问题。
- **🔄 反思与规划**: 引入 ReAct/LangGraph 模式，处理复杂任务拆解。

## 🛠️ 技术栈

*   **LLM**: OpenAI GPT-4o / DeepSeek (兼容 OpenAI 接口)
*   **Framework**: LangChain, LangGraph
*   **Backend**: FastAPI
*   **Vector DB**: Milvus (Docker) - 存储非结构化知识 (RAG)
*   **Search Engine**: Elasticsearch 8.x - 存储商品元数据 (Search)
*   **Database**: MySQL 8.0 - 存储用户/订单/日志 (Source of Truth)
*   **Cache**: Redis - 会话记忆

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
确保本地已安装 [Docker](https://www.docker.com/) 和 [Python 3.10+](https://www.python.org/)。

```bash
# 克隆项目
git clone https://github.com/your-username/rec_agent.git
cd rec_agent

# 安装 Python 依赖
pip install -r requirements.txt
```

### 2. 配置环境变量
复制 `.env.example` 为 `.env`，并填入你的 API Key 和数据库配置。

```bash
cp .env.example .env
# 编辑 .env 文件，填入 EMBEDDING_API_KEY 等信息
```

### 3. 启动基础设施 (Docker)
一键启动 MySQL, Elasticsearch, Milvus, Redis。

```bash
docker-compose up -d
```

### 4. 数据初始化 (Data Ingestion)
本项目提供了自动化脚本，用于生成 Mock 数据并同步到各类数据库。

```bash
# 1. 生成 Mock 数据 (JSON)
python scripts/generate_mock_data.py

# 2. 初始化 MySQL 表结构并入库
python scripts/init_mysql.py

# 3. 向量化并同步到 ES 和 Milvus
# 注意：此步需要消耗 Embedding API Token
python scripts/sync_to_vector_db.py
```

### 5. 启动服务 (暂未实现)
*目前项目处于数据层建设阶段，Agent 服务代码即将更新。*

## 📂 目录结构

```
rec_agent/
├── data/                   # 存放生成的 Mock 数据
├── scripts/                # 数据处理脚本
│   ├── config.py           # 脚本配置文件
│   ├── generate_mock_data.py
│   ├── init_mysql.py
│   └── sync_to_vector_db.py
├── .env                    # 环境变量 (不要提交到 Git)
├── docker-compose.yml      # 基础设施编排
└── requirements.txt        # Python 依赖
```

## 📝 脚本说明

| 脚本 | 作用 |
| :--- | :--- |
| `generate_mock_data.py` | 生成包含“数码相机”和“男装”品类的 80+ 条仿真数据，包含详细参数和用户评论。 |
| `init_mysql.py` | 初始化 MySQL 表结构 (users, products, orders) 并写入数据。 |
| `sync_to_vector_db.py` | **核心脚本**。将商品描述 Embedding 后存入 ES (用于搜索)，将评论切片 Embedding 后存入 Milvus (用于 RAG)。 |

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
