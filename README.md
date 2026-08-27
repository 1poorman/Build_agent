<p align="center">
  <h1 align="center">🤖 LangChain Agent 实战集</h1>
  <p align="center">基于 LangChain / LangGraph / deepagents 构建的多种 AI Agent 示例与实战代码</p>
</p>

---

## 📖 项目简介

本项目汇总了基于 **LangChain**、**LangGraph** 与 **deepagents** 框架构建的多种 Agent 实现，覆盖以下核心场景：

- 🔍 **RAG 文档问答**：向量检索 + 生成（支持 PostgreSQL 原生 Checkpoint 持久化）
- 🌐 **联网研究助手**：Tavily 网络搜索 + 连续对话记忆
- 🧩 **多子代理协作**：任务路由、并行检索、结果综合
- 🗄️ **SQL 技能增强**：通过 Middleware 渐进式揭示领域技能
- 📊 **数据分析**：本地文件系统后端 + 外部平台消息推送
- ⚖️ **合同审查自动化**：调用 agent-helper API 编排审查流水线
- 🤖 **API 调度 Agent**：LLM 根据问题自动路由调用文档审查 API，连续对话 + 持久化记忆

每个脚本均可独立运行，作为学习与二次开发的参考模板。

---

## ✨ 功能特性

| 特性 | 说明 |
|------|------|
| **多轮对话记忆** | `MemorySaver` / `InMemorySaver` / PostgreSQL Checkpoint 多种记忆方案 |
| **流式输出** | 实时显示 Agent 思考过程与工具调用 |
| **工具调用可视化** | 打印工具调用、子代理委派过程（含缩进层级） |
| **RAG 检索增强** | 在线抓取文档 → 切分 → Embedding → 向量检索 |
| **PostgreSQL 持久化** | 原生 LangGraph Checkpoint 落库，重启不丢上下文 |
| **用户确认机制** | 保存文件前中断等待用户确认（`interrupt` / `Command`） |
| **智能 API 路由** | LLM 自动选择并调用文档审查 API，无需预编排流程 |

---

## 📁 目录结构

```
langchain/
├── agents/                    # 核心 Agent 脚本
│   ├── agent-v1.py            # 研究助手 v1：create_react_agent + Tavily 搜索 + MemorySaver
│   ├── agent-v2.py            # 研究助手 v2：deepagents 多子代理 + 流式工具调用可视化 + 保存确认
│   ├── agent_rag.py           # RAG 文档问答：向量检索 + PostgreSQL Checkpoint 持久化
│   ├── agent_route.py         # 多源知识路由：LangGraph StateGraph + Send 并行分发
│   ├── agent_sql_skill.py     # SQL 助手：Skill Middleware 渐进式技能注入
│   ├── agent_data_analysis.py   # 数据分析：文件系统后端 + Slack 消息推送
│   ├── contract_review_agent.py # 合同审查自动化：调用 agent-helper API 编排审查流水线
│   └── api_dispatch_agent.py    # API 调度 Agent：LLM 自动路由调用文档审查 API + 记忆持久化
├── examples/                  # 入门与实验示例
│   ├── langgraph_minimal.py   # LangGraph 最小可运行示例
│   ├── deepagents_minimal.py  # deepagents + Tavily 最小示例
│   ├── weather_meteor.py      # 天气查询工具（wttr.in）示例
│   └── weather_openmeteo.py   # 天气查询工具（Open-Meteo）示例
├── data/                      # 输入数据（如 CSV）
├── reports/                   # 生成的报告输出（自动创建）
├── chat_logs/                 # 对话日志输出（自动创建）
├── .env.example               # 环境变量模板（复制为 .env 使用）
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- PostgreSQL 14+（`agent_rag.py` / `api_dispatch_agent.py` 需要）
- agent-helper 服务（`contract_review_agent.py` / `api_dispatch_agent.py` 需要）
- 兼容 OpenAI 协议的 LLM 服务（OpenAI / DeepSeek / Qwen 等）

### 2. 安装依赖

```bash
conda create -n agent python=3.11 -y
conda activate agent

pip install -r requirements.txt
# 或按需安装：
pip install langchain langgraph deepagents langchain-openai langchain-text-splitters
pip install tavily-python python-dotenv psycopg2-binary
pip install langgraph-checkpoint-postgres   # RAG Agent Postgres 持久化
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```ini
# LLM 服务（OpenAI 兼容协议）
OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
OPENAI_API_KEY="sk-xxx"
MODEL_NAME="deepseek-ai/DeepSeek-V4-Flash"

# Qwen 服务（agent_rag.py 使用）
QWEN_BASE_URL="http://your-qwen-endpoint/v1"
QWEN_API_KEY="sk-xxx"
QWEN_MODEL_NAME="Qwen3.6-27B-INT4"

# Tavily 搜索
TAVILY_API_KEY="tvly-xxx"

# PostgreSQL（agent_rag.py 使用）
POSTGRES_DSN="postgresql://postgres:postgres@localhost:5432/langchain_memory"

# 可选：LangSmith 追踪
LANGSMITH_TRACING="false"
LANGSMITH_API_KEY=""
```

### 4. 运行示例

```bash
# RAG 文档问答（需先启动 PostgreSQL）
python agents/agent_rag.py

# 研究助手 v1（联网搜索 + 连续对话）
python agents/agent-v1.py

# 研究助手 v2（多子代理，推荐）
python agents/agent-v2.py

# SQL 技能助手
python agents/agent_sql_skill.py

# 多源知识路由示例
python agents/agent_route.py

# 最小示例
python examples/langgraph_minimal.py

# 合同审查自动化（需先部署 agent-helper 服务）
python agents/contract_review_agent.py /path/to/contract.pdf [文件ID]

# API 调度 Agent（LLM 自动路由调用文档审查 API，连续对话）
python agents/api_dispatch_agent.py                      # 默认会话
python agents/api_dispatch_agent.py --thread my-session  # 指定会话
```

---

## 🧠 核心脚本详解

### `agent_rag.py` — RAG 文档问答 Agent

- **流程**：在线抓取 LangChain 官方文档 → `RecursiveCharacterTextSplitter` 切分 → `bge-m3` 向量化 → `InMemoryVectorStore` 检索 → deepagents 子代理分析 → 综合回答
- **持久化**：使用 LangGraph 原生 **PostgreSQL Checkpoint**（`PostgresSaver`），整个图状态落库，重启后通过 `thread_id` 恢复对话
- **交互**：支持将回答保存为 Markdown（保存前用户确认）

### `contract_review_agent.py` — 合同审查自动化 Agent

- **流程**：LangGraph 编排 6 步流水线 —— 文档解析（MinerU）→ 要素抽取 → 一致性核查 → 金额核查 → 印章/签名检测 → 汇总报告
- **后端**：通过 HTTP 调用 `agent-helper` 服务（`/api/mineru/textExtractDir`、`/api/agent/extractEntity`、`/api/agent/complianceAudit`、`/api/Contract/verifyContractAmount`、`/api/Contract/checkSignaturewithSeal`、`/api/detection/detectSeal`）
- **输出**：Markdown 审查报告（`reports/contract_review_*.md`）
- **配置**：服务地址通过环境变量 `AGENT_HELPER_BASE_URL` 指定（默认 `http://172.25.67.120:8359`）

### `api_dispatch_agent.py` — API 调度 Agent

- **核心**：LLM 根据用户问题**自动选择并调用**合适的文档审查 API，无需预编排固定流程
- **工具**：8 个 `@tool` 封装（全文提取、实体抽取、合规审查、金额核查、签章检测、印章检测、空白页、目录检测），对应 agent-helper 各接口
- **记忆**：`PostgresSaver` Checkpoint 持久化连续对话，重启后凭 `thread_id` 恢复完整上下文
- **交互**：CLI 连续对话，支持多轮追问（如"刚才那个文档的签章情况？"）

### `agent-v2.py` — 多子代理研究助手

- **架构**：主 Agent + `research-agent`（深入调研）+ `simple-search`（简单问答）两个子代理
- **记忆**：`MemorySaver`（短期）+ `InMemoryStore`（长期）双通道
- **特性**：流式显示工具调用与子代理委派、`interrupt` 实现保存前用户确认、对话自动落盘 `chat_logs/`

### `agent_route.py` — 多源知识路由

- **架构**：Router 分类查询 → `Send` 并行分发到 GitHub / Notion / Slack 三个专业 Agent → 综合结果
- **亮点**：演示 `StateGraph` + 条件路由 + 并行扇出（fan-out）模式

### `agent_sql_skill.py` — SQL 技能助手

- **亮点**：自定义 `AgentMiddleware`，将技能（schema + 业务逻辑）渐进式注入系统提示，避免上下文过载

---

## 🗄️ PostgreSQL 持久化说明（agent_rag.py）

启动前创建数据库：

```bash
# Ubuntu 安装 PostgreSQL
sudo apt-get install -y postgresql
sudo systemctl start postgresql

# 创建数据库与用户
sudo -u postgres psql -c "CREATE DATABASE langchain_memory;"
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
```

程序首次运行会自动创建 Checkpoint 表（`checkpoints`、`checkpoint_writes`、`checkpoint_blobs` 等）。

查看记忆数据：

```bash
PGPASSWORD=postgres psql -h localhost -U postgres -d langchain_memory \
  -c "SELECT thread_id, checkpoint_id FROM checkpoints ORDER BY checkpoint_id;"
```

---

## 🔒 安全须知

- **`.env` 含敏感密钥，已被 `.gitignore` 排除，切勿提交到仓库**
- 推送前检查：`git ls-files | grep -i env` 应无输出
- 生产环境建议：使用密钥管理服务（如 Vault / KMS）替代 `.env`

---

## 🧰 技术栈

| 组件 | 用途 |
|------|------|
| [LangChain](https://github.com/langchain-ai/langchain) | Agent 构建、工具、LLM 封装 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | 图状态编排、Checkpoint 持久化 |
| [deepagents](https://github.com/langchain-ai/deepagents) | 深度 Agent（子代理、文件系统后端） |
| [Tavily](https://tavily.com) | 联网搜索 API |
| [PostgreSQL](https://www.postgresql.org/) | Checkpoint 持久化存储 |

---

## 📄 许可证

[MIT](LICENSE)

---

> ⚠️ 本项目为学习与实验用途，部分示例使用模拟数据（如 `agent_route.py` 的搜索工具返回占位结果）。
