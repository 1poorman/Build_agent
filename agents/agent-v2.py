"""连续对话研究助手（基于 deepagents 的 Deep Agent）。

搜索后端支持两种模式，由环境变量 USE_TAVILY_MCP 切换（默认 0 = SDK 模式）：
    * SDK 模式  —— 直接调用 tavily-python，带 lru_cache 缓存，零额外依赖
    * MCP 模式  —— 通过 langchain-mcp-adapters 连接 Tavily 远程 MCP 服务器
                   （https://mcp.tavily.com/mcp/），工具由 MCP 提供，无本地缓存。
                   MCP 工具按轻重分级：tavily_search 给轻量子代理，
                   crawl/map/research 等重工具仅给研究子代理。
                   MCP 不可用或无工具时自动回退 SDK，保证 Agent 始终可用。

切换方式：
    python agents/agent-v2.py              # SDK 模式（默认，带缓存）
    USE_TAVILY_MCP=1 python agents/agent-v2.py   # MCP 模式

模块结构：
    1. 配置与常量        —— 环境变量、模型参数、时区等集中管理
    2. 工具定义          —— 供 Agent 调用的外部能力（网络搜索）
    3. 提示词            —— 系统提示模板
    4. 文本工具          —— 回复清洗等纯函数
    5. ConversationLogger —— 对话落盘日志
    6. ResearchAgent     —— Agent 构建与对话封装
    7. CLI 主流程        —— 交互式连续对话入口
"""

import os
import re
import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal
from functools import lru_cache
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.errors import GraphInterrupt
from deepagents import create_deep_agent
from langgraph.store.memory import InMemoryStore

# ==================== 1. 配置与常量 ====================
load_dotenv()

BEIJING_TZ = ZoneInfo("Asia/Shanghai")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "chat_logs")
EXIT_COMMANDS = {"exit", "quit", "退出"}


@dataclass
class Settings:
    """集中管理运行时配置，便于统一调整与测试。"""

    # 模型
    base_url: str = field(default_factory=lambda: os.getenv("QWEN3.8_BASE_URL"))
    api_key: str = field(default_factory=lambda: os.getenv("QWEN3.8_API_KEY", "empty"))
    model_name: str = field(default_factory=lambda: os.getenv("QWEN3.8_MODEL_NAME"))
    temperature: float = 0.7
    max_tokens: int = 16384
    streaming: bool = False  # 开启流式，降低首字延迟、提升体感速度
    # 是否启用模型思维链；关闭可大幅减少推理 token、显著提速。
    # 部分兼容 OpenAI 的模型（Qwen/DeepSeek 等）支持 enable_thinking 开关，
    # 设为 false 即关闭。字段名随服务商而异，可按需调整 _build_llm 中的 extra_body。
    thinking_enabled: bool = field(
        default_factory=lambda: os.getenv("ENABLE_THINKING", "true").lower() == "true"
    )

    # 工具
    tavily_api_key: str = field(default_factory=lambda: os.environ["TAVILY_API_KEY"])
    # 搜索后端开关：True = 走 Tavily 远程 MCP；False = 走 tavily-python SDK（带缓存）
    use_mcp: bool = field(
        default_factory=lambda: os.getenv("USE_TAVILY_MCP", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )

    # Agent 运行
    recursion_limit: int = 20  # 复杂任务（委派子代理 + 搜索 + 保存文件）需要更多步数


SETTINGS = Settings()


# ==================== 2. 工具定义 ====================
tavily_client = TavilyClient(api_key=SETTINGS.tavily_api_key)


@lru_cache(maxsize=128)
def _cached_tavily_search(
    query: str, max_results: int, topic: str, include_raw_content: bool
) -> dict:
    """带缓存的底层搜索；相同参数直接命中，省去网络往返与工具解析。"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


class InternetSearchInput(BaseModel):
    """深度检索入参：面向调研、多来源对比与报告撰写。"""

    query: str = Field(
        ...,
        description="完整且具体的检索问题，须包含主题与限定条件，不要只给关键词。"
                    "示例：'2026 年 LangChain 1.0 相比 0.3 有哪些破坏性变更'",
    )
    max_results: int = Field(
        default=3, ge=1, le=10,
        description="返回来源条数（1-10）。多方对比或撰写报告用 3-5。"
                    "若只需确认一条事实，说明这不是本工具的场景，应改用 quick_fact_check。",
    )
    topic: Literal["general", "news", "finance"] = Field(
        default="general",
        description="检索领域。general=通用；news=有时效性的新闻事件；"
                    "finance=公司、财报、行情等财经信息。",
    )
    include_raw_content: bool = Field(
        default=False,
        description="是否附带网页正文全文。会大幅增加 token 消耗，"
                    "仅在需要逐字引用原文时才开启；只要摘要请保持 False。",
    )


class QuickFactCheckInput(BaseModel):
    """快速事实核查入参：只需一个权威来源即可确认的单条事实。"""

    query: str = Field(
        ...,
        description="可一句话回答的事实性短问题，例如 'Python 3.12 发布于哪一年'。"
                    "不要用于开放式调研、多方案对比或报告撰写——那些属于 internet_search 的场景。",
    )
    max_results: int = Field(
        default=1, ge=1, le=3,
        description="来源条数（1-3，默认 1）。本工具只做单点核实；"
                    "若需要 3 条以上来源交叉对比，说明不是事实核查场景，应改用 internet_search。",
    )


@tool("internet_search", args_schema=InternetSearchInput)
def internet_search(
    query: str,
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> dict:
    """深度网络检索：多来源、可按领域（新闻/财经）筛选、可获取网页原文。

    适用场景：
      - 需要综合多个来源才能回答的开放式问题
      - 需要按领域筛选，或需要引用网页原文细节
      - 用户明确要求调研、对比、整理资料或输出报告

    不适用，请改用 quick_fact_check：
      - 只需确认一个日期、版本号、定义、人名等单条事实
      - 闲聊中顺带追问的小问题

    返回 Tavily 检索结果（带缓存）。
    """
    return _cached_tavily_search(query, max_results, topic, include_raw_content)


@tool("quick_fact_check", args_schema=QuickFactCheckInput)
def quick_fact_check(query: str, max_results: int = 1) -> dict:
    """快速事实核查：单来源、轻量，用于确认一条可一句话回答的事实。

    适用场景：
      - 确认一个具体日期、版本号、定义、人名或"是否成立"
      - 闲聊中需要联网核实的单条信息

    不适用，请改用 internet_search：
      - 需要多个来源对比、趋势梳理或撰写报告
      - 需要新闻/财经领域筛选，或需要网页原文

    返回 Tavily 检索结果（带缓存），最多 3 条来源。
    """
    return _cached_tavily_search(query, max_results, "general", False)


# ==================== 2b. MCP 工具（可选后端） ====================
SDK_BACKEND = "SDK（tavily-python，带缓存）"


def load_mcp_tools() -> list:
    """通过 MCP 协议加载 Tavily 远程服务器的工具。

    连接 Tavily 官方远程 MCP（Streamable HTTP），返回 LangChain 工具列表
    （当前服务端提供 tavily_search / tavily_extract / tavily_crawl /
    tavily_map / tavily_research）。工具在客户端被转换，因此不要求模型服务端
    支持 MCP——只要支持普通 function calling 即可。

    注意：MCP 路径无本地缓存，每次调用都是真实网络往返。

    Returns:
        LangChain 工具列表；连接失败时打印告警并返回空列表（由调用方回退 SDK）。
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={SETTINGS.tavily_api_key}",
            "headers": {
                # 透传给 Tavily API 的默认参数，避免每次调用重复指定
                "DEFAULT_PARAMETERS": json.dumps({
                    "include_raw_content": False,
                    "include_images": False,
                }),
            },
        }
    })
    # get_tools() 为异步接口，此处在启动时一次性拉取，之后同步复用
    tools = asyncio.run(client.get_tools())
    print(f"🔌 MCP 已连接 Tavily，加载工具: {[t.name for t in tools]}")
    return tools  # 原始（仅异步）工具，由 resolve_search_tools 包装并注入场景约束


# MCP 中较重的工具（爬取/站点地图/深度研究），只交给研究子代理
HEAVY_MCP_TOOLS = {"tavily_crawl", "tavily_map", "tavily_research"}
LIGHT_MCP_TOOLS = {"tavily_search"}

# MCP 工具同名同描述，两个子代理拿到完全一样的工具会失去区分度，
# 故在描述前注入场景前缀，等价于 SDK 模式下两个工具的语义分工。
MCP_RESEARCH_HINT = (
    "【调研场景】你服务于深度调研子代理：可综合多来源、可按领域检索、可读取网页正文，"
    "用于对比分析、趋势梳理与报告撰写。"
)
MCP_QUICK_HINT = (
    "【事实核查场景】你服务于轻量核查子代理：只做单点核实，一次调用即给出简短答案，"
    "不要展开多轮检索，不要使用 crawl/research 等重工具；"
    "若问题需要多来源对比或写报告，说明它不在本场景内。"
)


def _to_sync_tool(tool, scope_hint: str = ""):
    """把仅支持异步的 MCP 工具包装为同步可调用，并可注入场景约束。

    langchain-mcp-adapters 产出的 StructuredTool 只有 coroutine，
    同步 invoke 会抛 NotImplementedError。CLI 走的是同步 stream()，
    故此处用 asyncio.run 桥接，同时保留 coroutine 以兼容异步调用。

    Args:
        tool: MCP 原始工具。
        scope_hint: 追加到描述前的场景约束文本。
    """
    from langchain_core.tools import StructuredTool

    def _sync(**kwargs):
        return asyncio.run(tool.ainvoke(kwargs))

    desc = tool.description or tool.name
    return StructuredTool(
        name=tool.name,
        description=f"{scope_hint}\n\n{desc}" if scope_hint else desc,
        args_schema=tool.args_schema,
        func=_sync,
        coroutine=tool.coroutine,
    )


def resolve_search_tools() -> tuple[list, list, str]:
    """按 USE_TAVILY_MCP 决定搜索后端，返回 (研究工具, 简单工具, 模式名)。

    MCP 不可用时自动回退到 SDK 工具，保证 Agent 始终可用。
    """
    if not SETTINGS.use_mcp:
        return [internet_search], [quick_fact_check], "SDK（tavily-python，带缓存）"
    try:
        mcp_tools = load_mcp_tools()
    except Exception as e:
        print(f"⚠️ MCP 连接失败，回退 SDK 模式: {type(e).__name__}: {e}")
        return [internet_search], [quick_fact_check], "SDK（回退，MCP 不可用）"
    if not mcp_tools:
        print("⚠️ MCP 未返回任何工具，回退 SDK 模式")
        return [internet_search], [quick_fact_check], "SDK（回退，MCP 无工具）"

    # MCP 工具在同一子代理内无法靠名字区分，故用描述前缀注入场景约束
    research = [_to_sync_tool(t, MCP_RESEARCH_HINT) for t in mcp_tools]
    # 轻量子代理只用搜索类工具，避免误触爬取/深度研究等重操作
    light = [
        _to_sync_tool(t, MCP_QUICK_HINT)
        for t in mcp_tools if getattr(t, "name", "") in LIGHT_MCP_TOOLS
    ]
    if not light:  # 名称约定变化时至少保证有工具可用
        light = [
            _to_sync_tool(t, MCP_QUICK_HINT)
            for t in mcp_tools if getattr(t, "name", "") not in HEAVY_MCP_TOOLS
        ] or research
    # 未识别到的新工具（不在两个集合中）默认交给研究子代理
    return research, light, f"MCP（Tavily 远程，共 {len(mcp_tools)} 个工具）"


DEFAULT_REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")


def save_report_to_md(report: str, filename: str) -> str:
    """将报告保存为 Markdown 文件到 reports 目录。
    自动补全 .md 后缀、自动创建目录；写入前中断等待用户确认。"""
    if not filename.endswith(".md"):
        filename = f"{filename}.md"
    os.makedirs(DEFAULT_REPORT_DIR, exist_ok=True)
    filepath = os.path.join(DEFAULT_REPORT_DIR, filename)

    # 暂停图执行，等待用户确认
    preview = report[:300] + ("..." if len(report) > 300 else "")
    user_confirmed = interrupt({
        "action": "confirm_save",
        "filepath": filepath,
        "size": len(report),
        "preview": preview,
    })

    if not user_confirmed:
        return "用户取消了保存操作。"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    return f"已保存至 {filepath}"

# ==================== 3. 提示词 ====================
REPORT_PROMPT = """您是一位专家级研究员和技术写作专家，我们正在进行连续对话。

## 核心规则（必须严格遵守，违反即失败）
1. **禁止使用 write_todos 或任何任务计划工具**。不要规划，直接行动。
2. **整个回答过程最多调用 2 次 internet_search**。收到结果后立即撰写最终回答，严禁追加搜索。
3. **所有搜索必须在同一轮中并行发起**（一次发多个 tool_calls），绝对不要分多轮逐个搜索。
4. 检索用 internet_search（支持 topic 领域筛选、max_results 条数、include_raw_content 原文）。
   搜索结果只提取摘要，不要请求原始内容。

## 路由策略
- 纯闲聊、寒暄，或基于常识就能回答的问题：不要委派子代理，直接简洁回答。
- 需要最新信息时，用下面这条判据二选一，不要凭"问题看起来难不难"判断：
  * 答案只需一个来源即可确认（单个日期、版本号、定义、是否成立）→ 委派 simple-search
  * 答案需要多个来源交叉验证（对比、趋势、技术细节、写报告）→ 委派 research-agent

## 回复格式要求
- **普通提问**：请用清晰、结构化的文字直接回答，不要强行套用报告格式。
- **明确要求写报告**：请输出完整的 Markdown 报告，包含标题、摘要、核心内容、参考资料等结构。
- 报告中若需生成时间，请使用用户消息中提供的当前时间。

## 注意
- 对话是连续的，你可以结合上下文回答问题，不要反复搜索同一问题。
"""

MAIN_PROMPT = """您是一位高效的研究助手，负责协调子代理完成任务。

## 核心规则（必须严格遵守）
1. **禁止使用 write_todos 或任何任务计划工具**。直接行动，不要规划。
2. 调用 save_report_to_md 保存文件后，直接报告结果给用户，**严禁**再调用 ls/glob 等文件工具去「验证」文件是否存在。
3. **禁止调用 execute 执行 shell 命令**。

## 路由策略（按顺序判断，避免误委派）
- 纯闲聊或常识问题：直接回答，不委派子代理。
- 用户要求保存文件：调用 save_report_to_md 即可，不要再调用文件工具去"验证"。
- 需要联网才能回答时，用"答案是否需要多个来源交叉验证"来分流：
  * 只需确认单条事实（一个日期/版本号/定义/是否成立）→ 委派 simple-search
  * 需要多来源对比、趋势梳理、技术考证或写报告 → 委派 research-agent
- 两者拿不准时，按"是否需要交叉验证"决定，不要两个都派。

## 委派约束（防循环，必须遵守）
- **每个问题最多委派 2 次子代理**。子代理返回后，无论结果是否完整，
  都必须基于已有信息给用户作答，严禁再次委派。
- 子代理回复"未能核实"时，直接如实转告用户即可，**不要换个子代理重新试一遍**。
- 用户没要求重试时，不要因为答案不够完美就重复委派。

## 效率要求
- 尽量减少工具调用轮数。能用一轮解决就不要分多轮。
"""

SIMPLE_PROMPT = """您是一位友好的对话助手，负责处理简单问答与闲聊。

## 策略
- 大多数简单问题可直接回答，无需搜索，更不要委派其它子代理。
- 仅当问题涉及需要联网核实的最新事实时，才使用 quick_fact_check 进行轻量核查。
- **硬上限：最多调用 2 次 quick_fact_check**。收到结果后立即作答，
  严禁因为"没找到确切答案"就换措辞反复搜索（这属于深度调研，不是事实核查）。
- 两次都没查到，就直接说明未能核实，并给出你所知道的信息，不要继续搜。
- 若问题实则需要多来源对比或展开调研，说明它不属于本子代理，简短作答即可。
- 回答保持简洁、口语化，不要套用报告格式。
"""


# ==================== 4. 文本工具 ====================
_THINK_TAG_PATTERN = re.compile(r"<think\s*>.*?</think\s*>", flags=re.DOTALL)


def clean_think_tags(text: str) -> str:
    """移除 <think> 推理标签并去除首尾空白。"""
    return _THINK_TAG_PATTERN.sub("", text).strip()


def now_str() -> str:
    """返回北京时间的格式化字符串。"""
    return datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")


# ==================== 5. 对话日志 ====================
class ConversationLogger:
    """将每轮对话以 Markdown 形式落盘保存。"""

    def __init__(self, log_dir: str = LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"dialogue_{timestamp}.md")
        self._write_header()

    def _write_header(self) -> None:
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(f"# 研究对话记录\n> 创建时间：{now_str()}\n\n")

    def log(self, user_input: str, response: str) -> None:
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(f"## 🧑 你\n{user_input}\n\n")
            f.write(f"## 🤖 助手\n{response}\n\n---\n\n")


# ==================== 6. Agent 封装 ====================
class ResearchAgent:
    """封装 Deep Agent 的构建与连续对话逻辑。"""

    def __init__(self, settings: Settings = SETTINGS):
        self.settings = settings
        self.memory = MemorySaver()
        self.store = InMemoryStore()
        self.search_backend = SDK_BACKEND  # 由 _build_agent 更新为实际生效的后端
        self.agent = self._build_agent()

    def _build_llm(self) -> ChatOpenAI:
        kwargs = {
            "base_url": self.settings.base_url,
            "api_key": self.settings.api_key,
            "model": self.settings.model_name,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "streaming": self.settings.streaming,
        }
        if not self.settings.thinking_enabled:
            # 关闭模型思维链；字段名随服务商而异（Qwen/DeepSeek 等使用 enable_thinking），
            # 若你的服务商不兼容，可在此修改 extra_body 的 key。
            kwargs["extra_body"] = {"enable_thinking": False}
        return ChatOpenAI(**kwargs)

    def _build_agent(self):
        research_tools, simple_tools, self.search_backend = resolve_search_tools()
        research_subagent = {
            "name": "research-agent",
            "description": (
                "【委派条件】需要综合多个来源才能回答的调研类问题：横向对比、趋势梳理、"
                "技术细节考证，或用户明确要求调研/整理/撰写报告。\n"
                "【不要委派】只需确认单条事实（一个日期、版本号、定义）的问题，以及纯闲聊寒暄。"
            ),
            "system_prompt": REPORT_PROMPT,
            "tools": research_tools,
            # "model": "openai:gpt-4o",  # 可选覆盖，默认为主 Agent 模型
        }
        simple_search_subagent = {
            "name": "simple-search",
            "description": (
                "【委派条件】只需一个简短答案的场景：确认单条事实（日期、版本号、定义、"
                "某人某事是否成立），或闲聊中顺带需要联网核实一点信息。\n"
                "【不要委派】需要多来源对比、趋势分析、深度调研或写报告的问题——那些应派给 research-agent。"
            ),
            "system_prompt": SIMPLE_PROMPT,
            "tools": simple_tools,
        }
        return create_deep_agent(
            model=self._build_llm(),
            system_prompt=MAIN_PROMPT,
            subagents=[research_subagent, simple_search_subagent],
            tools=[save_report_to_md],
            checkpointer=self.memory,  # 短期记忆：同一 thread 内的连续对话
            store=self.store,          # 长期记忆：跨会话的持久化存储
        )

    @staticmethod
    def _extract_messages(node_output) -> list:
        """从节点输出中取出消息列表，兼容 Overwrite 包装与 __overwrite__ 字典。"""
        if not isinstance(node_output, dict):
            return []
        messages = node_output.get("messages", [])
        # langgraph 的 Overwrite(value=...) 包装：绕过 reducer 直接覆盖
        if hasattr(messages, "value"):
            messages = messages.value
        # 序列化形式 {"__overwrite__": value}
        elif isinstance(messages, dict) and "__overwrite__" in messages:
            messages = messages["__overwrite__"]
        return messages if isinstance(messages, list) else []

    @staticmethod
    def _print_tool_call(tool_call: dict, namespace: tuple) -> None:
        """打印一次工具调用；对 task 委派额外展示目标子代理。"""
        name = tool_call.get("name", "unknown")
        args = tool_call.get("args", {}) or {}
        depth = "  " * len(namespace)  # 子图层级缩进

        if name == "task":
            subagent = args.get("subagent_type", "?")
            desc = args.get("description", "")
            print(f"{depth}🧩 委派子代理 -> {subagent}｜任务: {desc}")
        else:
            preview = args.get("query", args)
            print(f"{depth}🔧 调用工具 -> {name}｜参数: {preview}")

    def chat(self, user_input: str, thread_id: str = "default_thread") -> str:
        """进行单轮对话（流式），实时显示调用的子代理与工具。
        写入文件时自动中断等待用户确认后继续。"""
        dynamic_input = f"[当前时间: {now_str()}]\n{user_input}"
        config = {
            "recursion_limit": self.settings.recursion_limit,
            "configurable": {"thread_id": thread_id},
        }

        current_input = {"messages": [{"role": "user", "content": dynamic_input}]}
        final_text = ""
        seen_calls: set = set()  # 已打印的 tool_call id，避免 Overwrite 重放导致重复

        while True:
            interrupted = False
            try:
                for namespace, update in self.agent.stream(
                    current_input,
                    config=config,
                    stream_mode="updates",
                    subgraphs=True,
                ):
                    # 检测流中的中断信息（stream 返回的 update 可能包含 __interrupt__）
                    if isinstance(update, dict) and "__interrupt__" in update:
                        interrupted = True
                        interrupt_list = update["__interrupt__"]
                        for interrupt_item in interrupt_list:
                            # interrupt_item 是 Interrupt 对象，取 .value 获得实际数据
                            interrupt_data = getattr(interrupt_item, "value", interrupt_item) if interrupt_item else {}
                            if isinstance(interrupt_data, dict) and interrupt_data.get("action") == "confirm_save":
                                print(f"\n📄 即将保存文件: {interrupt_data.get('filepath')}")
                                print(f"   大小: {interrupt_data.get('size', '?')} 字符")
                                print(f"   内容预览: {interrupt_data.get('preview', '')}")
                                try:
                                    answer = input("   >>> 确认保存? (y/n): ").strip().lower()
                                except (EOFError, KeyboardInterrupt):
                                    answer = "n"
                                confirmed = answer == "y"
                                current_input = Command(resume=confirmed)
                                print()
                            else:
                                current_input = Command(resume=True)
                        break

                    for node_output in update.values():
                        for msg in self._extract_messages(node_output):
                            # 打印工具/子代理调用（按 id 去重）
                            for tool_call in getattr(msg, "tool_calls", None) or []:
                                call_id = tool_call.get("id") or id(tool_call)
                                if call_id in seen_calls:
                                    continue
                                seen_calls.add(call_id)
                                self._print_tool_call(tool_call, namespace)
                            # 记录顶层代理的最终文本回复
                            content = getattr(msg, "content", "")
                            is_ai = getattr(msg, "type", "") == "ai"
                            if not namespace and is_ai and content and not getattr(msg, "tool_calls", None):
                                final_text = content
                if not interrupted:
                    break  # 流正常结束，退出循环
            except GraphInterrupt as e:
                # GraphInterrupt 的 e.args[0] 是 Interrupt 对象，需取 .value 获得实际数据
                interrupt_obj = e.args[0] if e.args else None
                interrupt_data = getattr(interrupt_obj, "value", interrupt_obj) if interrupt_obj else {}
                if isinstance(interrupt_data, dict) and interrupt_data.get("action") == "confirm_save":
                    print(f"\n📄 即将保存文件: {interrupt_data.get('filepath')}")
                    print(f"   大小: {interrupt_data.get('size', '?')} 字符")
                    print(f"   内容预览: {interrupt_data.get('preview', '')}")
                    try:
                        answer = input("   >>> 确认保存? (y/n): ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        answer = "n"
                    confirmed = answer == "y"
                    current_input = Command(resume=confirmed)
                    print()
                else:
                    # 未知中断类型，自动批准
                    current_input = Command(resume=True)

        return clean_think_tags(final_text)


# ==================== 7. CLI 主流程 ====================
def _print_banner(backend: str) -> None:
    print("=" * 60)
    print("🤖 研究助手（Deep Agent）已就绪，支持连续对话与上下文记忆！")
    print(f"🔎 搜索后端: {backend}")
    print(f"   切换方式: USE_TAVILY_MCP=1 python agents/agent-v2.py")
    print("💡 输入您的问题开始研究，输入 'exit' 或 'quit' 退出。")
    print("=" * 60)


def run_cli() -> None:
    """启动交互式连续对话。"""
    agent = ResearchAgent()
    _print_banner(agent.search_backend)
    logger = ConversationLogger()
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    while True:
        try:
            user_input = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 检测到退出信号，再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("👋 再见！")
            break

        print("\n🤖 助手思考中...\n")
        start = time.time()
        try:
            response = agent.chat(user_input, thread_id=session_id)
        except Exception as e:
            print(f"❌ 出错了: {e}")
            continue
        print(f"🔥 思考耗时: {time.time() - start:.2f} 秒")
        print(f"🤖 助手:\n{response}")
        logger.log(user_input, response)

    print(f"\n📝 完整对话记录已保存至: {logger.filepath}")


if __name__ == "__main__":
    run_cli()
