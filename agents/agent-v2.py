"""连续对话研究助手（基于 deepagents 的 Deep Agent）。

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
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal
from functools import lru_cache
import time
from dotenv import load_dotenv
from tavily import TavilyClient
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
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "empty"))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME"))
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


def internet_search(
    query: str,
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """运行网络搜索，返回 Tavily 检索结果（带缓存）。"""
    return _cached_tavily_search(query, max_results, topic, include_raw_content)

def simple_search(query: str, max_results: int = 1):
    """运行简单搜索，返回 Tavily 检索结果（带缓存）。"""
    return _cached_tavily_search(query, max_results, "general", False)

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
2. **整个回答过程最多调用 2 次搜索工具**。收到结果后立即撰写最终回答，严禁追加搜索。
3. **所有搜索必须在同一轮中并行发起**（一次发多个 tool_calls），绝对不要分多轮逐个搜索。
4. 搜索结果只提取摘要，不要请求原始内容。

## 路由策略
- 纯闲聊、寒暄，或基于常识就能回答的问题：不要委派子代理，直接简洁回答。
- 仅当问题需要最新信息或深入调研时，才委派 research-agent 子代理。

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

## 路由策略
- 纯闲聊或常识问题：直接回答，不委派子代理。
- 需要搜索和研究的问题：委派 research-agent。
- 用户要求保存文件：调用 save_report_to_md 即可。

## 效率要求
- 尽量减少工具调用轮数。能用一轮解决就不要分多轮。
"""

SIMPLE_PROMPT = """您是一位友好的对话助手，负责处理简单问答与闲聊。

## 策略
- 大多数简单问题可直接回答，无需搜索，更不要委派其它子代理。
- 仅当问题涉及需要联网核实的最新事实时，才使用 simple_search 进行一次轻量搜索。
- 不要对同一问题反复搜索。
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
        research_subagent = {
            "name": "research-agent",
            "description": "用于研究更深入的问题",
            "system_prompt": REPORT_PROMPT,
            "tools": [internet_search],
            # "model": "openai:gpt-4o",  # 可选覆盖，默认为主 Agent 模型
        }
        simple_search_subagent = {
            "name": "simple-search",
            "description": "用于简单回答或闲聊",
            "system_prompt": SIMPLE_PROMPT,
            "tools": [simple_search],
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
def _print_banner() -> None:
    print("=" * 60)
    print("🤖 研究助手（Deep Agent）已就绪，支持连续对话与上下文记忆！")
    print("💡 输入您的问题开始研究，输入 'exit' 或 'quit' 退出。")
    print("=" * 60)


def run_cli() -> None:
    """启动交互式连续对话。"""
    _print_banner()

    agent = ResearchAgent()
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
