import os, re
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# ==================== 初始化 Tavily ====================
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# ==================== 定义工具 ====================
@tool
def internet_search(
    query: str,
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> str:
    """对给定查询进行互联网搜索。可以指定最大结果数、主题类别以及是否包含原始内容。"""
    results = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    summaries = []
    for r in results.get("results", []):
        summaries.append(
            f"- 标题: {r.get('title', '')}\n"
            f"  摘要: {r.get('content', '')}\n"
            f"  链接: {r.get('url', '')}"
        )
    return "\n\n".join(summaries)

# ==================== 初始化模型 ====================
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY", "empty"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.7,
    max_tokens=8192,
    streaming=False,
)

BEIJING_TZ = ZoneInfo("Asia/Shanghai")

# ==================== 系统提示（适配对话模式）====================
report_prompt = """您是一位专家级研究员和技术写作专家，我们正在进行连续对话。

## 研究策略
1. 根据用户问题判断是否需要搜索。如果只需简单回答或闲聊，直接回答即可。
2. 如果需要查找最新信息或深入调研，最多搜索 2 次，每次使用不同角度的查询。
3. 搜索结果只提取摘要即可，不要请求原始内容。

## 回复格式要求
- **普通提问**：请用清晰、结构化的文字直接回答，不要强行套用报告格式。
- **明确要求写报告**：请输出完整的 Markdown 报告，包含标题、摘要、核心内容、参考资料等结构。
- 报告中若需生成时间，请使用用户消息中提供的当前时间。

## 注意
- 对话是连续的，你可以结合上下文回答问题，不要反复搜索同一问题。
"""

# ==================== 初始化记忆组件 & 构建 Agent ====================
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[internet_search],
    prompt=report_prompt,
    checkpointer=memory,
)

def clean_think_tags(text):
    """移除 <think > 推理标签"""
    return re.sub(r'<think\s*>.*?</think\s*>', '', text, flags=re.DOTALL).strip()


# ==================== 核心对话函数 ====================
def chat_with_agent(user_input: str, thread_id: str = "default_thread") -> str:
    """与 Agent 进行单轮对话"""
    # 1. 获取当前时间并注入到用户消息中
    current_time = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")
    dynamic_input = f"[当前时间: {current_time}]\n{user_input}"

    # 2. 调用 Agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": dynamic_input}]},
        config={
            "recursion_limit": 8,
            "configurable": {"thread_id": thread_id}
        },
    )

    # 3. 提取并清理回复
    final_message = result["messages"][-1].content
    return clean_think_tags(final_message)


# ==================== 主程序（交互式对话） ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 研究助手已就绪，支持连续对话与上下文记忆！")
    print("💡 输入您的问题开始研究，输入 'exit' 或 'quit' 退出。")
    print("=" * 60)
    
    # 为本次运行创建一个唯一的会话 ID
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 准备对话日志目录和文件（基于项目根目录，保证从任意位置运行都正确）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "chat_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"dialogue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    
    # 初始化日志文件
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"# 研究对话记录\n> 创建时间：{datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 开启交互循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 检测到退出信号，再见！")
            break

        # 空输入跳过
        if not user_input:
            continue
            
        # 退出指令
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("👋 再见！")
            break

        # 打印思考提示
        print("\n🤖 助手思考中...\n")
        
        # 调用 Agent 获取回复
        response = chat_with_agent(user_input, thread_id=session_id)
        
        # 打印 Agent 回复
        print(f"🤖 助手:\n{response}")

        # 将本轮对话追加写入日志文件
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"## 🧑 你\n{user_input}\n\n")
            f.write(f"## 🤖 助手\n{response}\n\n---\n\n")

    # 退出时提示日志保存位置
    print(f"\n📝 完整对话记录已保存至: {log_filename}")
