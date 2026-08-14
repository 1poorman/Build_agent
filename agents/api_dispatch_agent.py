"""API 调度 Agent：由大模型根据用户问题自动调用 agent-helper 的各类 API。

核心特性：
    1. 工具封装    —— 将 agent-helper 全部能力封装为 LangChain @tool
    2. 智能路由    —— LLM 理解用户意图，自动选择并调用合适的工具
    3. 连续对话    —— 多轮对话记忆（Postgres Checkpoint 持久化）
    4. 记忆持久化  —— 重启进程后通过 thread_id 恢复历史

用法：
    python agents/api_dispatch_agent.py            # 默认会话
    python agents/api_dispatch_agent.py --thread <id>   # 指定会话
"""

import os
import json
import re
import time
import datetime
from typing import Any, Optional, Literal
from dotenv import load_dotenv

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from deepagents import create_deep_agent

load_dotenv()

# ---------------------------------------------------------------------------
# 1. 配置与常量
# ---------------------------------------------------------------------------
AGENT_HELPER_BASE_URL = os.getenv(
    "AGENT_HELPER_BASE_URL", "http://172.25.67.120:8018"
)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# HTTP 超时：大文件 MinerU 解析可能超过 30 分钟
TIMEOUT = 3600.0

# Postgres 持久化
POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://postgres:postgres@localhost:5432/langchain_memory",
)
DEFAULT_THREAD = "api-dispatch-session-1"

# 合同要素抽取类型
CONTRACT_ENTITY_TYPES = {
    "合同编号": "合同编号或合同号",
    "甲方名称": "采购方/甲方公司全称",
    "乙方名称": "供应商/乙方公司全称",
    "合同签订日期": "合同签署日期",
    "合同有效期/起止日期": "合同生效与截止日期",
    "合同总金额": "合同总价/合同金额",
    "付款方式": "一次性付款或分期付款",
    "税率/发票类型": "增值税税率及发票类型",
}


# ---------------------------------------------------------------------------
# 2. agent-helper API 客户端
# ---------------------------------------------------------------------------
def _post_object(path: str, json_data: dict) -> Any:
    """POST JSON 对象请求体（接口参数为字段定义的场景）。"""
    url = f"{AGENT_HELPER_BASE_URL}{path}"
    resp = httpx.post(url, json=json_data, timeout=TIMEOUT)
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 200:
        raise RuntimeError(f"API {path} 返回错误: {body.get('msg')}")
    return body.get("data")


def _post_raw(path: str, body: Any) -> Any:
    """POST 裸 JSON 请求体（接口参数为 `str = Body(...)` 的场景）。"""
    url = f"{AGENT_HELPER_BASE_URL}{path}"
    resp = httpx.post(
        url,
        content=json.dumps(body, ensure_ascii=False),
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    resp_body = resp.json()
    if resp_body.get("code") != 200:
        raise RuntimeError(f"API {path} 返回错误: {resp_body.get('msg')}")
    return resp_body.get("data")


def _sanitize_file_id(file_id: str) -> str:
    """清洗 file_id，去除连字符等 Milvus 非法字符。"""
    return re.sub(r"[^0-9a-zA-Z_]", "", file_id)


# ---------------------------------------------------------------------------
# 3. 工具定义（LLM 可调用的能力）
# ---------------------------------------------------------------------------
@tool(parse_docstring=True)
def extract_document_text(pdf_path: str, file_id: str = "") -> str:
    """提取 PDF 文档的文本内容（MinerU 解析）。

    适用于从合同中提取全文、查找特定条款、分析文档内容等场景。

    Args:
        pdf_path: 合同/文档 PDF 的服务器绝对路径。
        file_id: 文件唯一标识（可选，UUID 或任意字符串）。

    Returns:
        JSON 字符串，包含 device_text_path（提取的文本文件路径）、
        text_path、milvus_uuid 等解析结果。
    """
    file_id = _sanitize_file_id(file_id) or str(int(time.time()))
    save_path = os.path.join(os.path.dirname(pdf_path), "mineru_result")
    result = _post_object("/api/mineru/textExtractDir", {
        "file_list": [{
            "file_path": pdf_path,
            "save_path": save_path,
            "file_id": file_id,
        }]
    })
    return json.dumps(result, ensure_ascii=False)


@tool(parse_docstring=True)
def extract_contract_entities(pdf_path: str, file_id: str = "") -> str:
    """抽取合同关键要素（甲方乙方、合同金额、付款方式、日期等）。

    适用于审查合同时获取结构化信息。

    Args:
        pdf_path: 合同 PDF 的服务器绝对路径。
        file_id: 文件唯一标识（可选）。

    Returns:
        JSON 字符串，包含合同编号、甲方名称、乙方名称、签订日期、
        有效期、总金额、付款方式、税率等字段。
    """
    file_id = _sanitize_file_id(file_id)
    # 先解析文本，再抽取实体
    save_path = os.path.join(os.path.dirname(pdf_path), "mineru_result")
    try:
        parse_res = _post_object("/api/mineru/textExtractDir", {
            "file_list": [{
                "file_path": pdf_path,
                "save_path": save_path,
                "file_id": file_id,
            }]
        })
        device_text = ""
        if isinstance(parse_res, list) and parse_res:
            device_text = parse_res[0].get("device_text", "")
            milvus_uuid = parse_res[0].get("milvus_uuid", "")
        else:
            milvus_uuid = ""
    except Exception:
        # embedding 失败时推断产物路径
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        device_text = os.path.join(save_path, f"{basename}_device.txt")
        milvus_uuid = ""

    file_list = [{
        "file_name": os.path.basename(pdf_path),
        "text": device_text,
        "entity_types": CONTRACT_ENTITY_TYPES,
        "task_constraint": "从合同文本中抽取上述要素，缺失字段为 null",
        "milvus_uuid": milvus_uuid,
    }]
    result = _post_object("/api/agent/extractEntity", {"file_list": file_list})
    return json.dumps(result, ensure_ascii=False)


@tool(parse_docstring=True)
def compliance_review(pdf_path: str, rules: list = None) -> str:
    """对合同/文档进行要素一致性合规审查。

    适用于检查合同金额大小写是否一致、签订日期是否在有效期内、
    主体名称是否完整等合规性问题。

    Args:
        pdf_path: 合同 PDF 的服务器绝对路径。
        rules: 审查规则列表（可选）。默认检查金额、日期、主体名称。

    Returns:
        JSON 字符串，包含每条规则的审查结果、风险内容与建议。
    """
    if rules is None:
        rules = [
            "核查合同总金额、大小写金额是否一致，若不一致需指出",
            "核查合同签订日期是否在有效期范围内，若超出需指出",
            "核查甲方、乙方名称是否完整且正确，与营业执照名称是否一致",
        ]
    save_path = os.path.join(os.path.dirname(pdf_path), "mineru_result")
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    device_text = os.path.join(save_path, f"{basename}_device.txt")

    review_list = [
        {"object": device_text, "rule": rule} for rule in rules
    ]
    result = _post_object("/api/agent/complianceAuditDir", {
        "review_list": review_list
    })
    return json.dumps(result, ensure_ascii=False)


@tool(parse_docstring=True)
def verify_contract_amount(pdf_path: str, file_id: str = "") -> str:
    """核查合同金额、付款方式与分期付款合计是否一致。

    适用于确认合同总金额与分期金额是否匹配、付款方式是否合理等场景。

    Args:
        pdf_path: 合同 PDF 的服务器绝对路径。
        file_id: 文件唯一标识（可选）。

    Returns:
        JSON 字符串，包含合同金额、付款方式、分期明细、金额一致性判断。
    """
    result = _post_object("/api/Contract/verifyContractAmount", {
        "pdf_path": pdf_path,
        "file_id": _sanitize_file_id(file_id),
    })
    return json.dumps(result, ensure_ascii=False)


@tool(parse_docstring=True)
def check_signature_with_seal(pdf_path: str) -> str:
    """检测盖章页上签名与印章是否在同一页（签章同页校验）。

    适用于确认合同关键页是否同时存在签名和盖章。

    Args:
        pdf_path: 合同/文档 PDF 的服务器绝对路径。

    Returns:
        JSON 字符串，包含签章是否同页、各匹配页的印章数与签名数。
    """
    result = _post_raw("/api/Contract/checkSignaturewithSeal", pdf_path)
    return json.dumps(result, ensure_ascii=False)


@tool(parse_docstring=True)
def detect_seal(pdf_path: str) -> str:
    """检测 PDF 文档中的印章完整性。

    适用于确认文档是否盖章、印章是否完整。

    Args:
        pdf_path: 合同/文档 PDF 的服务器绝对路径。

    Returns:
        JSON 字符串，包含印章是否完整、印章数量、印章位置等信息。
    """
    result = _post_raw("/api/detection/detectSeal", pdf_path)
    return json.dumps(result, ensure_ascii=False)


@tool(parse_docstring=True)
def find_blank_pages(pdf_path: str) -> str:
    """检测 PDF 文档中的空白页。

    适用于检查文档是否有缺页、空白页等异常。

    Args:
        pdf_path: 合同/文档 PDF 的服务器绝对路径。

    Returns:
        JSON 字符串，包含空白页页码列表。
    """
    result = _post_object("/api/detection/findBlankPages", {"pdf_path": pdf_path})
    return json.dumps(result, ensure_ascii=False)


@tool(parse_docstring=True)
def detect_catalogue(pdf_path: str) -> str:
    """检测 PDF 文档是否包含目录。

    适用于检查合同/标书是否缺少目录。

    Args:
        pdf_path: 合同/文档 PDF 的服务器绝对路径。

    Returns:
        JSON 字符串，包含目录存在性检测结果。
    """
    result = _post_object("/api/detection/catalogue", {"pdf_path": pdf_path})
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 4. 模型
# ---------------------------------------------------------------------------
def build_model() -> ChatOpenAI:
    """构建支持工具调用的 LLM。"""
    return ChatOpenAI(
        base_url=os.getenv("QWEN_BASE_URL"),
        api_key=os.getenv("QWEN_API_KEY"),
        model=os.getenv("QWEN_MODEL_NAME"),
        temperature=0.3,
        max_tokens=8192,
        streaming=True,
        timeout=300,
        max_retries=1,
    )


# ---------------------------------------------------------------------------
# 5. Agent 构建与 Postgres 持久化
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个智能文档审查与合同分析助手，负责根据用户的问题自动调用合适的 API 工具来处理文档。

可用工具及适用场景：
1. extract_document_text - 提取 PDF 全文内容（查找条款、分析内容）
2. extract_contract_entities - 抽取合同要素（甲乙双方、金额、日期、付款方式）
3. compliance_review - 合同合规性审查（金额一致性、日期有效性、主体名称）
4. verify_contract_amount - 金额与分期付款核查
5. check_signature_with_seal - 签章是否同页检测
6. detect_seal - 印章完整性检测
7. find_blank_pages - 空白页检测
8. detect_catalogue - 目录检测

工作原则：
- 分析用户意图，选择最合适的工具；复杂任务可多次调用不同工具
- 调用工具后，将结果整理成清晰、结构化的中文回答
- 涉及"审查合同/检查合同"时，通常需要依次调用多个工具给出综合结论
- 如果用户未提供 PDF 路径，请询问其提供
- 回答要专业、准确，明确标注哪些检查通过、哪些存在问题
"""


def build_agent(checkpointer: PostgresSaver):
    """构建 API 调度 Agent。"""
    return create_deep_agent(
        model=build_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=[
            extract_document_text,
            extract_contract_entities,
            compliance_review,
            verify_contract_amount,
            check_signature_with_seal,
            detect_seal,
            find_blank_pages,
            detect_catalogue,
        ],
        checkpointer=checkpointer,
    )


def build_checkpointer() -> PostgresSaver:
    """创建 Postgres Checkpointer（持久化连续对话）。"""
    from psycopg import Connection
    conn = Connection.connect(POSTGRES_DSN, autocommit=True, prepare_threshold=0)
    cp = PostgresSaver(conn)
    cp.setup()
    return cp


checkpointer = build_checkpointer()
agent = build_agent(checkpointer)


# ---------------------------------------------------------------------------
# 6. CLI 主流程
# ---------------------------------------------------------------------------
def main(thread_id: str = DEFAULT_THREAD) -> None:
    """交互式连续对话入口。"""
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": thread_id}}
    print(f"API 调度 Agent 已就绪（会话: {thread_id}）")
    print("输入问题开始，输入 'quit' 退出。")

    while True:
        try:
            user_input = input("\n请输入问题: ").strip()
            if user_input.lower() in ("quit", "exit", "q", "退出"):
                print("再见！")
                break
            if not user_input:
                continue

            start = time.time()
            print("\nAgent is working...\n")
            for chunk in agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            ):
                # 提取模型/工具输出内容
                if isinstance(chunk, dict):
                    for key, value in chunk.items():
                        if key in ("model", "agent"):
                            if isinstance(value, dict):
                                for val in value.values():
                                    if isinstance(val, list):
                                        for item in val:
                                            if hasattr(item, "content") and item.content:
                                                text = item.content
                                                if isinstance(text, str) and text.strip():
                                                    print(text, end="", flush=True)
                                                elif isinstance(text, list):
                                                    for block in text:
                                                        if isinstance(block, dict) and "text" in block:
                                                            print(block["text"], end="", flush=True)
                                    elif hasattr(val, "content") and val.content:
                                        text = val.content
                                        if isinstance(text, str) and text.strip():
                                            print(text, end="", flush=True)
            print(f"\n\n⏱️ 耗时 {time.time() - start:.1f}s")
        except KeyboardInterrupt:
            print("\n再见！")
            break


if __name__ == "__main__":
    import sys
    thread = DEFAULT_THREAD
    if "--thread" in sys.argv:
        idx = sys.argv.index("--thread")
        if idx + 1 < len(sys.argv):
            thread = sys.argv[idx + 1]
    main(thread)
