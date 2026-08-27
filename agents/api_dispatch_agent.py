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
import logging
from typing import Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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
    "AGENT_HELPER_BASE_URL", "http://172.25.67.120:8359"
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


def _split_text(text: str, max_chars: int = 20000) -> list:
    """将长文本按字符阈值切块（退化回退用，避免超子 LLM 上下文上限）。"""
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks, cur, cur_len = [], "", 0
    for ln in text.splitlines():
        if cur_len + len(ln) > max_chars and cur:
            chunks.append(cur)
            cur, cur_len = "", 0
        cur += ln + "\n"
        cur_len += len(ln) + 1
    if cur:
        chunks.append(cur)
    return chunks


def _load_device_blocks(device_text: str, save_path: str, pdf_path: str, device_text_original: str = "") -> list:
    """从 mineru 结构化中间文件读取按版面切好的文本块。

    mineru 在 PDF 同级 ``mineru_result`` 下产出多份中间文件：
      * ``*_device_original.txt`` —— 原始抽取文本（JSON 行，每行 ``{"text": ...}``），
        内容最完整，优先使用；
      * ``*_device.txt`` —— 后处理（按标题分块）版本，某些文档类型会被切成空块，
        需作为回退；
      * ``*_device.json`` —— 早期格式（``[{"block": N, "text": ...}]`` 列表）。

    从中提取文本块，按块合并、分批送入实体抽取，避免一次性读入整份大文件触发
    子 LLM 上下文上限（约 40960 token）而 400。

    Args:
        device_text: ``textExtractDir`` 返回的 device 文本路径（后处理版）。
        device_text_original: ``textExtractDir`` 返回的原始 device 文本路径。
        save_path: mineru 结果目录。
        pdf_path: 原始 PDF 路径，用于推断候选文件名。

    Returns:
        文本块字符串列表（无可用内容时返回空列表）。
    """
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    candidates = []
    # 优先原始文本，其次后处理文本，再次按命名推断
    for p in (device_text_original, device_text):
        if p and os.path.isfile(p):
            candidates.append(p)
    candidates += [
        os.path.join(save_path, f"{basename}_device_original.txt"),
        os.path.join(save_path, f"{basename}_device.txt"),
        os.path.join(save_path, f"{basename}_device.json"),
    ]

    MIN_CHARS = 200  # 低于此长度视为退化（空块）文件，跳过

    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    blocks = [b.get("text", "") for b in data if isinstance(b, dict) and b.get("text")]
                    if sum(len(b) for b in blocks) >= MIN_CHARS:
                        return blocks
                continue
            # .txt：可能是 JSON 行（每行一个 {"text": ...}）或纯文本
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            stripped = raw.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                # JSON 行：逐行解析取 text 字段
                blocks = []
                for ln in raw.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        txt = obj.get("text") or ""
                        if isinstance(txt, str) and txt.strip():
                            blocks.append(txt)
                if sum(len(b) for b in blocks) >= MIN_CHARS:
                    return blocks
                continue
            # 纯文本：整段读入后自行切块
            if len(raw) >= MIN_CHARS:
                return _split_text(raw)
        except Exception:
            continue
    return []


def _retrieve_relevant_blocks(rule: str, milvus_uuid: str, top_k: int = 10) -> list:
    """通过向量检索召回与查询最相关的文本块，避免对所有块都做实体抽取。

    调用 agent-helper 的 ``/api/milvus/search``，按 ``rule``（用户问题）在
    对应 collection（``milvus_uuid``）中做混合检索（Dense + BM25 + RRF），
    返回 top_k 个文本块内容。

    Args:
        rule: 检索问题/规则（通常是用户当前提问）。
        milvus_uuid: 解析阶段返回的 milvus collection 名称。
        top_k: 召回块数，默认 10。

    Returns:
        相关文本块字符串列表；检索失败或参数为空时返回空列表，调用方回退到全量。
    """
    if not rule or not milvus_uuid:
        return []
    try:
        resp = _post_object("/api/milvus/search", {
            "rule": rule,
            "file_uuid": milvus_uuid,
            "top_k": top_k,
        })
    except Exception as e:
        logger.warning("向量检索失败，回退全量抽取: %s", e)
        return []
    hits = resp.get("data") if isinstance(resp, dict) else resp
    if not isinstance(hits, list):
        return []
    blocks = []
    for hit in hits:
        ent = hit.get("entity") if isinstance(hit, dict) else None
        if not isinstance(ent, dict):
            continue
        text = ent.get("text") or ent.get("title_parent") or ent.get("title") or ""
        text = text.strip() if isinstance(text, str) else ""
        if text:
            blocks.append(text)
    return blocks


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
def extract_entities(pdf_path: str, entity_types: dict = None, file_id: str = "", query: str = "") -> str:
    """从文档中抽取用户指定的任意实体/要素。

    通过 agent-helper 的 /api/agent/extractEntity 接口抽取，可针对合同、投标书、
    说明书等任意文档提取用户关心的字段（如响应人地址、联系人、资质要求、交付
    日期、技术参数等），而非固定的一组合同要素。

    Args:
        pdf_path: 文档 PDF 的服务器绝对路径。
        entity_types: 要抽取的实体定义（dict[str, str]，键为实体名、值为描述）。
            例如 {"响应人地址": "响应人的注册或办公地址", "联系人": "项目联系人姓名"}。
            不传则抽取一组常用合同要素。
        file_id: 文件唯一标识（可选）。
        query: 用户当前关注的问题（可选）。提供后仅对向量检索召回的 top-10
            相关文本块抽取，显著减少大文档的 token 消耗与耗时；不提供则对
            全篇文本块抽取。

    Returns:
        JSON 字符串：键为实体名，值为抽取结果（缺失为 null）。
    """
    if not entity_types:
        entity_types = CONTRACT_ENTITY_TYPES
    file_id = _sanitize_file_id(file_id)
    # 先解析文本，再抽取实体
    save_path = os.path.join(os.path.dirname(pdf_path), "mineru_result")
    device_text = ""
    milvus_uuid = ""
    try:
        parse_res = _post_object("/api/mineru/textExtractDir", {
            "file_list": [{
                "file_path": pdf_path,
                "save_path": save_path,
                "file_id": file_id,
            }]
        })
        if isinstance(parse_res, list) and parse_res:
            device_text = parse_res[0].get("device_text", "")
            device_text_original = parse_res[0].get("device_text_original", "")
            milvus_uuid = parse_res[0].get("milvus_uuid", "")
    except Exception:
        # embedding 失败时推断产物路径
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        device_text = os.path.join(save_path, f"{basename}_device.txt")

    # 优先通过向量检索召回与 query 最相关的文本块；未提供 query 或检索失败时，
    # 回退为读取 mineru 在 PDF 同级 mineru_result 下产出的 *_device.json（已按
    # 版面切好的自然文本块），避免一次性读入整份大文件。
    device_blocks = []
    if query:
        device_blocks = _retrieve_relevant_blocks(query, milvus_uuid, top_k=10)
    if not device_blocks:
        device_blocks = _load_device_blocks(device_text, save_path, pdf_path, device_text_original)
    if not device_blocks:
        return json.dumps({}, ensure_ascii=False)

    # 按字符阈值合并相邻块，每块作为独立 file 送入实体抽取接口，再合并结果，
    # 避免单块超出子 LLM 约 40960 token 的上限导致 400。
    MAX_CHARS = 5000
    chunks, cur, cur_len = [], [], 0
    for blk in device_blocks:
        if cur_len + len(blk) > MAX_CHARS and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(blk)
        cur_len += len(blk)
    if cur:
        chunks.append("\n".join(cur))

    file_list = [{
        "file_name": f"{os.path.basename(pdf_path)}#part{i + 1}",
        "text": ch,
        "entity_types": entity_types,
        "task_constraint": "从合同文本中抽取上述要素，缺失字段为 null",
        "milvus_uuid": milvus_uuid,
    } for i, ch in enumerate(chunks)]

    raw = _post_object("/api/agent/extractEntity", {"file_list": file_list})

    # 合并多块结果：同一实体键保留首个非空值
    merged: dict = {}
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        if not isinstance(item, dict):
            continue
        part = item.get("result") or {}
        if not isinstance(part, dict):
            continue
        for key, val in part.items():
            if key not in merged or merged[key] is None:
                if val is not None:
                    merged[key] = val
    return json.dumps(merged, ensure_ascii=False)


@tool(parse_docstring=True)
def compliance_review(pdf_path: str, rules: list = None) -> str:
    """对合同/文档进行要素一致性合规审查。

    适用于检查合同金额大小写是否一致、签订日期是否在有效期内、
    主体名称是否完整等合规性问题。

    Args:
        pdf_path: 合同/文档 PDF 的服务器绝对路径。
        rules: 审查规则列表（list[str]，可选）。每条为一条自然语言审查规则，
            例如 "核查合同总金额大小写是否一致"。不传则检查金额、日期、主体名称。

    Returns:
        JSON 字符串，包含每条规则的审查结果、风险内容与建议。
    """
    # 子模型（Qwen3.6-27B-INT4）无法准确获知真实当前日期，会把年份误判为训练
    # 截止前的年份，导致“日期是否在有效期内”等审查结论错误。这里把真实当前日期
    # 显式注入到每条规则中，作为模型判断的时间基准。
    today = datetime.date.today().strftime("%Y年%m月%d日")
    if rules is None:
        rules = [
            f"当前真实日期为{today}。核查合同总金额、大小写金额是否一致，若不一致需指出",
            f"当前真实日期为{today}。核查合同签订日期是否在有效期范围内，若超出或明显不合理需指出",
            f"当前真实日期为{today}。核查甲方、乙方名称是否完整且正确，与营业执照名称是否一致",
        ]
    else:
        rules = [f"当前真实日期为{today}。{r}" for r in rules]
    save_path = os.path.join(os.path.dirname(pdf_path), "mineru_result")
    basename = os.path.splitext(os.path.basename(pdf_path))[0]

    # complianceAuditDir 按行读取 JSON-lines 文件并取每行 text 字段，需使用
    # 原始抽取文本（device_text_original），不能用后处理会切成空块的 device.txt。
    device_text = os.path.join(save_path, f"{basename}_device_original.txt")
    try:
        parse_res = _post_object("/api/mineru/textExtractDir", {
            "file_list": [{
                "file_path": pdf_path,
                "save_path": save_path,
                "file_id": _sanitize_file_id(""),
            }]
        })
        if isinstance(parse_res, list) and parse_res:
            device_text = parse_res[0].get("device_text_original", device_text)
    except Exception:
        pass

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
    result = _post_raw("/api/detection/findBlankPages", pdf_path)
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
    result = _post_raw("/api/detection/catalogue", pdf_path)
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 4. 模型
# ---------------------------------------------------------------------------
def build_model() -> ChatOpenAI:
    """构建支持工具调用的 LLM。"""
    return ChatOpenAI(
        base_url=os.getenv("QWEN3.8_BASE_URL"),
        api_key=os.getenv("QWEN3.8_API_KEY"),
        model=os.getenv("QWEN3.8_MODEL_NAME"),
        temperature=0.3,
        max_tokens=8192,
        streaming=True,
        timeout=300,
        max_retries=1,
    )


# ---------------------------------------------------------------------------
# 5. Agent 构建与 Postgres 持久化
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    f"你是一个智能文档审查与合同分析助手，负责根据用户的问题自动调用合适的 API 工具来处理文档。\n\n"
    f"当前真实日期：{datetime.date.today().strftime('%Y年%m月%d日')}。"
    f"涉及任何日期、期限、有效期判断时，必须以该日期为基准，不要使用你训练数据中的日期。\n\n"
    """可用工具及适用场景：
1. extract_document_text - 提取 PDF 全文内容（查找条款、分析内容）
2. extract_entities - 抽取任意实体/要素（可针对合同、投标书、说明书等任意文档）。
   把用户想提取的字段通过 entity_types 参数传入（格式 {"实体名": "描述"}，例如
   {"响应人地址": "响应人的注册或办公地址", "联系人": "项目联系人姓名"}）；不传则抽取
   一组常用合同要素。务必把用户当前问题填入 query 参数（如问"响应人地址"传
   query="响应人地址"），工具只对向量检索召回的 top-10 相关块抽取，大幅降低大文档的
   耗时与 token 消耗；全局性要素可不传 query。
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
)


def build_agent(checkpointer: PostgresSaver):
    """构建 API 调度 Agent。"""
    return create_deep_agent(
        model=build_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=[
            extract_document_text,
            extract_entities,
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
                stream_mode="updates",
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
