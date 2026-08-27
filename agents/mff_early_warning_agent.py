"""中频炉预警 Agent：由大模型根据用户问题自动调用中频炉预警智能体的各类 API。

对接项目：Agent_MFF_Early_Warning（中频炉水冷系统多参数融合预警智能体）
接口依据：docs/中频炉预警智能体 OpenAPI 接口文档.md（http://124.65.133.158:9000/docs）

核心特性：
    1. 工具封装    —— 将数据管理/预警分析/故障处置/持续优化四大智能体能力封装为 LangChain @tool
    2. 智能路由    —— LLM 理解用户意图，自动选择并调用合适的工具
    3. 连续对话    —— 多轮对话记忆（Postgres Checkpoint 持久化）
    4. 记忆持久化  —— 重启进程后通过 thread_id 恢复历史

用法：
    python agents/mff_early_warning_agent.py            # 默认会话
    python agents/mff_early_warning_agent.py --thread <id>   # 指定会话
    python agents/mff_early_warning_agent.py --trace    # 开启运行追踪（也可设 MFF_TRACE=1）
"""

import os
import re
import json
import time
import datetime
import logging
from typing import Any

logger = logging.getLogger(__name__)

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from deepagents import create_deep_agent
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 1. 配置与常量
# ---------------------------------------------------------------------------
MFF_BASE_URL = os.getenv("MFF_BASE_URL", "http://124.65.133.158:9000")

# 故障注入类型（物理仿真器支持的四类可复现故障）
FAULT_TYPES = "filter_clog(过滤器堵塞)/pump_cavitation(水泵气蚀)/pipe_leak(管道泄漏)/scale_buildup(线圈结垢)"

HTTP_TIMEOUT = 600.0  # 全链路工作流可能较慢

# Postgres 持久化
POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://postgres:postgres@localhost:5432/langchain_memory",
)
DEFAULT_THREAD = "mff-warning-session-1"

# ---------------------------------------------------------------------------
# 运行追踪（Trace）：记录每次运行调用了哪些工具、传入数据格式与接收到的数据
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACES_DIR = os.path.join(PROJECT_ROOT, "traces")
TRACE_MAX_CHARS = 3000  # 单条追踪记录中请求体/响应体的最大保存长度

# 追踪开关（bool）：默认关闭。开启方式任选其一：
#   1) 环境变量 MFF_TRACE=1        2) 命令行参数 --trace
TRACE_ENABLED = os.getenv("MFF_TRACE", "").strip().lower() in ("1", "true", "yes", "on")


def _truncate(text: str, limit: int = TRACE_MAX_CHARS) -> str:
    """追踪日志用：超长文本截断。"""
    return text if len(text) <= limit else text[:limit] + "...(已截断)"


# ---------------------------------------------------------------------------
# 2. 运行追踪器（AgentTracer）
# ---------------------------------------------------------------------------
def _describe(value: Any) -> Any:
    """将值压缩为"数据格式描述"：类型 + 规模 + 键名，避免把完整数据写进摘要。

    完整原始数据仍单独保存在 input/result 字段中（超长截断）。
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return f"number({value})"
    if isinstance(value, str):
        return value if len(value) <= 60 else f"str(len={len(value)}, 前60字: {value[:60]}...)"
    if isinstance(value, list):
        first = _describe(value[0]) if value else None
        return f"list(n={len(value)}) 示例元素: {first}"
    if isinstance(value, dict):
        items = [f"{k}: {_describe(v)}" for k, v in list(value.items())[:15]]
        suffix = ", ..." if len(value) > 15 else ""
        return "{" + ", ".join(items) + suffix + "}"
    return type(value).__name__


class AgentTracer:
    """单次问答的运行追踪器。

    对每次工具调用记录：
      * tool         —— LLM 调用了哪个工具；
      * input        —— LLM 传入工具的原始参数（JSON 字符串，超长截断）；
      * input_schema —— 各入参的数据格式说明（类型/长度/键名）；
      * http_calls   —— 工具底层发起的 HTTP API 请求（method/path/请求体/响应/耗时）；
      * result       —— 工具收到的返回数据（JSON 字符串预览，超长截断）；
      * elapsed_s    —— 工具执行耗时。

    每轮问答结束后将完整追踪保存为 traces/*.json，并返回控制台摘要。

    通过 ``enabled``（bool）开关控制：默认关闭时所有记录方法为空操作，
    零开销；仅在 MFF_TRACE=1 环境变量或 --trace 命令行参数开启时生效。
    """

    def __init__(self, enabled: bool = TRACE_ENABLED):
        self.enabled = bool(enabled)
        self.thread_id = ""
        self.question = ""
        self.started_at = ""
        self.steps: list[dict] = []
        self._by_call_id: dict[str, int] = {}  # tool_call_id -> step 下标

    def start_run(self, thread_id: str, question: str) -> None:
        """开始一轮新的问答追踪。"""
        if not self.enabled:
            return
        self.thread_id = thread_id
        self.question = question
        self.started_at = datetime.datetime.now().isoformat(timespec="seconds")
        self.steps = []
        self._by_call_id = {}

    def log_tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        """记录 LLM 决定调用某工具及其入参。"""
        if not self.enabled:
            return
        step = {
            "order": len(self.steps) + 1,
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "tool": name,
            "input": _truncate(json.dumps(args, ensure_ascii=False, default=str)),
            "input_schema": _describe(args),
            "http_calls": [],
            "result": "",
            "status": "running",
            "_t0": time.time(),
        }
        self.steps.append(step)
        if call_id:
            self._by_call_id[call_id] = len(self.steps) - 1
        logger.info("[trace] 工具调用 #%d %s 入参格式=%s", step["order"], name, step["input_schema"])

    def log_http(self, method: str, path: str, payload=None, status: int | None = None,
                 response_body: Any = None, error: str = "", elapsed: float = 0.0) -> None:
        """记录一次 HTTP API 调用（由 _request 自动调用）。"""
        if not self.enabled:
            return
        call: dict = {
            "request": f"{method} {MFF_BASE_URL}{path}",
            "payload": _truncate(json.dumps(payload or {}, ensure_ascii=False, default=str)) if payload else "{}",
            "elapsed_s": round(elapsed, 3),
        }
        if response_body is not None:
            call["status"] = status
            call["response"] = _truncate(json.dumps(response_body, ensure_ascii=False, default=str))
        else:
            call["error"] = error
        # 归属到最近一个仍在执行的步骤；无步骤时也要保留（便于排查）
        target = next((s for s in reversed(self.steps) if s["status"] == "running"), None)
        if target is not None:
            target["http_calls"].append(call)
        else:
            self.steps.append({
                "order": len(self.steps) + 1,
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "tool": "(直接 HTTP 调用)",
                "input": "", "input_schema": "",
                "http_calls": [call], "result": "", "status": "done", "_t0": time.time(),
            })
            logger.warning("[trace] 收到无归属的 HTTP 记录 %s %s", method, path)

    def finish(self, idx: int, result_text: str) -> None:
        """记录某个工具步骤的返回数据与耗时。"""
        if not self.enabled:
            return
        try:
            step = self.steps[idx]
        except IndexError:
            return
        step["result_len"] = len(result_text)
        step["result"] = _truncate(result_text or "")
        step["elapsed_s"] = round(time.time() - step.pop("_t0", time.time()), 3)
        step["status"] = "ok"
        logger.info("[trace] 工具 #%d %s 完成 耗时=%.2fs 返回%d字符",
                    step["order"], step["tool"], step["elapsed_s"], step["result_len"])

    def finish_by_call_id(self, call_id: str, result_text: str) -> None:
        """根据 LLM 分配的 tool_call_id 关联并收尾对应工具步骤。"""
        idx = self._by_call_id.get(call_id)
        if idx is not None:
            self.finish(idx, result_text)

    def summarize(self) -> str:
        """生成控制台摘要（一行一个工具步骤）。"""
        if not self.enabled or not self.steps:
            return ""
        lines = ["📋 本次运行工具调用追踪:"]
        for s in self.steps:
            apis = "; ".join(
                f"{c['request']}({c.get('elapsed_s', '?')}s)"
                for c in s["http_calls"]
            ) or "无 HTTP 调用"
            fmt = json.dumps(s["input_schema"], ensure_ascii=False) if isinstance(s["input_schema"], (dict, list)) else str(s["input_schema"])
            err = f" [HTTP错误]" if any(c.get("error") for c in s["http_calls"]) else ""
            lines.append(
                f"  #{s['order']} {s['tool']}{err} | 输入格式: {_truncate(fmt, 200)} "
                f"| 耗时 {s.get('elapsed_s', '-')}s | 返回 {s.get('result_len', 0)} 字符\n"
                f"     APIs: {_truncate(apis, 400)}"
            )
        return "\n".join(lines)

    def save(self) -> str:
        """将完整追踪写入 traces/*.json，返回文件路径。"""
        if not self.enabled or not self.steps:
            return ""
        os.makedirs(TRACES_DIR, exist_ok=True)
        safe_thread = re.sub(r"[^\w.-]", "_", self.thread_id) or "session"
        fname = f"mff-trace_{safe_thread}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(TRACES_DIR, fname)
        out = {
            "agent": "mff_early_warning_agent",
            "thread_id": self.thread_id,
            "started_at": self.started_at,
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "question": self.question,
            "total_tool_calls": len(self.steps),
            "steps": [{k: v for k, v in s.items() if k != "_t0"} for s in self.steps],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info("[trace] 追踪已保存: %s", path)
        return path


tracer = AgentTracer()


# ---------------------------------------------------------------------------
# 3. 中频炉预警 API 客户端
# ---------------------------------------------------------------------------
def _request(method: str, path: str, json_data: dict | None = None) -> Any:
    """调用中频炉预警智能体 REST API，返回 data 字段。

    业务约定：成功返回 {"code": 0, "data": {...}}，失败返回非 0 code。
    所有请求/响应均自动写入运行追踪。
    """
    url = f"{MFF_BASE_URL}{path}"
    t0 = time.time()
    try:
        resp = httpx.request(method, url, json=json_data, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        tracer.log_http(method, path, json_data, status=resp.status_code,
                        response_body=body, elapsed=time.time() - t0)
    except Exception as exc:
        tracer.log_http(method, path, json_data,
                        error=f"{type(exc).__name__}: {exc}", elapsed=time.time() - t0)
        raise
    if isinstance(body, dict) and "code" in body:
        if body.get("code") != 0:
            raise RuntimeError(f"API {path} 返回错误: {body.get('detail') or body}")
        return body.get("data")
    return body


def _get(path: str) -> Any:
    return _request("GET", path)


def _post(path: str, json_data: dict) -> Any:
    return _request("POST", path, json_data)


def _to_json(obj: Any) -> str:
    """工具统一返回：超长序列截断，避免撑爆上下文。"""
    text = json.dumps(obj, ensure_ascii=False, default=str)
    MAX_CHARS = 15000
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS] + "\n...(结果过长已截断)"
    return text


# ---------------------------------------------------------------------------
# 3. 工具定义（LLM 可调用的能力）
# ---------------------------------------------------------------------------
@tool(parse_docstring=True)
def collect_sensor_data(duration: int = 300, fault: str = "", fault_start: int = 120, severity: float = 0.9) -> str:
    """采集中频炉水冷系统传感器数据（由物理仿真器供数）。

    用于获取一段规整的时序监测数据；可选择注入故障以模拟异常工况。

    Args:
        duration: 采集时长（秒），范围 1–86400，默认 300。
        fault: 可选注入的故障类型，取值 filter_clog/pump_cavitation/pipe_leak/scale_buildup，
            留空则采集正常数据。
        fault_start: 故障起始时刻（秒），默认 120。
        severity: 故障严重度 0–1，默认 0.9。

    Returns:
        JSON 字符串，包含 count 与 records（含进/出水温度、压力、流量、湿度等字段）。
    """
    payload = {"duration": int(duration), "fault_start": int(fault_start), "severity": float(severity)}
    if fault:
        payload["fault"] = fault
    result = _post("/api/v1/agents/data-manager/collect", payload)
    # records 数量可能很大，仅保留统计摘要和首尾样本供 LLM 参考
    if isinstance(result, dict) and isinstance(result.get("records"), list):
        records = result["records"]
        result["count"] = len(records)
        if len(records) > 20:
            fault_period = [r for r in records[fault_start:] if fault]
            sample = fault_period[-10:] if fault_period else records[-10:]
            result["records"] = records[:5] + sample
            result["_note"] = f"共{len(records)}条记录，已仅展示前5条+故障期最后10条"
    return _to_json(result)


@tool(parse_docstring=True)
def get_data_schema() -> str:
    """查询 L1/L2 预警分析可直接使用的数据格式契约（字段/单位/精度）。

    当需要了解传感器数据的字段定义、单位或构造接入记录时调用。

    Returns:
        JSON 字符串，包含各字段的类型、单位与说明。
    """
    result = _get("/api/v1/agents/data-manager/schema")
    return _to_json(result)


@tool(parse_docstring=True)
def analyze_warnings(records: list) -> str:
    """对传感器数据进行 L1/L2/L3 多级预警分析（核心工具）。

    分析链路：L1 规则预警 → L2 异常检测/趋势预测 → L3 大模型根因诊断
    （自动注入知识图谱/维修工单/工况表上下文），输出预警等级、根因、证据链与处置 SOP。

    Args:
        records: 传感器数据数组（来自 collect_sensor_data 或用户提供的记录），
            每条至少包含 timestamp 与若干传感器字段。

    Returns:
        JSON 字符串，包含 level（none/yellow/orange/red）、l1 告警、l2 异常分与预测、
        l3 根因诊断结果。
    """
    result = _post("/api/v1/agents/warning-analyzer/analyze", {"records": records})
    return _to_json(result)


@tool(parse_docstring=True)
def l1_rule_warning(records: list) -> str:
    """对数据执行 L1 规则预警检测（快速阈值判断）。

    适用于只想快速判断当前读数是否触发超限规则（如出水温度过高、压差偏移等），
    不需要 L2 模型预测与 L3 根因诊断的场景。

    Args:
        records: 传感器数据数组，每条至少包含 timestamp 与相关传感器字段。

    Returns:
        JSON 字符串，包含触发的告警数量与告警明细（rule_id/level/message/value）。
    """
    result = _post("/api/v1/warn/l1", {"records": records})
    return _to_json(result)


@tool(parse_docstring=True)
def diagnose_root_cause(features: dict, condition: str = "") -> str:
    """根据异常特征进行 L3 大模型根因诊断。

    用户已知部分异常读数、希望判断故障根因（如管道泄漏/过滤器堵塞/水泵气蚀/
    线圈结垢等）、置信度、证据链及处置 SOP 时调用。

    Args:
        features: 异常特征字典，如 {"outlet_temp": 56.2, "pressure": 175.0, "flow_rate": 6.1}。
        condition: 工况上下文，可选 startup/melting/holding/tapping/idle，
            缺省为 unknown。

    Returns:
        JSON 字符串，包含 root_cause/confidence/evidence/sop/level/hallucination_check 等。
    """
    payload: dict = {"features": features}
    if condition:
        payload["condition"] = condition
    result = _post("/api/v1/diagnose", payload)
    return _to_json(result)


@tool(parse_docstring=True)
def handle_fault(analysis: dict) -> str:
    """根据预警分析结果生成运维工单、联动应急预案并完成分级通知。

    用户要求"开工单""生成处置方案""通知运维"或完成故障闭环处理时调用；
    analysis 应传入 analyze_warnings 的分析结果（至少包含 level 与 l3 字段）。

    Args:
        analysis: 预警分析智能体返回的分析结果对象。

    Returns:
        JSON 字符串，包含 order_id/level/sop/spare_parts/emergency_plan/push_records。
    """
    result = _post("/api/v1/agents/fault-handler/handle", {"analysis": analysis})
    return _to_json(result)


@tool(parse_docstring=True)
def create_work_order(features: dict, condition: str = "") -> str:
    """根据异常特征直接生成运维工单（快捷方式）。

    用户不需要完整预警分析、直接想由特征值出工单时使用。

    Args:
        features: 异常特征字典，如 {"pressure": 150.0, "cabinet_humidity": 75.0}。
        condition: 工况上下文，可选 startup/melting/holding/tapping/idle。

    Returns:
        JSON 字符串，包含工单全部字段 + emergency_plan + push_records。
    """
    payload: dict = {"features": features}
    if condition:
        payload["condition"] = condition
    result = _post("/api/v1/workorder", payload)
    return _to_json(result)


@tool(parse_docstring=True)
def submit_feedback(order_id: str, actual_root_cause: str, is_true_fault: bool,
                    handling_time_min: float, effect: str) -> str:
    """归档处置反馈，积累真实故障样本用于持续优化。

    处置完成后，用户提供实际根因、耗时与效果时调用。

    Args:
        order_id: 工单号（如 WO-20260821-0007）。
        actual_root_cause: 实际根因（如 管道泄漏）。
        is_true_fault: 是否为真实故障（False 表示误报）。
        handling_time_min: 处置耗时（分钟）。
        effect: 处置效果描述（如 更换密封圈后恢复）。

    Returns:
        JSON 字符串，包含归档状态、反馈统计与自动微调触发情况。
    """
    result = _post("/api/v1/feedback", {
        "order_id": order_id,
        "actual_root_cause": actual_root_cause,
        "is_true_fault": bool(is_true_fault),
        "handling_time_min": float(handling_time_min),
        "effect": effect,
    })
    return _to_json(result)


@tool(parse_docstring=True)
def update_knowledge(component: str, action: str, note: str = "", date: str = "") -> str:
    """向知识库新增维修工单记录（将作为后续 L3 诊断的上下文）。

    处置完成后把维修经验沉淀进知识库时调用，例如"把这次换密封圈的经验记下来"。

    Args:
        component: 部件名称（如 管道/过滤器/水泵/线圈）。
        action: 处置动作描述。
        note: 备注（可选）。
        date: 日期字符串 YYYY-MM-DD（可选，缺省为服务端当天）。

    Returns:
        JSON 字符串，包含更新状态与知识库规模统计。
    """
    payload: dict = {"component": component, "action": action}
    if note:
        payload["note"] = note
    if date:
        payload["date"] = date
    result = _post("/api/v1/agents/optimizer/update-knowledge", payload)
    return _to_json(result)


@tool(parse_docstring=True)
def query_optimizer_status() -> str:
    """查询持续优化状态：反馈统计、微调触发情况、知识库规模。

    用户询问"优化进展""知识库情况""误报统计"时调用。

    Returns:
        JSON 字符串，包含 feedback_stats/retrain_due/knowledge_base。
    """
    result = _get("/api/v1/agents/optimizer/status")
    return _to_json(result)


@tool(parse_docstring=True)
def query_forecast_model() -> str:
    """查询当前 L2 趋势预测模型及可用模型列表。

    用户询问预测模型配置时可调用。

    Returns:
        JSON 字符串，包含 current（当前模型）/available/horizon_s。
    """
    result = _get("/api/v1/forecast-model")
    return _to_json(result)


@tool(parse_docstring=True)
def run_full_workflow(duration: int = 600, fault: str = "pipe_leak",
                      fault_start: int = 180, severity: float = 0.9) -> str:
    """一键运行四大智能体全链路演示：数据采集 → 预警分析 → 故障处置 → 持续优化。

    用户说"跑一遍完整流程""演示全链路""做一次端到端测试"时调用。
    注意：duration 较长时耗时明显增加；pipe_leak 需足够预热时间才能触发预警。

    Args:
        duration: 数据时长（秒），默认 600。
        fault: 注入故障，filter_clog/pump_cavitation/pipe_leak/scale_buildup，默认 pipe_leak。
        fault_start: 故障起始（秒），默认 180。
        severity: 故障严重度 0–1，默认 0.9。

    Returns:
        JSON 字符串，包含 warning/work_order/optimization 及各环节 trace 与总耗时。
    """
    result = _post("/api/v1/workflow/run", {
        "duration": int(duration),
        "fault": fault,
        "fault_start": int(fault_start),
        "severity": float(severity),
    })
    return _to_json(result)


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
        #max_tokens=8192,
        streaming=True,
        timeout=300,
        max_retries=1,
    )


# ---------------------------------------------------------------------------
# 5. Agent 构建与 Postgres 持久化
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    f"你是一个中频炉水冷系统预警助手，负责根据用户的问题调用中频炉预警智能体的 API 来完成"
    f"数据采集、多级预警分析、根因诊断、工单处置与持续优化。\n\n"
    f"当前真实日期：{datetime.date.today().strftime('%Y年%m月%d日')}。"
    f"涉及任何日期判断时，必须以该日期为基准。\n\n"
    f"""可用工具及适用场景：
1. collect_sensor_data - 采集仿真器传感器数据（可选注入故障：{FAULT_TYPES}）
2. get_data_schema - 查询传感器数据格式契约（字段/单位）
3. analyze_warnings - L1/L2/L3 多级预警分析（核心分析入口）
4. l1_rule_warning - 仅快速 L1 规则阈值判断
5. diagnose_root_cause - 已知异常特征时的 L3 根因诊断
6. handle_fault - 由分析结果生成工单+应急+通知（故障闭环）
7. create_work_order - 由特征值直接快捷生成工单
8. submit_feedback - 归档处置反馈（实际根因/耗时/效果）
9. update_knowledge - 把维修经验沉淀进知识库
10. query_optimizer_status - 查询优化/微调/知识库状态
11. query_forecast_model - 查询 L2 预测模型配置
12. run_full_workflow - 四大智能体一键串联全链路演示

典型工作流：
- 完整闭环："采集 pipe_leak 数据"→ analyze_warnings → handle_fault →（用户给反馈后）submit_feedback
- 快速判断："这个读数正常吗"→ l1_rule_warning
- 经验沉淀："这次维修记到知识库"→ update_knowledge
- 综合演示："跑一遍完整流程"→ run_full_workflow

工作原则：
- 分析用户意图选择最合适的工具；复杂任务可串联多个工具
- 数据中的 records 参数要原样传给下游工具，不要截断或改写字段名
- 调用工具后，将结果整理成清晰、结构化的中文回答，明确标注预警等级、根因、
  证据链与建议措施；高风险（orange/red）结论要用醒目方式提示
- 对用户提供的具体数值，若无必要勿擅自编造；无法获得的工具结果要如实说明
"""
)


def build_agent(checkpointer: PostgresSaver):
    """构建中频炉预警对话式 Agent。"""
    return create_deep_agent(
        model=build_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=[
            collect_sensor_data,
            get_data_schema,
            analyze_warnings,
            l1_rule_warning,
            diagnose_root_cause,
            handle_fault,
            create_work_order,
            submit_feedback,
            update_knowledge,
            query_optimizer_status,
            query_forecast_model,
            run_full_workflow,
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
def main(thread_id: str = DEFAULT_THREAD, trace: bool | None = None) -> None:
    """交互式连续对话入口。

    Args:
        thread_id: 会话 ID。
        trace: 追踪开关（bool）。None 表示由环境变量 MFF_TRACE 决定；
            显式传入时以该值为准（命令行 --trace 即此场景）。
    """
    from langchain_core.messages import HumanMessage, ToolMessage

    tracer.enabled = TRACE_ENABLED if trace is None else bool(trace)
    config = {"configurable": {"thread_id": thread_id}}
    print(f"中频炉预警 Agent 已就绪（会话: {thread_id}，API: {MFF_BASE_URL}）")
    if tracer.enabled:
        print(f"运行追踪已开启（MFF_TRACE/--trace），结果保存到: {TRACES_DIR}")
    else:
        print("运行追踪未开启（设置 MFF_TRACE=1 或加 --trace 参数可开启）")
    print("输入问题开始，输入 'quit' 退出。")

    while True:
        try:
            user_input = input("\n请输入问题: ").strip()
            if user_input.lower() in ("quit", "exit", "q", "退出"):
                print("再见！")
                break
            if not user_input:
                continue

            tracer.start_run(thread_id, user_input)
            start = time.time()
            print("\nAgent is working...\n")
            for chunk in agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="updates",
            ):
                if not isinstance(chunk, dict):
                    continue
                for key, value in chunk.items():
                    if not isinstance(value, dict):
                        continue
                    for val in value.values():
                        msgs = val if isinstance(val, list) else [val]
                        for item in msgs:
                            # —— 追踪1：LLM 的工具调用决策（工具名 + 入参）——
                            for tc in getattr(item, "tool_calls", None) or []:
                                tracer.log_tool_call(tc.get("name"), tc.get("args"), tc.get("id"))
                            # —— 追踪2：工具执行返回的数据 ——
                            if isinstance(item, ToolMessage):
                                content = item.content
                                if not isinstance(content, str):
                                    content = json.dumps(content, ensure_ascii=False, default=str)
                                tracer.finish_by_call_id(getattr(item, "tool_call_id", ""), content)
                            # 打印模型回复文本
                            if key in ("model", "agent"):
                                text = getattr(item, "content", None)
                                if isinstance(text, str) and text.strip():
                                    print(text, end="", flush=True)
                                elif isinstance(text, list):
                                    for block in text:
                                        if isinstance(block, dict) and "text" in block:
                                            print(block["text"], end="", flush=True)
            print(f"\n\n⏱️ 耗时 {time.time() - start:.1f}s")
            # 保存并展示本轮运行追踪
            if tracer.steps:
                print(tracer.summarize())
                print(f"🗂️ 完整追踪已保存: {tracer.save()}")
        except KeyboardInterrupt:
            print("\n再见！")
            break


if __name__ == "__main__":
    import sys
    thread = DEFAULT_THREAD
    # 仅当显式给出 --trace 时传入 True；否则传 None 交给环境变量 MFF_TRACE 决定
    trace_arg = True if "--trace" in sys.argv else None
    if "--thread" in sys.argv:
        idx = sys.argv.index("--thread")
        if idx + 1 < len(sys.argv):
            thread = sys.argv[idx + 1]
    main(thread, trace=trace_arg)
