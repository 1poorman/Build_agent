"""
合同类文件审查自动化 Agent
============================
基于 agent-helper 服务（FastAPI, 默认端口 8018）提供的文档解析、LLM 抽取、
合规审查、印章检测等能力，编排一个端到端的合同审查流水线。

流程:
    1. parse_document   - 调用 MinerU 提取合同文本
    2. extract_entities - 抽取合同关键要素（甲乙双方、金额、日期、付款方式等）
    3. compliance_check - 要素一致性核查（金额/日期/主体）
    4. amount_check     - 合同金额与付款方式核查
    5. seal_check       - 盖章页签名 / 印章检测
    6. generate_report  - 汇总全部结果，生成 Markdown 审查报告

用法:
    python agents/contract_review_agent.py <合同PDF路径> [文件ID]
"""

import os
import json
import time
import datetime
from typing import TypedDict, Any, Optional

import httpx
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
# agent-helper 服务地址（按实际部署环境修改）
AGENT_HELPER_BASE_URL = os.getenv(
    "AGENT_HELPER_BASE_URL", "http://172.25.67.120:8018"
)

# 报告输出目录（项目根目录下 reports/）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# HTTP 超时（MinerU 解析大文件较慢）
TIMEOUT = 600.0

# 合同审查要抽取的要素
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

# 一致性核查规则（审查对象为从合同中抽取的要素）
CONTRACT_COMPLIANCE_RULES = [
    {
        "rule": "核查合同总金额、大小写金额是否一致，若不一致需指出",
        "object_key": "合同总金额",
    },
    {
        "rule": "核查合同签订日期是否在有效期范围内，若超出需指出",
        "object_key": "合同签订日期",
    },
    {
        "rule": "核查甲方、乙方名称是否完整且正确，与营业执照名称是否一致",
        "object_key": "甲方名称",
    },
]


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------
class ContractReviewState(TypedDict, total=False):
    pdf_path: str                    # 合同 PDF 路径
    file_id: str                     # 文件 ID（向量库关联）
    parse_result: Any                # MinerU 解析结果
    document_text: str               # 合同全文
    entities: dict                   # 抽取的合同要素
    compliance_results: list         # 一致性核查结果
    amount_result: dict              # 金额核查结果
    seal_result: dict                # 印章/签名检测结果
    report: str                      # 最终报告
    errors: list                     # 错误信息收集


# ---------------------------------------------------------------------------
# 工具函数：调用 agent-helper API
# ---------------------------------------------------------------------------
def _post(path: str, json_data: dict) -> dict:
    """POST 到 agent-helper 服务并返回 data 字段。"""
    url = f"{AGENT_HELPER_BASE_URL}{path}"
    resp = httpx.post(url, json=json_data, timeout=TIMEOUT)
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 200:
        raise RuntimeError(f"API {path} 返回错误: {body.get('msg')}")
    return body.get("data")


# ---------------------------------------------------------------------------
# 流程节点
# ---------------------------------------------------------------------------
def parse_document(state: ContractReviewState) -> dict:
    """步骤 1: 调用 MinerU 提取合同文本。"""
    pdf_path = state["pdf_path"]
    file_id = state.get("file_id", str(int(time.time())))

    save_path = os.path.join(
        REPORTS_DIR, f"mineru_{os.path.splitext(os.path.basename(pdf_path))[0]}"
    )

    result = _post("/api/mineru/textExtractDir", {
        "file_list": [{
            "file_path": pdf_path,
            "save_path": save_path,
            "file_id": file_id,
        }]
    })

    # 提取纯文本（从 device_text 文件读取，或直接用 text）
    document_text = ""
    if isinstance(result, list) and result:
        info = result[0]
        text_path = info.get("text")
        if text_path and os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                document_text = f.read()
        if not document_text:
            document_text = info.get("text", "") or ""

    return {
        "parse_result": result,
        "document_text": document_text,
        "file_id": file_id,
    }


def extract_entities(state: ContractReviewState) -> dict:
    """步骤 2: 抽取合同关键要素。"""
    result = _post("/api/agent/extractEntity", {
        "file_list": [{
            "file_name": os.path.basename(state["pdf_path"]),
            "file_content": state["document_text"][:20000],  # 控制 token
            "entity_types": CONTRACT_ENTITY_TYPES,
            "task_constraint": "从合同文本中抽取上述要素，缺失字段为 null",
        }]
    })

    entities = {}
    if isinstance(result, list) and result:
        # 兼容返回格式：可能是 [{"要素": 值}] 或 [{"result": {...}}]
        first = result[0]
        if isinstance(first, dict):
            entities = first.get("result", first)
        elif isinstance(first, str):
            try:
                entities = json.loads(first)
            except Exception:
                entities = {"raw": first}

    return {"entities": entities}


def compliance_check(state: ContractReviewState) -> dict:
    """步骤 3: 要素一致性核查。"""
    # 构造审查对象：将抽取的要素作为文本提供给一致性核查
    object_text = json.dumps(state.get("entities", {}), ensure_ascii=False, indent=2)

    review_list = []
    for rule in CONTRACT_COMPLIANCE_RULES:
        review_list.append({
            "object": {"text": object_text},
            "rule": rule["rule"],
            "knowledge_base_ids": [],
            "preprocess": True,
        })

    result = _post("/api/agent/complianceAudit", {"review_list": review_list})

    compliance_results = result if isinstance(result, list) else [result]
    return {"compliance_results": compliance_results}


def amount_check(state: ContractReviewState) -> dict:
    """步骤 4: 合同金额与付款方式核查。"""
    result = _post("/api/Contract/verifyContractAmount", {
        "pdf_path": state["pdf_path"],
        "file_id": state.get("file_id", ""),
    })
    return {"amount_result": result if isinstance(result, dict) else {"raw": result}}


def seal_check(state: ContractReviewState) -> dict:
    """步骤 5: 盖章页签名与印章检测。"""
    seal_result = {}
    try:
        seal_result["signature_with_seal"] = _post(
            "/api/Contract/checkSignaturewithSeal",
            {"pdf_path": state["pdf_path"]},
        )
    except Exception as e:
        seal_result["signature_with_seal"] = {"error": str(e)}

    try:
        seal_result["seal_detection"] = _post(
            "/api/detection/detectSeal",
            {"pdf_path": state["pdf_path"]},
        )
    except Exception as e:
        seal_result["seal_detection"] = {"error": str(e)}

    return {"seal_result": seal_result}


def generate_report(state: ContractReviewState) -> dict:
    """步骤 6: 汇总生成 Markdown 审查报告。"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# 合同审查报告",
        f"",
        f"- **文件**: `{state['pdf_path']}`",
        f"- **审查时间**: {now}",
        f"",
    ]

    # 1. 合同要素
    lines.append("## 一、合同要素抽取")
    entities = state.get("entities", {})
    if entities:
        for k, v in entities.items():
            lines.append(f"- **{k}**: {v if v is not None else '（未提取到）'}")
    else:
        lines.append("- 未能抽取到合同要素")
    lines.append("")

    # 2. 金额核查
    lines.append("## 二、合同金额核查")
    amount = state.get("amount_result", {})
    if isinstance(amount, dict):
        lines.append(f"- 合同金额: {amount.get('contract_amount', 'N/A')}")
        lines.append(f"- 付款方式: {amount.get('payment_method', 'N/A')}")
        lines.append(f"- 分期合计: {amount.get('installment_total', 'N/A')}")
        matched = amount.get("amount_matched")
        if matched is True:
            lines.append("- **金额一致性: ✅ 通过**")
        elif matched is False:
            lines.append("- **金额一致性: ❌ 不通过**")
        else:
            lines.append("- 金额一致性: ⚠️ 无法判定")
        if amount.get("message"):
            lines.append(f"- 说明: {amount['message']}")
    lines.append("")

    # 3. 一致性核查
    lines.append("## 三、要素一致性核查")
    for i, r in enumerate(state.get("compliance_results", []), 1):
        if isinstance(r, dict):
            conclusion = r.get("conclusion") or r.get("result") or json.dumps(r, ensure_ascii=False)
        else:
            conclusion = str(r)
        lines.append(f"{i}. {conclusion}")
    lines.append("")

    # 4. 印章检测
    lines.append("## 四、盖章与签名检测")
    seal = state.get("seal_result", {})
    for k, v in seal.items():
        if isinstance(v, dict) and "error" in v:
            lines.append(f"- {k}: ⚠️ 检测失败 ({v['error']})")
        else:
            lines.append(f"- {k}: {json.dumps(v, ensure_ascii=False)[:500]}")
    lines.append("")

    # 5. 审查结论
    lines.append("## 五、审查结论")
    issues = []
    if state.get("errors"):
        issues.extend(state["errors"])
    if isinstance(amount, dict) and amount.get("amount_matched") is False:
        issues.append("合同金额与分期付款合计不一致")
    lines.append("")
    if issues:
        lines.append("**发现以下问题需关注：**")
        for i, iss in enumerate(issues, 1):
            lines.append(f"{i}. {iss}")
    else:
        lines.append("✅ 未发现明显异常。")

    report = "\n".join(lines)

    # 保存报告
    safe_name = os.path.splitext(os.path.basename(state["pdf_path"]))[0]
    report_path = os.path.join(REPORTS_DIR, f"contract_review_{safe_name}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return {"report": report_path}


# ---------------------------------------------------------------------------
# 构建 LangGraph
# ---------------------------------------------------------------------------
def build_review_graph():
    g = StateGraph(ContractReviewState)

    g.add_node("parse_document", parse_document)
    g.add_node("extract_entities", extract_entities)
    g.add_node("compliance_check", compliance_check)
    g.add_node("amount_check", amount_check)
    g.add_node("seal_check", seal_check)
    g.add_node("generate_report", generate_report)

    g.add_edge(START, "parse_document")
    g.add_edge("parse_document", "extract_entities")
    g.add_edge("extract_entities", "compliance_check")

    # 金额核查与印章检测可在实体抽取后并行
    g.add_edge("extract_entities", "amount_check")
    g.add_edge("extract_entities", "seal_check")
    g.add_edge("compliance_check", "generate_report")
    g.add_edge("amount_check", "generate_report")
    g.add_edge("seal_check", "generate_report")
    g.add_edge("generate_report", END)

    return g.compile()


review_agent = build_review_graph()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def run_contract_review(pdf_path: str, file_id: str = "") -> dict:
    """运行完整合同审查流程，返回最终 state（含报告路径）。"""
    print(f"开始审查合同: {pdf_path}")
    start = time.time()

    result = review_agent.invoke({
        "pdf_path": pdf_path,
        "file_id": file_id,
    })

    print(f"审查完成，耗时 {time.time() - start:.1f}s")
    print(f"报告已生成: {result.get('report')}")
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python agents/contract_review_agent.py <合同PDF路径> [文件ID]")
        sys.exit(1)

    pdf = sys.argv[1]
    fid = sys.argv[2] if len(sys.argv) > 2 else ""
    run_contract_review(pdf, fid)
