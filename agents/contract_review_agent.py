"""
合同类文件审查自动化 Agent
============================
基于 agent-helper 服务（FastAPI, 默认端口 8018）提供的文档解析、LLM 抽取、
合规审查、印章检测等能力，编排一个端到端的合同审查流水线。

架构说明:
    - agent-helper 部署在服务器(172.25.67.120)，MinerU 解析、LLM 调用、印章检测
      都在服务器上执行，因此 file_path / save_path 必须使用服务器可访问的路径。
    - 本 Agent 在本地/任意位置运行，通过 HTTP 编排调用各 API。

流程:
    1. parse_document   - 调用 MinerU 提取合同文本 (输出到 PDF 同目录 mineru_result/)
    2. extract_entities - 抽取合同关键要素（传 device_text 服务器路径）
    3. compliance_check - 要素一致性核查（complianceAuditDir，传路径逐行审查）
    4. amount_check     - 合同金额与付款方式核查（内部自动调 MinerU）
    5. seal_check       - 盖章页签名 / 印章检测
    6. generate_report  - 汇总全部结果，生成 Markdown 审查报告

用法:
    python agents/contract_review_agent.py <合同PDF路径(服务器路径)> [文件ID]
"""

import os
import json
import re
import time
import datetime
from typing import TypedDict, Any

import httpx
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
AGENT_HELPER_BASE_URL = os.getenv(
    "AGENT_HELPER_BASE_URL", "http://172.25.67.120:8018"
)

# 报告输出目录（本地项目根目录下 reports/）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# HTTP 超时：256MB 大文件 MinerU 解析可能超过 30 分钟
TIMEOUT = 3600.0

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

# 一致性核查规则
CONTRACT_COMPLIANCE_RULES = [
    "核查合同总金额、大小写金额是否一致，若不一致需指出",
    "核查合同签订日期是否在有效期范围内，若超出需指出",
    "核查甲方、乙方名称是否完整且正确，与营业执照名称是否一致",
]


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------
class ContractReviewState(TypedDict, total=False):
    pdf_path: str                    # 合同 PDF 路径（服务器路径）
    file_id: str                     # 文件 ID（向量库关联）
    parse_result: Any                # MinerU 解析结果
    device_text_path: str            # 服务器上的 device_text 文件路径
    text_path: str                   # 服务器上的 txt 文件路径
    milvus_uuid: str                 # 向量库 collection 名
    entities: dict                   # 抽取的合同要素
    compliance_results: list         # 一致性核查结果
    amount_result: dict              # 金额核查结果
    seal_result: dict                # 印章/签名检测结果
    report: str                      # 最终报告（本地路径）
    errors: list                     # 错误信息收集


# ---------------------------------------------------------------------------
# 工具函数：调用 agent-helper API
# ---------------------------------------------------------------------------
def _post(path: str, json_data: dict) -> Any:
    """POST 到 agent-helper 服务并返回 data 字段。"""
    url = f"{AGENT_HELPER_BASE_URL}{path}"
    resp = httpx.post(url, json=json_data, timeout=TIMEOUT)
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 200:
        raise RuntimeError(f"API {path} 返回错误: {body.get('msg')}")
    return body.get("data")


def _safe_str(v: Any) -> str:
    """将值转为安全的字符串表示。"""
    if v is None:
        return "（未提取到）"
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def _sanitize_file_id(file_id: str) -> str:
    """清洗 file_id，使其符合 Milvus collection 命名规范（仅数字、字母、下划线）。

    agent-helper 的 MinerU 服务用 "doc_" + file_id 作为 Milvus collection 名，
    若 file_id 含连字符（如 UUID 格式）会导致 Milvus 报
    "Invalid collection name" 错误。这里统一去掉非法字符。
    """
    return re.sub(r"[^0-9a-zA-Z_]", "", file_id)


# ---------------------------------------------------------------------------
# 流程节点
# ---------------------------------------------------------------------------
def parse_document(state: ContractReviewState) -> dict:
    """步骤 1: 调用 MinerU 提取合同文本（在服务器上执行，输出到 PDF 同目录）。

    健壮性处理:
        textExtractDir 是 "解析 + 向量化 + Milvus 入库" 捆绑流程。若 embedding 服务
        异常，入库会失败并整体报错，但 `_device.txt` / `_middle_original.json` 等
        解析产物已写入磁盘。此时无法从接口返回值拿路径，故根据产物命名规则
        (PDF同目录/mineru_result/{basename}_device.txt) 推断路径继续后续审查。
    """
    pdf_path = state["pdf_path"]
    file_id = _sanitize_file_id(state.get("file_id", str(int(time.time()))))

    # save_path 必须是服务器路径：PDF 同目录下的 mineru_result/
    pdf_dir = os.path.dirname(pdf_path)
    save_path = os.path.join(pdf_dir, "mineru_result")
    basename = os.path.splitext(os.path.basename(pdf_path))[0]

    result = None
    device_text_path = ""
    text_path = ""
    milvus_uuid = ""
    try:
        result = _post("/api/mineru/textExtractDir", {
            "file_list": [{
                "file_path": pdf_path,
                "save_path": save_path,
                "file_id": file_id,
            }]
        })

        # 解析返回：files_info_list，每项含 device_text / text / middle_json / milvus_uuid
        if isinstance(result, list) and result:
            info = result[0]
            device_text_path = info.get("device_text", "") or ""
            text_path = info.get("text", "") or ""
            milvus_uuid = info.get("milvus_uuid", "") or ""
    except Exception as e:
        # 已知情形：embedding 服务异常导致入库失败，接口整体报错，但产物已落盘。
        # 根据产物命名规则推断路径（device_text 同时是 _device.txt 和 _device_original.txt）。
        error_msg = str(e)
        print(f"[warn] textExtractDir 调用失败（可能为 embedding 入库问题）: {error_msg}")
        device_text_path = os.path.join(save_path, f"{basename}_device.txt")
        text_path = os.path.join(save_path, f"{basename}_device.txt")

    return {
        "parse_result": result,
        "device_text_path": device_text_path,
        "text_path": text_path,
        "milvus_uuid": milvus_uuid,
        "file_id": file_id,
    }


def extract_entities(state: ContractReviewState) -> dict:
    """步骤 2: 抽取合同关键要素。直接传 device_text 服务器路径，API 内部读取。"""
    # 优先用 device_text 路径；若为空则退回 text 路径
    content_path = state.get("device_text_path") or state.get("text_path")
    if not content_path:
        return {"entities": {}, "errors": state.get("errors", []) + ["MinerU 未生成文本文件"]}

    file_list = [{
        "file_name": os.path.basename(state["pdf_path"]),
        "text": content_path,   # 服务器路径，API 内部 os.path.isfile 判断后读取
        "entity_types": CONTRACT_ENTITY_TYPES,
        "task_constraint": "从合同文本中抽取上述要素，缺失字段为 null",
        "milvus_uuid": state.get("milvus_uuid", ""),
    }]

    result = _post("/api/agent/extractEntity", {"file_list": file_list})

    entities = {}
    if isinstance(result, list) and result:
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
    """步骤 3: 要素一致性核查。使用 complianceAuditDir，传 device_text 路径逐行审查。

    注意: device_text_path 是服务器路径，本地 os.path.exists 恒为 False，
    因此这里不再做本地存在性判断，直接按服务器路径处理。
    """
    device_text_path = state.get("device_text_path")
    if device_text_path:
        review_list = [
            {"object": device_text_path, "rule": rule}
            for rule in CONTRACT_COMPLIANCE_RULES
        ]
        result = _post("/api/agent/complianceAuditDir", {"review_list": review_list})
        compliance_results = result if isinstance(result, list) else [result]
        return {"compliance_results": compliance_results}

    # 兜底：无 device_text 路径时，退回用抽取的要素文本审查
    object_text = json.dumps(state.get("entities", {}), ensure_ascii=False, indent=2)
    review_list = [
        {"object": {"text": object_text}, "rule": rule,
         "knowledge_base_ids": [], "preprocess": True}
        for rule in CONTRACT_COMPLIANCE_RULES
    ]
    result = _post("/api/agent/complianceAudit", {"review_list": review_list})
    compliance_results = result if isinstance(result, list) else [result]
    return {"compliance_results": compliance_results}


def amount_check(state: ContractReviewState) -> dict:
    """步骤 4: 合同金额与付款方式核查（服务器内部自动调 MinerU，超时 3600s）。

    注意: verify_contract_amount 内部会先检查 mineru_result 产物是否存在，
    若存在则直接复用，不会重复解析。
    """
    result = _post("/api/Contract/verifyContractAmount", {
        "pdf_path": state["pdf_path"],
        "file_id": _sanitize_file_id(state.get("file_id", "")),
    })
    return {"amount_result": result if isinstance(result, dict) else {"raw": result}}


def _post_raw(path: str, body: Any) -> Any:
    """POST，body 直接作为裸 JSON 请求体（非 {字段: 值} 包装）。

    适用: pdf_path 定义为 `str = Body(...)` 的接口，FastAPI 期望请求体为裸字符串。
    """
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


def seal_check(state: ContractReviewState) -> dict:
    """步骤 5: 盖章页签名与印章检测。

    注意: 这两个接口的 pdf_path 为 `str = Body(...)`，请求体需为**裸字符串**。
    """
    pdf_path = state["pdf_path"]
    seal_result = {}
    try:
        seal_result["signature_with_seal"] = _post_raw(
            "/api/Contract/checkSignaturewithSeal", pdf_path
        )
    except Exception as e:
        seal_result["signature_with_seal"] = {"error": str(e)}

    try:
        seal_result["seal_detection"] = _post_raw(
            "/api/detection/detectSeal", pdf_path
        )
    except Exception as e:
        seal_result["seal_detection"] = {"error": str(e)}

    return {"seal_result": seal_result}


def generate_report(state: ContractReviewState) -> dict:
    """步骤 6: 汇总生成 Markdown 审查报告（保存到本地 reports/）。"""
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
            lines.append(f"- **{k}**: {_safe_str(v)}")
    else:
        lines.append("- 未能抽取到合同要素")
    lines.append("")

    # 2. 金额核查
    lines.append("## 二、合同金额核查")
    amount = state.get("amount_result", {})
    if isinstance(amount, dict):
        lines.append(f"- 合同金额: {_safe_str(amount.get('contract_amount'))}")
        lines.append(f"- 付款方式: {_safe_str(amount.get('payment_method'))}")
        lines.append(f"- 分期合计: {_safe_str(amount.get('installment_total'))}")
        matched = amount.get("amount_matched")
        if matched is True:
            lines.append("- **金额一致性: ✅ 通过**")
        elif matched is False:
            lines.append("- **金额一致性: ❌ 不通过**")
        else:
            lines.append("- 金额一致性: ⚠️ 无法判定")
        if amount.get("message"):
            lines.append(f"- 说明: {amount['message']}")
        installments = amount.get("installment_items", [])
        if installments:
            lines.append("- 分期明细:")
            for item in installments:
                if isinstance(item, dict):
                    lines.append(f"  - {item.get('phase', '')}: "
                                 f"{item.get('amount', '')} "
                                 f"({item.get('ratio', '')})")
    lines.append("")

    # 3. 一致性核查（去重：complianceAuditDir 会按文本块返回多条，只保留唯一结论）
    lines.append("## 三、要素一致性核查")
    compliance_items = state.get("compliance_results", [])
    # 展平嵌套列表，并提取 (rule, result, risk_content, advise)
    flat_items = []
    seen = set()
    for item in compliance_items:
        if isinstance(item, list):
            for sub in item:
                flat_items.append(sub)
        else:
            flat_items.append(item)

    deduped = []
    for item in flat_items:
        if not isinstance(item, dict):
            continue
        # 归一化：提取核心字段
        rule = item.get("rule", "") or ""
        result = str(item.get("result", "")).strip()
        risk = (item.get("risk_content") or "").strip()
        advise = (item.get("advise") or "").strip()
        # 过滤异常/空条目：rule 为空、result 为"格式错误"等无意义结果
        if not rule.strip():
            continue
        if result.lower() == "true" and not risk:
            continue
        if result in ("格式错误", "false", "true") and not risk and not advise:
            continue
        key = (rule, result, risk)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"rule": rule, "result": result, "risk": risk, "advise": advise})

    if deduped:
        for i, item in enumerate(deduped, 1):
            status = "✅ 通过" if item["result"].lower() == "true" else "⚠️ 需关注"
            lines.append(f"{i}. **{item['rule']}** — {status}")
            if item["risk"]:
                lines.append(f"   - 风险: {item['risk']}")
            if item["advise"]:
                lines.append(f"   - 建议: {item['advise']}")
    else:
        lines.append("- 未发现规则违反。")
    lines.append("")

    # 4. 印章检测
    lines.append("## 四、盖章与签名检测")
    seal = state.get("seal_result", {})
    for k, v in seal.items():
        if isinstance(v, dict) and "error" in v:
            lines.append(f"- {k}: ⚠️ 检测失败 ({v['error']})")
        elif isinstance(v, dict):
            if k == "signature_with_seal":
                ok = v.get("sign_with_seal")
                lines.append(f"- 签章同页: {'✅ 是' if ok else '❌ 否'}")
                pages = v.get("matched_pages", [])
                for p in pages:
                    if isinstance(p, dict):
                        lines.append(f"  - 第 {p.get('page_number')} 页: "
                                     f"印章 {p.get('seal_counts')} 个 / "
                                     f"签名 {p.get('sign_counts')} 个 / "
                                     f"签章同页 {'有' if p.get('has_sign_with_seal') else '无'}")
            elif k == "seal_detection":
                complete = v.get("seal_complete")
                lines.append(f"- 印章完整: {'✅ 是' if complete else '❌ 否'}")
                lines.append(f"- 印章数量: {v.get('seal_counts')}")
                for bbox in v.get("seal_bbox", []):
                    if isinstance(bbox, dict):
                        lines.append(f"  - 类型 {bbox.get('type')} / "
                                     f"置信度 {round(bbox.get('confidence', 0), 3)}")
            else:
                lines.append(f"- {k}: {_safe_str(v)[:500]}")
        else:
            lines.append(f"- {k}: {_safe_str(v)[:500]}")
    lines.append("")

    # 5. 审查结论（汇总所有风险项）
    lines.append("## 五、审查结论")
    issues = list(state.get("errors", []))
    if isinstance(amount, dict) and amount.get("amount_matched") is False:
        issues.append("合同金额与分期付款合计不一致")
    # 一致性核查中的风险项
    for item in deduped:
        if item["result"].lower() == "false" or item["risk"]:
            issues.append(f"[{item['rule']}] {item['risk'][:200] if item['risk'] else '存在违规'}")
    lines.append("")
    if issues:
        lines.append("**发现以下问题需关注：**")
        for i, iss in enumerate(issues, 1):
            lines.append(f"{i}. {iss}")
    else:
        lines.append("✅ 未发现明显异常。")

    report = "\n".join(lines)

    # 保存报告（本地）
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
        print("用法: python agents/contract_review_agent.py <合同PDF路径(服务器路径)> [文件ID]")
        sys.exit(1)

    pdf = sys.argv[1]
    fid = sys.argv[2] if len(sys.argv) > 2 else ""
    run_contract_review(pdf, fid)
