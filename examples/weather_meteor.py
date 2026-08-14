import os
import re
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import requests
from urllib.parse import quote
from tavily import TavilyClient

load_dotenv()

# ================= 1. 定义工具的参数 Schema =================

class WeatherInput(BaseModel):
    city: str = Field(description="要查询天气的城市名称，例如：北京")

class WebSearchInput(BaseModel):
    query: str = Field(description="用于网络搜索的关键词，例如：'北京 暴雨 预警'")

# ================= 2. 实现具体的工具函数 =================

def weather_tool(city: str) -> str:
    """获取指定城市的实时天气信息"""
    try:
        url = f"https://wttr.in/{quote(city)}?format=j1"
        headers = {"Accept-Language": "zh-CN"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current_condition = data["current_condition"][0]
        temp_C = current_condition["temp_C"]
        
        desc_list = current_condition.get("lang_zh", [])
        weather_desc = desc_list[0]["value"] if desc_list and desc_list[0].get("value") else current_condition["weatherDesc"][0]["value"]
            
        humidity = current_condition["humidity"]
        windspeed = current_condition["windspeedKmph"]
        
        return f"【实时天气】{city}当前：{weather_desc}，温度 {temp_C}°C，湿度 {humidity}%，风速 {windspeed} km/h。"
        
    except Exception as e:
        return f"获取天气失败：{str(e)}"

def web_search_tool(query: str) -> str:
    """搜索近期的网络新闻或相关信息"""
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=3,
            include_answer=True
        )

        answer = response.get("answer", "")
        results = response.get("results", [])

        if not results:
            return f"未找到关于 '{query}' 的相关新闻或信息。"

        output = []
        if answer:
            output.append(f"【摘要】{answer}\n")

        for res in results:
            title = res.get("title", "无标题")
            content = res.get("content", "无摘要")
            url = res.get("url", "")
            score = res.get("score", 0)

            output.append(f"- **{title}**\n  摘要: {content}\n  来源: {url}")

        return "【搜索结果】\n" + "\n\n".join(output)
    except Exception as e:
        return f"网络搜索时发生错误: {str(e)}"

# ================= 3. 辅助函数 =================

def clean_response(text):
    """清理模型输出中的思考标签，只保留最终回答"""
    # 兼容清理 <think> 或  标签
    cleaned = re.sub(
        r'<think[^>]*>.*?</think\s*>|<thinking[^>]*>.*?</thinking\s*>',
        '', text, flags=re.DOTALL
    )
    return cleaned.strip()

# ================= 4. 初始化 LLM =================

llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("MODEL_NAME"), 
    temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.0)),
    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", 1024))
)

# ================= 5. 配置系统提示词 (防吞标签 + 强制单步) =================

SYSTEM_PROMPT = """你是一个有用的助手。你可以使用工具来获取信息。

当前可用的工具：
1. weather_tool: 获取指定城市的实时天气情况。参数: city (字符串)。
2. web_search_tool: 搜索近期的网络新闻或相关信息。参数: query (字符串)。

【极其重要的规则】
1. 每次回复只能调用【一个】工具。如果你需要多个信息，请先调用第一个工具，获取结果后再调用第二个。
2. 调用工具时，必须严格使用 [TOOL_CALL] 和 [/TOOL_CALL] 标签包裹 JSON 数据。不要使用尖括号 <>。
3. JSON 数据必须是标准的字典格式，包含 "name" 和 "arguments" 两个键。

格式示例：
<think>
我需要先查天气，再查新闻。这次先查天气。
</think>
[TOOL_CALL]
{"name": "weather_tool", "arguments": {"city": "北京"}}
[/TOOL_CALL]

当你获得了工具返回的信息后，请判断信息是否足够。如果不够，请继续调用下一个工具；如果足够，请直接回答用户的最终问题，无需再次输出 [TOOL_CALL]。
"""

# ================= 6. Agent 核心循环 =================

def run_manual_agent(user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    max_iterations = 5
    for i in range(max_iterations):
        print(f"\n--- 思考与执行迭代 {i+1} ---")
        response = llm.invoke(messages)
        content = response.content
        print(f"模型输出:\n{content}\n")

        # 1. 优先尝试匹配 [TOOL_CALL] 标签
        match = re.search(r'\[TOOL_CALL\](.*?)\[/TOOL_CALL\]', content, re.DOTALL)
        json_str = None
        
        if match:
            json_str = match.group(1).strip()
        else:
            # 2. 后备方案：如果模型忘记加标签，直接输出了裸 JSON，尝试强行提取
            # 匹配包含 "name" 和 "arguments" 的字典结构
            json_match = re.search(r'(\{\s*"name"\s*:\s*".*?"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                print("⚠️ 检测到模型未使用标签，已启用后备 JSON 提取机制。")

        if not json_str:
            print("✅ 未检测到工具调用，视为最终回答。")
            return content

        try:
            # 清理可能存在的代码块标记
            if json_str.startswith("```"):
                json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
                json_str = re.sub(r'\s*```$', '', json_str)

            tool_info = json.loads(json_str)
            tool_name = tool_info.get("name")
            tool_args = tool_info.get("arguments", {})

            print(f"⚙️ 准备执行工具: {tool_name}, 参数: {tool_args}")

            if tool_name == "weather_tool":
                tool_result = weather_tool(**tool_args)
            elif tool_name == "web_search_tool":
                tool_result = web_search_tool(**tool_args)
            else:
                tool_result = f"错误：未找到名为 {tool_name} 的工具。"

            print(f"📥 工具执行结果:\n{tool_result}\n")

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"工具 '{tool_name}' 执行结果如下：\n{tool_result}\n请根据此结果决定下一步操作或给出最终回答。"})

        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "你输出的工具调用 JSON 格式不正确，解析失败。请检查格式并重新输出正确的 [TOOL_CALL]。"})

    return "达到最大迭代次数，未能得出最终结论。"

# ================= 7. 运行测试 =================

if __name__ == "__main__":
    query = "新德里最近天气怎么样？有没有什么相关的极端天气新闻？"
    print(f"👤 用户提问: {query}")
    
    raw_result = run_manual_agent(query)
    final_result = clean_response(raw_result)
    
    print("\n" + "="*40)
    print("🤖 最终结果 (已过滤思考过程)")
    print("="*40)
    print(final_result)