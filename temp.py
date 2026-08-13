import os
import re
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import requests
from typing import Optional

load_dotenv()


# ==================== 工具定义（Open-Meteo 实现） ====================

# Open-Meteo 地理编码：城市名 → 经纬度
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Open-Meteo 天气 API：经纬度 → 天气数据
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"


@tool
def weather_tool(city: str) -> str:
    """查询指定城市的实时天气信息。

    参数:
        city: 城市名称，支持中文或英文，例如 "旧金山"、"北京"、"Tokyo"、"London"
    """
    try:
        # 1. 城市名 → 经纬度（Open-Meteo Geocoding API）
        geo_params = {
            "name": city,
            "count": 1,
            "language": "zh",  # 支持中文城市名
            "format": "json",
        }
        geo_resp = requests.get(GEOCODING_API_URL, params=geo_params, timeout=15)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        results = geo_data.get("results") or []
        if not results:
            return f"未找到城市 '{city}'，请尝试更具体的名称（如 '北京, 中国'）或使用英文名。"

        loc = results[0]
        lat = loc["latitude"]
        lon = loc["longitude"]
        city_name = loc.get("name", city)

        # 2. 经纬度 → 当前天气（Open-Meteo Weather API）
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "timezone": "auto",
        }
        weather_resp = requests.get(WEATHER_API_URL, params=weather_params, timeout=15)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()

        current = weather_data.get("current_weather")
        if not current:
            return f"未获取到 '{city_name}' 的当前天气数据，API 返回异常。"

        temp = current.get("temperature")
        windspeed = current.get("windspeed")
        winddirection = current.get("winddirection")
        weather_code = current.get("weathercode")
        time = current.get("time")

        # 简单的天气代码 → 中文描述（可按需扩展）
        weather_desc_map = {
            0: "晴",
            1: "基本晴",
            2: "局部多云",
            3: "阴",
            45: "雾",
            48: "沉积雾",
            51: "轻毛毛雨",
            53: "中毛毛雨",
            55: "浓毛毛雨",
            61: "小雨",
            63: "中雨",
            65: "大雨",
            71: "小雪",
            73: "中雪",
            75: "大雪",
            80: "小阵雨",
            81: "中阵雨",
            82: "大阵雨",
            95: "雷暴",
            96: "雷暴伴小冰雹",
            99: "雷暴伴大冰雹",
        }
        weather_desc = weather_desc_map.get(weather_code, f"未知代码({weather_code})")

        # 风向角度 → 中文方位
        def wind_dir_to_text(degrees):
            if degrees is None:
                return "未知"
            dirs = [
                "北风", "东北风", "东风", "东南风",
                "南风", "西南风", "西风", "西北风",
            ]
            idx = round(degrees / 45) % 8
            return dirs[idx]

        wind_dir = wind_dir_to_text(winddirection)

        return (
            f"{city_name}（{lat},{lon}）当前天气："
            f"{weather_desc}，气温 {temp}°C，"
            f"风速 {windspeed} km/h，风向 {wind_dir}，"
            f"观测时间 {time}。"
        )

    except requests.RequestException as e:
        return f"网络或 API 请求出错：{str(e)}"
    except Exception as e:
        return f"天气查询内部错误：{str(e)}"


# ==================== 工具调用解析 ====================

def extract_tool_call(text):
    """从模型输出文本中提取工具调用 JSON"""
    parts = re.split(r'</think\s*>', text)
    search_text = parts[-1] if len(parts) > 1 else text

    for match in re.finditer(r'\{', search_text):
        start = match.start()
        depth = 0
        for end in range(start, len(search_text)):
            if search_text[end] == '{':
                depth += 1
            elif search_text[end] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(search_text[start:end + 1])
                        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break
    return None


def clean_think_tags(text):
    """移除 <think > 推理标签，只保留实际回复内容"""
    return re.sub(r'<think\s*>.*?</think\s*>', '', text, flags=re.DOTALL).strip()


# ==================== 对话记忆 ====================

class AgentMemory:
    """对话记忆管理器。

    维护一个有界的历史对话列表，当超出最大长度时自动截断最早的对话，
    保留最近 N 轮用户-助手交互，确保上下文窗口不会无限增长。
    """

    def __init__(self, max_history: int = 10):
        """
        参数:
            max_history: 保留的最大对话轮数（一轮 = 一条用户消息 + 一条助手回复）
        """
        self.max_history = max_history
        # 存储 (HumanMessage, AIMessage) 元组列表
        self._history: list[tuple[HumanMessage, AIMessage]] = []

    def add_exchange(self, user_msg: HumanMessage, ai_msg: AIMessage) -> None:
        """添加一轮对话（用户提问 + 助手回复）到记忆中。"""
        self._history.append((user_msg, ai_msg))
        # 超出上限时丢弃最早的对话
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def get_context_messages(self) -> list:
        """获取历史对话上下文（不含 SystemMessage），按时间顺序排列。"""
        messages = []
        for user_msg, ai_msg in self._history:
            messages.append(user_msg)
            messages.append(ai_msg)
        return messages

    def clear(self) -> None:
        """清空所有历史记忆。"""
        self._history.clear()

    @property
    def turn_count(self) -> int:
        """当前记忆中的对话轮数。"""
        return len(self._history)

    def __repr__(self) -> str:
        return f"AgentMemory(turns={self.turn_count}/{self.max_history})"


# ==================== 手动 ReAct 循环 ====================

TOOL_MAP = {
    "weather_tool": weather_tool,
}

SYSTEM_PROMPT = """你是一个有用的天气查询助手，可以使用工具来获取天气信息。

你可以使用以下工具：
- weather_tool: 查询指定城市的实时天气信息，参数: {"city": "城市名称"}

当你需要调用工具时，请严格使用以下 JSON 格式输出（不要加代码 代码块）：
{"name": "工具名称", "arguments": {"参数名": "参数值"}}

当你已经获得足够信息可以回答用户问题时，直接用自然语言回复，不再输出 JSON。

注意：你具有对话记忆能力，可以记住用户之前问过的问题和你的回复。
如果用户的提问涉及之前的对话内容（如"那个城市"、"再查一下"、"和刚才相比"等），
请结合历史上下文来理解和回答。"""


def run_agent(
    query: str,
    memory: Optional[AgentMemory] = None,
    max_iterations: int = 5,
) -> str:
    """手动驱动 ReAct 循环。

    参数:
        query:          用户输入的问题
        memory:         对话记忆实例，传入后会自动加载历史上下文并保存本轮对话；
                        传 None 则无记忆（单次对话模式）
        max_iterations: 单轮推理最大工具调用迭代次数
    """
    llm = ChatOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("MODEL_NAME"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.0)),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", 1024)),
        top_p=float(os.getenv("OPENAI_TOP_P", 1.0)),
        streaming=os.getenv("OPENAI_STREAM") == "true"
    )

    # 构建消息列表：SystemMessage + 历史上下文 + 当前用户输入
    user_msg = HumanMessage(content=query)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if memory is not None:
        messages.extend(memory.get_context_messages())

    messages.append(user_msg)

    final_answer = ""

    for i in range(max_iterations):
        print(f"\n--- 第 {i + 1} 轮 ---")

        # 1. 调用模型
        response = llm.invoke(messages)
        ai_content = response.content
        messages.append(response)

        # 2. 尝试从输出中提取工具调用
        tool_call = extract_tool_call(ai_content)

        if tool_call is None:
            # 没有工具调用 → 模型已经给出最终回复
            final_answer = clean_think_tags(ai_content)

            # 将本轮对话保存到记忆
            if memory is not None:
                ai_msg = AIMessage(content=final_answer)
                memory.add_exchange(user_msg, ai_msg)

            return final_answer

        # 3. 执行工具
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        print(f"  🔧 调用工具: {tool_name}({tool_args})")

        tool_func = TOOL_MAP.get(tool_name)
        if tool_func is None:
            tool_result = f"错误：未知工具 '{tool_name}'"
        else:
            try:
                tool_result = tool_func.invoke(tool_args)
            except Exception as e:
                tool_result = f"工具执行出错：{str(e)}"

        print(f"  📋 工具结果: {tool_result[:100]}...")

        # 4. 将工具结果加入消息历史，让模型基于结果生成最终回复
        messages.append(ToolMessage(content=tool_result, tool_call_id=f"call_{i}"))

    return "⚠️ 已达到最大迭代次数，未能获得最终回复。"


# ==================== 运行 ====================

if __name__ == "__main__":
    # 初始化对话记忆（保留最近 10 轮对话）
    memory = AgentMemory(max_history=10)

    print("=" * 50)
    print("🌤  天气查询助手（支持多轮对话记忆）")
    print("   输入 'quit' 或 'exit' 退出")
    print("   输入 'clear' 清空对话记忆")
    print("   输入 'history' 查看当前记忆轮数")
    print("=" * 50)

    while True:
        try:
            query = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("再见！")
            break
        if query.lower() == "clear":
            memory.clear()
            print("🗑  对话记忆已清空。")
            continue
        if query.lower() == "history":
            print(f"📝 当前记忆轮数: {memory.turn_count}/{memory.max_history}")
            continue

        result = run_agent(query, memory=memory)
        print(f"\n🤖 助手: {result}")
