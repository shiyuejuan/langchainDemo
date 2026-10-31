"""
06-model-bind-tools.py
工具绑定（bind_tools）：让大模型「会调用函数」。

对应课程里 CLOSEAI + gpt-5.4-mini 的写法，这里改用本项目已验证可用的 DeepSeek：
- .env 里没有 CLOSEAI_API_KEY / CLOSEAI_BASE_URL，直接照抄会报 Missing credentials
- deepseek-chat 支持 function calling，bind_tools 写法完全一致，只换模型和密钥

完整闭环：
用户提问 → 模型返回 tool_calls（并不执行）→ 我们执行函数
→ 用 ToolMessage 把结果回传 → 模型生成最终回答
"""
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage

# 加载配置文件
load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

model = init_chat_model(
    model="openai:deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=30,
    max_retries=1,
)


def get_weather(city: str) -> str:
    """查询指定城市当前的天气"""
    return "不错哦~"


# 模拟模型绑定工具：返回一个「知道有哪些工具可调」的新模型对象
model_with_tools = model.bind_tools([get_weather])

query = "北京今天天气怎么样？"
messages = [HumanMessage(content=query)]

# 第 1 次调用：模型大概率不直接回答，而是返回 tool_calls
ai_msg = model_with_tools.invoke(messages)
print(">>> 第一次调用，模型想调用的工具：")
print("    content:", ai_msg.content or "(空)")
print("    tool_calls:", ai_msg.tool_calls)

# 模型真的想调工具时，我们替它执行，并把结果回传
if ai_msg.tool_calls:
    messages.append(ai_msg)
    for tc in ai_msg.tool_calls:
        result = get_weather(**tc["args"])  # 真正执行函数的是我们，不是模型
        print(f">>> 执行 {tc['name']}({tc['args']}) -> {result!r}")
        messages.append(
            ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"])
        )

    # 第 2 次调用：模型拿到工具结果，生成面向用户的最终回答
    final = model_with_tools.invoke(messages)
    print(">>> 最终回答：", final.content)
else:
    print(">>> 模型没有请求调用工具，直接回答了。")
