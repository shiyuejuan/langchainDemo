"""
19 - PIIMiddleware 演示（敏感信息脱敏中间件，LangChain 1.x + DeepSeek）

课程片段做了什么：给 agent 装两个 PIIMiddleware——
  · api_key：strategy="hash"，detector=正则 r"sk-[a-zA-Z0-9]+"   → key 被换成确定性哈希
  · phone_number：strategy="mask"，detector=自定义函数            → 手机号部分打码
实测要点：
  1. phone_number 不在内置类型（内置只有 email/credit_card/ip/mac_address/url），
     所以必须自定义 detector（正则串 或 返回 PIIMatch 列表的函数）。
  2. apply_to_input=True 时 before_model 会【永久修改 state】里最后一条用户消息
     （不只是发给模型时临时改），在 result["messages"] 里就能看到脱敏结果。
  3. 四种策略：block / redact(默认) / mask / hash。
"""
import os
import re
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.agents.middleware.pii import PIIMatch
from langchain.messages import HumanMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)
model = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

# ===== 自定义手机号 detector：返回 PIIMatch 列表 =====
# PIIMatch = TypedDict：{"type": str, "value": str, "start": int, "end": int}
def detect_phone_number(text: str) -> list[PIIMatch]:
    return [
        {"type": "phone_number", "value": m.group(), "start": m.start(), "end": m.end()}
        for m in re.finditer(r"1[3-9]\d{9}", text)
    ]

# ===== 和课程一致的中间件组合（另加一个内置 email redact 做对比）=====
agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        PIIMiddleware("api_key", strategy="hash", apply_to_input=True,
                      detector=r"sk-[a-zA-Z0-9]+"),
        PIIMiddleware("phone_number", strategy="mask", apply_to_input=True,
                      detector=detect_phone_number),
        PIIMiddleware("email", strategy="redact", apply_to_input=True),  # 内置类型，默认策略演示
    ],
)

test_text = ("我的 API key 是 sk-abc123XYZ456defGHI789jklMNO012，"
             "手机号 13812345678，邮箱 zhangsan@example.com。"
             "请把这三样复述一遍，确认你看到了什么。")

result = agent.invoke({"messages": [HumanMessage(test_text)]})

print("=" * 64)
print("state 里实际存下来的用户消息（脱敏后）：")
print("  原文：", test_text)
print("  实际：", result["messages"][0].content)

print("\n" + "=" * 64)
print("模型看到的并回答的：")
print(result["messages"][-1].content)

# ===== 附加：block 策略——检测到就直接抛异常，请求都不会发给模型 =====
print("\n" + "=" * 64)
agent_block = create_agent(
    model=model,
    tools=[],
    middleware=[PIIMiddleware("api_key", strategy="block", detector=r"sk-[a-zA-Z0-9]+")],
)
try:
    agent_block.invoke({"messages": [HumanMessage("帮我检查这个 key：sk-SECRET123456")]})
    print("（没有抛异常？说明没拦住）")
except Exception as e:
    print("block 策略触发异常：", type(e).__name__, "-", str(e)[:120])
