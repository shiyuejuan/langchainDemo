"""
21 - ModelCallLimitMiddleware 演示（模型调用限流中间件，LangChain 1.x + DeepSeek）

课程片段：ModelCallLimitMiddleware(thread_limit=2(注释掉了), run_limit=3, exit_behavior="end")
         + checkpointer=InMemorySaver()（注释写明 Required for thread limiting）

两种限制口径（源码确认）：
  · run_limit   —— 单次 invoke 里最多几次模型调用（计数器 run_model_call_count，不持久化）
  · thread_limit —— 同一个 thread（整个对话历史）累计最多几次模型调用
                   （计数器 thread_model_call_count，要持久化 → 必须配 checkpointer）
两种退出行为：
  · exit_behavior="end"   —— 达到上限后直接跳到图终点（不报错，但也不会有最终回答）
  · exit_behavior="error" —— 达到上限后抛 ModelCallLimitExceededError
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.tools import tool
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)
model = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

@tool
def get_weather(city: str) -> str:
    """查询指定城市天气。"""
    return f"{city} 晴，25度"

def count_model_calls(messages):
    return sum(1 for m in messages if getattr(m, "tool_calls", None) or m.__class__.__name__ == "AIMessage")

# ===== 场景 1：run_limit=3 + exit_behavior="end" =====
agent_end = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[ModelCallLimitMiddleware(run_limit=3, exit_behavior="end")],
)
print("=" * 64)
print("场景1：run_limit=3, exit_behavior='end'（任务正常需要 5 次模型调用）")
result = agent_end.invoke({"messages": [HumanMessage(
    "依次查询北京、上海、广州、深圳四个城市的天气，然后总结成一句话。")]})
n = count_model_calls(result["messages"])
last = result["messages"][-1]
print(f"  实际模型调用次数：{n}（达到 run_limit 后被掐断）")
print(f"  最后一条消息类型：{last.__class__.__name__}")
print(f"  最后一条内容：{str(last.content)[:80] or '(空——end 行为直接跳终点，没有最终总结)'}")

# ===== 场景 2：run_limit=3 + exit_behavior="error" =====
agent_err = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[ModelCallLimitMiddleware(run_limit=3, exit_behavior="error")],
)
print("\n" + "=" * 64)
print("场景2：run_limit=3, exit_behavior='error'")
try:
    agent_err.invoke({"messages": [HumanMessage("依次查北京、上海、广州、深圳的天气并总结。")]})
    print("  （未抛异常？）")
except Exception as e:
    print(f"  抛出异常：{type(e).__name__}：{str(e)[:100]}")

# ===== 场景 3：thread_limit=2 + checkpointer（跨 invoke 累计）=====
print("\n" + "=" * 64)
print("场景3：thread_limit=2 + checkpointer（同一 thread 三次 invoke）")
agent_thread = create_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver(),   # thread 计数要持久化 → 必须 checkpointer
    middleware=[ModelCallLimitMiddleware(thread_limit=2, exit_behavior="end")],
)
config = {"configurable": {"thread_id": "t-1"}}
for i, q in enumerate(["你好", "今天星期几", "再聊两句"], 1):
    r = agent_thread.invoke({"messages": [HumanMessage(q)]}, config=config)
    answered = bool(r["messages"][-1].content)
    print(f"  第 {i} 次 invoke：有模型回答？{answered}（thread 累计上限 2，第 3 次应被直接跳终点）")
