"""
45 - 第09章开篇：上下文与记忆（checkpointer + thread_id 入门）

本章地图：
  · 上下文（context）= 每次调模型时"发出去"的内容（消息历史+系统提示+工具定义）
  · 记忆（memory）   = 跨调用保留的内容（state 里的消息历史）
  · 短期记忆：checkpointer 把 state 存下来，靠 thread_id 找回 → 多轮对话的"记性"
  · 长期记忆：store（跨 thread 共享），本章后面讲

本 demo 三步实证：
  ① 没接 checkpointer：两次 invoke 各自独立 → agent 不记得名字（金鱼脑）
  ② 接 InMemorySaver + 同一 thread_id：第二次 invoke 记得名字
  ③ 同一 checkpointer + 不同 thread_id：互不串台（线程隔离）
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


def last_ai(result):
    for m in reversed(result["messages"]):
        if getattr(m, "type", None) == "ai" and m.content:
            return m.content
    return ""


if __name__ == "__main__":
    # ===== ① 没接 checkpointer：金鱼脑 =====
    print("=" * 60)
    print("① 无 checkpointer：两次 invoke 互相不认识")
    print("=" * 60)
    bare = create_agent(model=deepseek, tools=[])
    r = bare.invoke({"messages": [HumanMessage("你好，我叫小明，请记住我的名字")]})
    print(f"第1次（{len(r['messages'])} 条消息）：", last_ai(r))
    r = bare.invoke({"messages": [HumanMessage("我叫什么名字？")]})
    print(f"第2次（{len(r['messages'])} 条消息）：", last_ai(r))
    print("→ 只有 1 条新消息进去，历史没带上，必然答不出。")

    # ===== ② 接 checkpointer + thread_id：有记性 =====
    print("\n" + "=" * 60)
    print("② InMemorySaver + thread_id='xiaoming'：记得住了")
    print("=" * 60)
    memory = InMemorySaver()
    agent = create_agent(model=deepseek, tools=[], checkpointer=memory)
    cfg1 = {"configurable": {"thread_id": "xiaoming"}}
    r = agent.invoke({"messages": [HumanMessage("你好，我叫小明，请记住我的名字")]}, cfg1)
    print(f"第1次（{len(r['messages'])} 条消息）：", last_ai(r))
    r = agent.invoke({"messages": [HumanMessage("我叫什么名字？")]}, cfg1)
    print(f"第2次（{len(r['messages'])} 条消息）：", last_ai(r))
    print("→ checkpointer 把 state（含消息历史）存了下来，第2次 invoke 自动续上。")

    # ===== ③ 换 thread_id：线程隔离 =====
    print("\n" + "=" * 60)
    print("③ 同一 checkpointer 换 thread_id='hacker'：互不串台")
    print("=" * 60)
    cfg2 = {"configurable": {"thread_id": "hacker"}}
    r = agent.invoke({"messages": [HumanMessage("我叫什么名字？")]}, cfg2)
    print(f"新线程（{len(r['messages'])} 条消息）：", last_ai(r))
    print("→ 新线程是空 state，只有 1 条消息，查不到'小明'。")
