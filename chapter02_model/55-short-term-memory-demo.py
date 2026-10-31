# -*- coding: utf-8 -*-
"""55 · 短期记忆三脚凳实验：State + Checkpointer + Thread ID 缺一不可

课程「短期记忆」公式：State（会话内部状态）+ Checkpointer（持久化机制）+ Thread ID（会话作用域）
本 demo 每次拆掉一条腿，看短期记忆怎么塌：
  实验 0：三条腿齐全      → 记得住（2→4 条）
  实验 1：拆 Checkpointer → 金鱼脑（2→2 条）
  实验 2：拆 Thread ID    → 直接报错（Checkpointer requires thread_id）
  实验 3：只给 Thread ID（无 Checkpointer）→ 无处可存，照样金鱼脑
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url="https://api.deepseek.com/v1", temperature=0)
SYS = "你是助手。"
CFG_T1 = {"configurable": {"thread_id": "t1"}}


def run(agent, name, cfg):
    """同一个 agent 连问两轮：第 1 轮告知名字，第 2 轮问名字，看记不记得住。"""
    r1 = agent.invoke({"messages": [{"role": "user", "content": "我叫小测。"}]}, config=cfg)
    r2 = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]}, config=cfg)
    print(f"【{name}】 第1轮消息 {len(r1['messages'])} 条 → 第2轮消息 {len(r2['messages'])} 条")
    print(f"   第2轮回答：{r2['messages'][-1].content[:40]}")
    print()


print("=" * 64)
print("实验 0：三条腿齐全（state 内置 + checkpointer + thread_id）")
print("=" * 64)
agent0 = create_agent(model=llm, tools=[], system_prompt=SYS, checkpointer=InMemorySaver())
run(agent0, "齐全", CFG_T1)

print("=" * 64)
print("实验 1：拆掉 Checkpointer（有 state 有 thread_id 但没人存档）")
print("=" * 64)
agent1 = create_agent(model=llm, tools=[], system_prompt=SYS)  # 无 checkpointer
run(agent1, "无 Checkpointer", CFG_T1)

print("=" * 64)
print("实验 2：拆掉 Thread ID（有 checkpointer 但不给线程身份证）")
print("=" * 64)
agent2 = create_agent(model=llm, tools=[], system_prompt=SYS, checkpointer=InMemorySaver())
try:
    r = agent2.invoke({"messages": [{"role": "user", "content": "我叫小测。"}]},
                      config={"configurable": {}})
    print("居然跑通了？", len(r["messages"]))
except ValueError as e:
    print(f"直接报错 → {e}")
print()

print("=" * 64)
print("实验 3：只给 Thread ID，但 agent 没接 checkpointer（身份证没处刷）")
print("=" * 64)
agent3 = create_agent(model=llm, tools=[], system_prompt=SYS)  # 又是无 checkpointer
run(agent3, "thread_id 无 checkpointer", CFG_T1)

print("结论：短期记忆 = State + Checkpointer + Thread ID，三条腿缺一，凳子就塌。")
