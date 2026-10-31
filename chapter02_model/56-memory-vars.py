# -*- coding: utf-8 -*-
"""56 · 短期记忆：用代码打开"抽屉"，看变量怎么变（无比喻，纯代码追踪）

三样东西，落到变量上到底是啥：
  State      = agent 运行时的 state 字典（含 messages + 你加的字段）
  Checkpointer = 一个对象 saver，它内部用 thread_id 当钥匙存 state
  Thread ID  = config 里的一个字符串，是 saver 存/取的钥匙
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

# Checkpointer 就是一个对象，我们给它取个名字 cp，方便后面"打开抽屉"
cp = InMemorySaver()
agent = create_agent(model=llm, tools=[], system_prompt="复述用户告诉你的名字。",
                     checkpointer=cp)
CFG = {"configurable": {"thread_id": "t1"}}


def show(tag, state_or_response):
    """把 state 里我们关心的变量打印出来"""
    # invoke 的返回值就是整份 state 字典
    s = state_or_response
    print(f"  [{tag}] state 里的变量：")
    print(f"      messages 条数 = {len(s['messages'])}")
    print(f"      messages 内容 = {[ (m.type, m.content[:18]) for m in s['messages'] ]}")


print("=" * 70)
print("第 1 次 invoke（thread_id=t1）")
print("=" * 70)
r1 = agent.invoke({"messages": [{"role": "user", "content": "我叫小明"}]}, config=CFG)
show("第1次返回", r1)

print("\n>>> 现在用 get_tuple 直接打开 saver 里 thread_id=t1 的抽屉，看存了啥：")
tup = cp.get_tuple(CFG)                      # 这就是"抽屉"
cv = tup.checkpoint["channel_values"]         # 抽屉里的 state 变量
print("  抽屉里的 state 变量：")
print("      messages 条数 =", len(cv["messages"]))
print("      messages 内容 =", [(m.type, m.content[:18]) for m in cv["messages"]])

print("\n" + "=" * 70)
print("第 2 次 invoke（同一个 thread_id=t1）—— 看 state 怎么被续上")
print("=" * 70)
r2 = agent.invoke({"messages": [{"role": "user", "content": "我叫什么"}]}, config=CFG)
show("第2次返回", r2)
print("  ↑ 第 2 次返回的 messages 是 4 条：旧 2 条(从抽屉取回) + 新 2 条")

print("\n" + "=" * 70)
print("对比：没有 Checkpointer 会怎样（cp=None）")
print("=" * 70)
agent2 = create_agent(model=llm, tools=[], system_prompt="复述用户告诉你的名字。")  # 没传 checkpointer
r3 = agent2.invoke({"messages": [{"role": "user", "content": "我叫小明"}]}, config=CFG)
r4 = agent2.invoke({"messages": [{"role": "user", "content": "我叫什么"}]}, config=CFG)
print(f"  无 checkpointer：第2次返回的 messages 条数 = {len(r4['messages'])}（永远是 2，因为每次都从空 state 开始）")
print("  ↑ 每次 invoke 的 state 都是新建的全新字典，上一轮的变量没地方存")

print("\n" + "=" * 70)
print("对比：有 checkpointer 但不给 thread_id 会怎样")
print("=" * 70)
try:
    agent.invoke({"messages": [{"role": "user", "content": "hi"}]}, config={"configurable": {}})
except ValueError as e:
    print("  直接报错：", e)
print("  ↑ saver 拿不到钥匙(thread_id)，不知道往哪个抽屉存，拒绝工作")
