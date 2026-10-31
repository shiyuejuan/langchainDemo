# -*- coding: utf-8 -*-
"""59 · 短期记忆案例②：按线程计数（counter 存进 state，跨 invoke 累加）

和 47 课的 user_name 不同，这里演示的是：state 里一个 *会自己增长* 的字段，
靠 checkpointer 跨 invoke 存住。每次对话，工具把 counter +1 并回写 state。

核心机制：工具返回 langgraph.types.Command(update={...}) 来“写回 state 的某个键”。
这比 user_name（每次靠输入传）更动态地展示了“短期记忆在变、且被存住”。
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from langgraph.types import Command

llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url="https://api.deepseek.com/v1", temperature=0)


# 自定义 state：除了默认 messages，加一个 counter，初始 0
class MyState(MessagesState):
    counter: int = 0


@tool
def bump(question: str, runtime: ToolRuntime) -> Command:
    """把计数器加 1 并写回 state。question 是用户的问题。"""
    cur = runtime.state.get("counter", 0)
    # Command.update 用“通道名 -> 值”的字典形式：
    #  - "messages" 通道（带 add_messages reducer）放对账用的 ToolMessage
    #  - "counter"  通道放我们要写回的数字
    return Command(update={
        "messages": [ToolMessage(content=f"计数已更新为 {cur + 1}",
                                 tool_call_id=runtime.tool_call_id)],
        "counter": cur + 1,
    })


cp = InMemorySaver()
agent = create_agent(model=llm, tools=[bump],
                     system_prompt="用户每来一次，就调用 bump 工具记一次数。",
                     state_schema=MyState, checkpointer=cp)
CFG = {"configurable": {"thread_id": "t1"}}   # 同一个线程 = 同一段短期记忆

print("===== 同一个 thread_id=t1，连问 3 次 =====")
for i in range(3):
    r = agent.invoke({"messages": [{"role": "user", "content": "我来了"}]}, config=CFG)
    print(f"第 {i+1} 次 invoke 后 → state.counter = {r['counter']}")

print("\n===== 换个线程 thread_id=t2（全新记忆，应该从头开始） =====")
CFG2 = {"configurable": {"thread_id": "t2"}}
r = agent.invoke({"messages": [{"role": "user", "content": "我来了"}]}, config=CFG2)
print(f"t2 第 1 次 invoke 后 → state.counter = {r['counter']}  （不是 4，而是 1 的起始 0+1）")

print("\n===== 回到 t1（记忆还在，继续累加） =====")
r = agent.invoke({"messages": [{"role": "user", "content": "我又来了"}]}, config=CFG)
print(f"t1 第 4 次 invoke 后 → state.counter = {r['counter']}  （接着 3 变成 4，证明 t1 记忆没丢）")
