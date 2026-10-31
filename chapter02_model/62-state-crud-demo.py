# -*- coding: utf-8 -*-
"""62-state-crud-demo.py
state 的增删改边界实跑（对照 context 的 60/61 课）

核心区别回顾（源码已证实）：
  - state_schema 被 StateGraph._add_schema 拆成「通道(channel)」，每个字段是一条通道
  - context_schema 只被存着当类型标注，不拆通道、不进 checkpoint
所以 state 是「字典形态、字段=通道、被 checkpointer 记住」；
      context 是「单个对象、点属性、不记住」。
本课验证 state 在「设计期」和「运行时」分别能怎么增删改。
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit("缺少 DEEPSEEK_API_KEY，请在项目根目录 .env 配置")
llm = ChatOpenAI(model="deepseek-chat", api_key=API_KEY,
                 base_url="https://api.deepseek.com/v1", temperature=0)


# ============================================================
# 设计期 state 定义：加一个字段 = 加一条「通道(channel)」
# 这部分在 create_agent 时冻结，运行时不能改类定义（和 context 同理）
# ============================================================
class S(MessagesState):
    user_name: str = ""     # 自定义字段 → 编译后成为一条通道
    counter: int = 0


def banner(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


# ------------------------------------------------------------
# 实验 A：运行时【改值】—— state 的核心能力，且会被记住
# ------------------------------------------------------------
@tool
def set_name(q: str, runtime: ToolRuntime) -> Command:
    """把用户名设置为“明明”。q 是用户问题。"""
    return Command(update={
        "messages": [ToolMessage(content="已将 user_name 设为 明明",
                                 tool_call_id=runtime.tool_call_id)],
        "user_name": "明明",
    })


banner("A. 运行时【改值】：写回 user_name，并验证跨 invoke 被记住")
cpA = InMemorySaver()
agentA = create_agent(model=llm, tools=[set_name],
                      system_prompt="用户说话就调用 set_name 工具。",
                      state_schema=S, checkpointer=cpA)
cfgA = {"configurable": {"thread_id": "a"}}
r1 = agentA.invoke({"messages": [{"role": "user", "content": "我是明明"}]}, config=cfgA)
print("第 1 次 invoke 后, state['user_name'] =", repr(r1["user_name"]))
r2 = agentA.invoke({"messages": [{"role": "user", "content": "你是谁"}]}, config=cfgA)
print("第 2 次 invoke 后, state['user_name'] =", repr(r2["user_name"]),
      "← 仍是明明，证明 state 被 checkpoint 记住（对比 context 不记住）")


# ------------------------------------------------------------
# 实验 B：运行时【增】一个 schema 未声明的键，看 LangGraph 行为
# ------------------------------------------------------------
@tool
def add_extra(q: str, runtime: ToolRuntime) -> Command:
    """尝试写入一个 state 里没声明的字段 extra_field。q 是用户问题。"""
    return Command(update={
        "messages": [ToolMessage(content="尝试写入 extra_field",
                                 tool_call_id=runtime.tool_call_id)],
        "extra_field": "hello",
    })


banner("B. 运行时【增】未声明键：Command(update={'extra_field':...}) 会怎样")
cpB = InMemorySaver()
agentB = create_agent(model=llm, tools=[add_extra],
                      system_prompt="只能调用 add_extra 工具。",
                      state_schema=S, checkpointer=cpB)
cfgB = {"configurable": {"thread_id": "b"}}
try:
    rB = agentB.invoke({"messages": [{"role": "user", "content": "加字段"}]}, config=cfgB)
    print("invoke 返回里有没有 extra_field 键:", "extra_field" in rB)
    tp = cpB.get_tuple(cfgB)
    print("checkpoint channel_values 的键:", list(tp.checkpoint["channel_values"].keys()))
except Exception as e:
    print("报错:", type(e).__name__, "→", str(e).splitlines()[0])


# ------------------------------------------------------------
# 实验 C：运行时【删】—— 覆盖置空 vs 真正删除通道
# ------------------------------------------------------------
@tool
def clear_name(q: str, runtime: ToolRuntime) -> Command:
    """把 user_name 覆盖为空字符串（安全做法）。q 是用户问题。"""
    return Command(update={
        "messages": [ToolMessage(content="已将 user_name 置空",
                                 tool_call_id=runtime.tool_call_id)],
        "user_name": "",
    })


banner("C. 运行时【删】值：覆盖置空（通道本身还在，只是值为空）")
cpC = InMemorySaver()
agentC = create_agent(model=llm, tools=[clear_name],
                      system_prompt="只能调用 clear_name 工具。",
                      state_schema=S, checkpointer=cpC)
cfgC = {"configurable": {"thread_id": "c"}}
rC = agentC.invoke({"messages": [{"role": "user", "content": "清空名字"}]}, config=cfgC)
print("置空后 state['user_name'] =", repr(rC["user_name"]))
tpC = cpC.get_tuple(cfgC)
print("通道里仍有 user_name 键吗:", "user_name" in tpC.checkpoint["channel_values"],
      "← 删的是值，不是通道；通道定义由 schema 决定")


# ------------------------------------------------------------
# 实验 D：设计期【增/删字段】—— 改类定义后必须重建 agent
# （这里用代码展示，不实际重建两次，避免冗长）
# ------------------------------------------------------------
banner("D. 设计期【增/删字段】：改类定义 = 加/删通道，必须重建 agent")
print("加字段：在 class S(MessagesState) 里加一行 `vip: bool = False` → 编译多一条通道")
print("删字段：把那行删掉 → 编译少一条通道；重建 agent 后旧 checkpoint 里该通道的值不再被读取")
print("结论：state 的「字段增删」属于设计期；运行时既不能加未声明键(见 B)，也不能删通道(见 C)")

print("\n演示结束：state 运行时能【改值】(且被记住)、能【尝试增未声明键】(看 B 结果)、"
      "能【覆盖值】但不能删通道；通道的增删属于设计期(改类定义+重建 agent)。")
