# -*- coding: utf-8 -*-
"""64-state-write-ways.py
state 到底有几种「写数据」的方式？证明并不只有 Command(update=...)
验证三种用户可控入口：① invoke 输入字典  ② 工具普通 return（只进 messages）  ③ 工具 Command(update=...) 写自定义字段
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.messages import ToolMessage, AIMessage
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit("缺少 DEEPSEEK_API_KEY，请在项目根目录 .env 配置")
llm = ChatOpenAI(model="deepseek-chat", api_key=API_KEY,
                 base_url="https://api.deepseek.com/v1", temperature=0)


class S(MessagesState):
    user_name: str = ""


# 普通工具：只 return 一个字符串，不用 Command
@tool
def plain(q: str) -> str:
    """一个普通工具，只返回字符串，不写任何自定义字段。q 是用户问题。"""
    return "工具只返回了字符串（没有 Command）"


# Command 工具：写自定义字段 user_name
@tool
def cmd_set(q: str, runtime: ToolRuntime) -> Command:
    """通过 Command 把 user_name 写成「命令写的」。q 是用户问题。"""
    return Command(update={
        "messages": [ToolMessage(content="用 Command 写了 user_name",
                                 tool_call_id=runtime.tool_call_id)],
        "user_name": "命令写的",
    })


cp = InMemorySaver()
agent = create_agent(
    model=llm, tools=[plain, cmd_set],
    system_prompt="根据用户要求调用对应工具，并原样返回工具结果。",
    state_schema=S, checkpointer=cp,
)


def banner(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


# ------------------------------------------------------------
# 方式 ①：invoke 输入字典直接写 state（根本不需要 Command）
# ------------------------------------------------------------
banner("方式①：invoke 输入字典写 user_name（不调任何工具）")
r1 = agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}], "user_name": "输入小明"},
    config={"configurable": {"thread_id": "w1"}},
)
print("输入写 user_name =", repr(r1.get("user_name")),
      "← 没用 Command，输入字典直接写进了 state")
print("messages 条数 =", len(r1["messages"]), "（模型没调工具，只有用户+AI 两条）")

# ------------------------------------------------------------
# 方式 ②：工具普通 return —— 只进 messages 通道，不碰 user_name
# ------------------------------------------------------------
banner("方式②：工具普通 return（不 Command）—— 只写 messages，不写 user_name")
r2a = agent.invoke(
    {"messages": [{"role": "user", "content": "我是 A"}], "user_name": "A"},
    config={"configurable": {"thread_id": "w2"}},
)
r2b = agent.invoke(
    {"messages": [{"role": "user", "content": "调用 plain 工具"}]},
    config={"configurable": {"thread_id": "w2"}},
)
print("调普通工具前 user_name =", repr(r2a.get("user_name")))
print("调普通工具后 user_name =", repr(r2b.get("user_name")),
      "← 没变，仍是 A（普通 return 只进 messages，碰不到 user_name 通道）")
print("messages 条数 =", len(r2b["messages"]), "（多了 ToolMessage + 它的 AI 回复）")

# ------------------------------------------------------------
# 方式 ③：工具 Command(update=...) 写自定义字段（复习）
# ------------------------------------------------------------
banner("方式③：工具里 Command(update={'user_name':...}) 写自定义字段")
r3 = agent.invoke(
    {"messages": [{"role": "user", "content": "调用 cmd_set 工具"}]},
    config={"configurable": {"thread_id": "w3"}},
)
print("Command 写后 user_name =", repr(r3.get("user_name")),
      "← 这是工具改「自定义字段」的唯一正道")

print("\n演示结束：state 写数据不止 Command——"
      "①输入字典可写 ②工具普通return写messages ③工具Command写自定义字段。")
