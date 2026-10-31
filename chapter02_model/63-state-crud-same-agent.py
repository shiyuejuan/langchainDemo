# -*- coding: utf-8 -*-
"""63-state-crud-same-agent.py
同一个 agent 内，针对「某个具体属性」的增删改边界实跑
对照 61 课：context 在同一 agent 内只能改值、不能增删字段（schema 冻结）
本课验证 state 是否也一样，以及 state 改值相比 context 多出的「记住」能力
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit("缺少 DEEPSEEK_API_KEY，请在项目根目录 .env 配置")
llm = ChatOpenAI(model="deepseek-chat", api_key=API_KEY,
                 base_url="https://api.deepseek.com/v1", temperature=0)


# 同一个 state_schema：声明了 user_name / city / counter 三条通道
class S(MessagesState):
    user_name: str = ""
    city: str = ""        # 第二个属性，用来演示「某个属性」独立改、不影响别的
    counter: int = 0


# 同一个 agent 对象，挂了 4 个工具（改/增/删都在这一个 agent 里）
@tool
def set_name(q: str, runtime: ToolRuntime) -> Command:
    """把 user_name 设为「明明」。q 是用户问题。"""
    return Command(update={
        "messages": [ToolMessage(content="user_name=明明", tool_call_id=runtime.tool_call_id)],
        "user_name": "明明",
    })


@tool
def set_city(q: str, runtime: ToolRuntime) -> Command:
    """把 city 设为「北京」。q 是用户问题。"""
    return Command(update={
        "messages": [ToolMessage(content="city=北京", tool_call_id=runtime.tool_call_id)],
        "city": "北京",
    })


@tool
def add_field(q: str, runtime: ToolRuntime) -> Command:
    """尝试写入一个【未声明】的属性 vip。q 是用户问题。"""
    return Command(update={
        "messages": [ToolMessage(content="尝试写入 vip", tool_call_id=runtime.tool_call_id)],
        "vip": "yes",   # ← 未在 class S 里声明
    })


@tool
def clear_city(q: str, runtime: ToolRuntime) -> Command:
    """把 city 覆盖为空字符串（安全做法，不是删通道）。q 是用户问题。"""
    return Command(update={
        "messages": [ToolMessage(content="city 置空", tool_call_id=runtime.tool_call_id)],
        "city": "",
    })


cp = InMemorySaver()   # 独立 saver，自己持有，方便事后 get_tuple 看通道
agent = create_agent(
    model=llm,
    tools=[set_name, set_city, add_field, clear_city],
    system_prompt="根据用户要求调用对应工具，并原样返回工具结果。",
    state_schema=S,
    checkpointer=cp,
)


def run(thread: str, msg: str):
    """用同一个 agent 对象，在不同 thread 上触发不同工具。"""
    cfg = {"configurable": {"thread_id": thread}}
    r = agent.invoke({"messages": [{"role": "user", "content": msg}]}, config=cfg)
    tp = cp.get_tuple(cfg)
    ch = tp.checkpoint["channel_values"]
    return r, ch


def banner(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


# ------------------------------------------------------------
# 实验 1：同一 agent 内【改某个属性的值】，且跨轮记住、多属性互不影响
# ------------------------------------------------------------
banner("① 同一 agent 内【改值】：改 user_name，再改 city，证明都记住且互不干扰")
r1, ch1 = run("t1", "调用 set_name 工具")
print("调 set_name 后: user_name=", repr(r1.get("user_name", "<未写入>")),
      "| city=", repr(r1.get("city", "<未写入>")))
r2, ch2 = run("t1", "调用 set_city 工具")   # 同一 thread t1，不调 set_name
print("再调 set_city 后: user_name=", repr(r2.get("user_name", "<未写入>")),
      "| city=", repr(r2.get("city", "<未写入>")),
      "← user_name 仍是明明(记住)，city=北京(新改)，两属性独立")
print("checkpoint 里两条通道都在:", "user_name" in ch2, "和", "city" in ch2)

# ------------------------------------------------------------
# 实验 2：同一 agent 内【增】一个未声明的属性
# ------------------------------------------------------------
banner("② 同一 agent 内【增】未声明属性 vip：会被忽略")
r3, ch3 = run("t2", "调用 add_field 工具")
print("返回值里有 vip 吗:", "vip" in r3)
print("checkpoint channel_values 键:", list(ch3.keys()))
print("→ vip 既不在返回值也不在 checkpoint，证明『增属性』被忽略(schema 冻结)")

# ------------------------------------------------------------
# 实验 3：同一 agent 内【删】某个属性：只能覆盖值，不能删通道
# ------------------------------------------------------------
banner("③ 同一 agent 内【删】city：置空(覆盖值)，通道本身仍在")
r4, ch4 = run("t3", "调用 set_city 工具")
print("先设 city:", repr(r4.get("city", "<未写入>")))
r5, ch5 = run("t3", "调用 clear_city 工具")   # 同一 thread t3
print("再 clear_city 后 city=", repr(r5.get("city", "<未写入>")))
print("通道 city 还在吗:", "city" in ch5, "← True：删的是值，不是通道；删字段只能改类定义重建")

print("\n演示结束：同一个 agent 内，针对某个属性——"
      "改值✅(且跨轮记住、多属性独立)；增未声明属性❌(忽略)；删❌(只能覆盖值，通道由 schema 定)。")
