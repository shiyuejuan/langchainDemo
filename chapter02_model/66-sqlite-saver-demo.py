# -*- coding: utf-8 -*-
"""
66 - SqliteSaver：把对话保存到数据库（磁盘持久化）

对应课程：第 09 章「上下文与记忆」——短期记忆的"换后端"实验。
目标：把 59 课计数器的 InMemorySaver 换成 SqliteSaver(.sqlite 文件)，
      证明『对话/状态真的落盘，进程重启也不丢』。

关键 API（已查源码确认）：
    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
        ...   # 上下文管理器：with 块结束自动关连接，但 .sqlite 文件留在磁盘
    SqliteSaver 是 BaseCheckpointSaver 的子类，接口和 InMemorySaver 完全一致。
"""
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langgraph.graph import MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit("缺少 DEEPSEEK_API_KEY，请在项目根目录 .env 配置")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)

# 每次运行都从干净状态开始，避免重复累加干扰观察
DB = "checkpoints_demo.sqlite"
if os.path.exists(DB):
    os.remove(DB)


# ---- 自定义 state：和 59 课一样，多一个 counter 字段 ----
class MyState(MessagesState):
    counter: int = 0


@tool
def bump(question: str, runtime: ToolRuntime) -> Command:
    """把计数器加 1 并写回 state。question 是用户的问题。"""
    cur = runtime.state.get("counter", 0)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"计数已更新为 {cur + 1}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "counter": cur + 1,
        }
    )


def make_agent(saver):
    """用同一个 saver 造一个会计数的 agent（state_schema / 工具 / prompt 都固定）。"""
    return create_agent(
        model=llm,
        tools=[bump],
        system_prompt="用户每来一次，就调用 bump 工具记一次数。",
        state_schema=MyState,
        checkpointer=saver,
    )


CFG = {"configurable": {"thread_id": "t1"}}


def show(saver, label):
    tp = saver.get_tuple(CFG)
    if tp is None:
        print(f"  [{label}] get_tuple → None（没有任何 checkpoint）")
        return None
    ch = tp.checkpoint["channel_values"]
    print(
        f"  [{label}] counter={ch.get('counter')!r} | "
        f"messages 条数={len(ch.get('messages', []))}"
    )
    return ch


def banner(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


# ============ 阶段一：第一次运行（写入数据库） ============
banner("阶段一：第一次运行，对话写到 .sqlite 文件")
with SqliteSaver.from_conn_string(DB) as saver:
    agent = make_agent(saver)
    for i in range(3):
        agent.invoke(
            {"messages": [{"role": "user", "content": f"第 {i + 1} 次来"}]},
            config=CFG,
        )
    show(saver, "阶段一结束后（在 with 块内读）")

print(f"\n此时磁盘上数据库文件：{DB}  大小={os.path.getsize(DB)} bytes")
print("（进程还没结束，但数据已经写进文件了——这才是和内存版的本质区别）")

# ============ 阶段二：模拟『进程重启』 ============
# 关键：saver 已经随 with 块关闭（连接断开），我们重新打开【同一个文件】，
#       并且新建一个全新的 agent / 全新的 saver 对象——完全不依赖阶段一的内存。
banner("阶段二：模拟进程重启 —— 重新打开同一个 .sqlite 文件")
print(f"数据库文件还在吗：{os.path.exists(DB)}  大小={os.path.getsize(DB)} bytes")

with SqliteSaver.from_conn_string(DB) as saver2:
    # 全新 saver2、全新 agent2，但指向同一文件
    agent2 = make_agent(saver2)
    ch2 = show(saver2, "重启后立即读（没做任何 invoke）")
    assert ch2 is not None, "文件里居然没有数据？那说明没持久化成功"

    # 接着聊：第 4 次
    agent2.invoke(
        {"messages": [{"role": "user", "content": "第 4 次来"}]},
        config=CFG,
    )
    show(saver2, "重启后第 4 次 invoke 之后")

# ============ 对照：InMemorySaver 做不到这点 ============
banner("对照实验：全新 InMemorySaver 读同一个 thread_id")
mem = InMemorySaver()  # 一个全新的、和文件毫无关系的纯内存 saver
show(mem, "全新 InMemorySaver")
print("  ↑ None：内存里啥都没有。SqliteSaver 因为落盘了，重启后还能读到。")

# ============ 阶段三：直接打开 .sqlite 看里面到底存了啥 ============
banner("阶段三：直接用 sqlite3 打开文件，看数据库里的表")
conn = sqlite3.connect(DB)
cur = conn.cursor()
tables = cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()
print("数据库里的表：", [t[0] for t in tables])

try:
    rows = cur.execute(
        "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id"
    ).fetchall()
    print("checkpoints 表（按线程统计记录数）：", rows)
except Exception as e:
    print("读取 checkpoints 表出错：", e)
conn.close()

print("\n演示结束：SqliteSaver 把 checkpoint 存进 .sqlite 文件，")
print("进程退出后再打开同一文件，历史对话和 counter 都还在。")
