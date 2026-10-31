# -*- coding: utf-8 -*-
"""
67-postgres-saver-demo.py
主题: SqliteSaver → PostgresSaver，证明"换一个 checkpointer 后端只改一行代码"。
和 66 课(SqliteSaver)的唯一区别: 用 PostgresSaver 替换 InMemorySaver/SqliteSaver。
其余 state / 工具 / prompt / thread_id 全部复用。

运行前环境要求(本课已为你装好):
  1. PostgreSQL 服务跑在 localhost:5432
  2. 库 langchain / 角色 langchain / 密码 langchain
  3. pip 包 langgraph-checkpoint-postgres(含 psycopg)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 关键: 让 demo 能找到项目根目录的 .env(里面有 DEEPSEEK_API_KEY)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from dataclasses import dataclass
from typing import Optional

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langgraph.graph import MessagesState
from langgraph.types import Command
from langgraph.checkpoint.postgres import PostgresSaver  # ← 本课主角

# ---- 1. 模型(和 59/66 课一样) ----
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)

# ---- 2. 自定义 state: 加一个 counter 字段(和 59/66 课一样) ----
class MyState(MessagesState):
    counter: int = 0

# ---- 3. 工具: 把 counter 加 1 并写回 state(和 59/66 课一样) ----
@tool
def bump(question: str, runtime: ToolRuntime) -> Command:
    """把计数器加 1 并写回 state。question 是用户的问题。"""
    cur = runtime.state.get("counter", 0)
    return Command(update={
        "messages": [ToolMessage(
            content=f"计数已更新为 {cur + 1}",
            tool_call_id=runtime.tool_call_id)],
        "counter": cur + 1,
    })

# ---- 4. 系统提示(和 59/66 课一样) ----
SYSTEM = "用户每来一次，就调用 bump 工具记一次数。"

# ---- 5. 数据库地址(唯一和 66 课不同的地方) ----
DB_URI = "postgresql://langchain:langchain@localhost:5432/langchain"

def banner(t):
    print("\n" + "=" * 56)
    print(t)
    print("=" * 56)

def read_counter(saver, thread):
    """不调任何工具，直接读某 thread 的 counter——证明数据真从数据库来。"""
    cfg = {"configurable": {"thread_id": thread}}
    tp = saver.get_tuple(cfg)
    if tp is None:
        return None
    return tp.checkpoint["channel_values"].get("counter")

# ===== 阶段一: 第一次启动，写入 3 次 =====
banner("阶段一: 第一次启动，thread=t1 连问 3 次")
with PostgresSaver.from_conn_string(DB_URI) as saver:
    saver.setup()  # ← 首次必须建表(幂等)，之后调用会跳过
    agent = create_agent(
        model=llm, tools=[bump], system_prompt=SYSTEM,
        state_schema=MyState, checkpointer=saver,
    )
    cfg = {"configurable": {"thread_id": "t1"}}
    for i in range(1, 4):
        agent.invoke({"messages": [{"role": "user", "content": f"第 {i} 次"}]}, config=cfg)
        print(f"  第 {i} 次 invoke 后 → state.counter = {read_counter(saver, 't1')}")
    print("  (进程此时可以'正常退出')")

# ===== 阶段二: 模拟'重启'——全新 saver + 全新 agent，打开同一个库 =====
banner("阶段二: 模拟重启(全新 saver + 全新 agent，读同一个 PostgreSQL 库)")
with PostgresSaver.from_conn_string(DB_URI) as saver2:
    # 注意: 不再 setup 也没关系(表已存在，setup 幂等)；这里演示'新 agent'
    agent2 = create_agent(
        model=llm, tools=[bump], system_prompt=SYSTEM,
        state_schema=MyState, checkpointer=saver2,
    )
    print("  重启后未做任何 invoke, 直接读 t1 → counter =", read_counter(saver2, "t1"))
    # 接着问第 4 次
    agent2.invoke({"messages": [{"role": "user", "content": "第 4 次"}]},
                  config={"configurable": {"thread_id": "t1"}})
    print("  第 4 次 invoke 后 → state.counter =", read_counter(saver2, "t1"),
          "← 从 3 接着到 4，记忆没丢")

# ===== 阶段三: 直接看 PostgreSQL 里的真实表 =====
banner("阶段三: 直接用 psql 看 PostgreSQL 里的真实表")
os.system(
    "PGPASSWORD=langchain psql -h localhost -p 5432 -U langchain -d langchain "
    "-c \"\\dt\" 2>/dev/null"
)
os.system(
    "PGPASSWORD=langchain psql -h localhost -p 5432 -U langchain -d langchain "
    "-c \"SELECT thread_id, count(*) AS rows FROM checkpoints GROUP BY thread_id;\" 2>/dev/null"
)

print("\n全部演示结束: PostgresSaver 把短期记忆落进 PostgreSQL，'重启'后照样续命。")
print("和 66 课(SqliteSaver)的区别——只有一行: 换 from_conn_string 的 URI。")
