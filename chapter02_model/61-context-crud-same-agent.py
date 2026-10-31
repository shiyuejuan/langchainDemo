# -*- coding: utf-8 -*-
"""61 · 同一个 agent 里，context 到底能不能"增删改"

结论预告：
  - 字段【值】可以改（重传实例）→ 同一 agent 完全支持
  - 字段【定义/schema】在同一 agent 内冻结 → 不能运行时加新字段/删字段
  - 工具里能动态 obj.新属性=... 或 del obj.字段 → 但只改了"那一个对象"，
    不进 schema、不被框架记住，换实例/换线程立刻归零
  - 真正要"运行期动态增删键值 + 跨轮记住" → 那是 store 的活儿，不是 context

依赖 DEEPSEEK_API_KEY。
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState

if not os.getenv("DEEPSEEK_API_KEY"):
    raise SystemExit("缺少 DEEPSEEK_API_KEY")

llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url="https://api.deepseek.com/v1", temperature=0)


# 同一个 agent，schema 固定为这 3 个字段
@dataclass
class AppContext:
    user_id: str
    env: str
    db_host: str


class S(MessagesState):
    pass


@tool
def read_all(q: str, runtime: ToolRuntime) -> str:
    """读取 context 当前所有字段，逐个判断是否存在。q 是用户问题。"""
    ctx = runtime.context
    if ctx is None:
        return "context 为 None"
    out = []
    for name in ["user_id", "env", "db_host", "region"]:   # region 不在 schema 里
        try:
            out.append(f"{name}={getattr(ctx, name)!r}")
        except AttributeError:
            out.append(f"{name}=<不存在>")
    return " | ".join(out)


@tool
def add_region(q: str, runtime: ToolRuntime) -> str:
    """在 context 对象上【动态】加一个 region 属性。q 是用户问题。"""
    runtime.context.region = "cn-east"     # schema 没这字段，但 Python 对象能硬加
    return "已在对象上加了 region=cn-east（仅当前对象，schema 并不知道）"


@tool
def del_dbhost(q: str, runtime: ToolRuntime) -> str:
    """把 context 的 db_host 字段【置空】（安全做法）。q 是用户问题。"""
    runtime.context.db_host = None      # 置空，而不是用 del 删属性
    return "已将 db_host 置空为 None（仅当前对象；字段定义仍在 schema 里）"


@tool
def crash_del(q: str, runtime: ToolRuntime) -> str:
    """【危险】真的用 Python del 删掉 db_host 属性（仅用于演示后果）。q 是用户问题。"""
    del runtime.context.db_host        # 删已声明字段：会让该字段在对象上彻底消失
    return "已 del db_host（字段从对象上消失了）"


agent = create_agent(
    model=llm, tools=[read_all, add_region, del_dbhost],
    system_prompt="你只能调用工具，并把工具返回内容原样告诉用户，不要编造。",
    state_schema=S, context_schema=AppContext,
    checkpointer=InMemorySaver(),
)

cfg = {"configurable": {"thread_id": "same-agent"}}
shared = AppContext("u-1", "test", "db:5432")   # 同一个可变实例


def banner(t):
    print("\n" + "=" * 64)
    print(t)
    print("=" * 64)


def ask(text, ctx):
    r = agent.invoke({"messages": [{"role": "user", "content": text}]},
                     config=cfg, context=ctx)
    return r["messages"][-1].content


banner("① 同一 agent，改【值】：先 env=test，再 env=prod（重传实例）")
print("env=test :", ask("读取所有字段。", AppContext("u-1", "test", "db:5432")))
print("env=prod :", ask("读取所有字段。", AppContext("u-1", "prod", "db:5432")))

banner("② 同一 agent，想'增'一个 schema 里没有的字段 region")
print("改前(shared):", ask("读取所有字段。", shared))            # region 不存在
print("加 region   :", ask("给 context 加一个 region 字段。", shared))   # 动态加
print("加后(同对象):", ask("读取所有字段。", shared))            # 同对象能看到
print("换全新实例  :", ask("读取所有字段。", AppContext("u-1", "test", "db:5432")),
      "← region 又没了：没进 schema、没被记住")

banner("③ 同一 agent，想'删'一个已声明字段 db_host（动态 del）")
fresh = AppContext("u-2", "test", "db:5432")
print("删前(fresh) :", ask("读取所有字段。", fresh))
print("删除 db_host:", ask("删掉 db_host 字段。", fresh))
print("删后(fresh) :", ask("读取所有字段。", fresh))            # 同对象没了
print("换全新实例  :", ask("读取所有字段。", AppContext("u-3", "test", "db:5432")),
      "← db_host 又回来了：schema 从没变过，置空只作用于那个对象")

banner("④ 危险演示：在同一 agent 里用 del 删【已声明】字段的真实后果（受保护）")
# 只有 crash_del 的专属 agent，先 del，再让框架把"残缺 context"重新注入下一个工具
agent_crash = create_agent(
    model=llm, tools=[crash_del, read_all],
    system_prompt="按用户要求调用工具，并原样返回结果。",
    state_schema=S, context_schema=AppContext, checkpointer=InMemorySaver(),
)
bad = AppContext("u-9", "test", "db:5432")
r1 = agent_crash.invoke({"messages": [{"role": "user", "content": "删掉 db_host 字段。"}]},
                         config=cfg, context=bad)
print("del 之后(本轮没事):", r1["messages"][-1].content)
# 下一个工具调用：框架要把 context 重新注入 read_all 的 ToolRuntime → 按 schema 校验
try:
    r2 = agent_crash.invoke({"messages": [{"role": "user", "content": "读取所有字段。"}]},
                            config=cfg, context=bad)   # 同一个残缺 bad 对象
    print("再读 db_host:", r2["messages"][-1].content, "（意外：没崩）")
except Exception as e:
    print("再调用工具 → 果然崩了:", type(e).__name__, ":",
          str(e).splitlines()[-1] if str(e) else e)
    print("  根因：框架把 context 重新注入下一个工具的 ToolRuntime 时，")
    print("        按 AppContext schema 校验发现 db_host 字段缺失 → AttributeError。")
    print("  结论：别用 del 删【已声明】字段；要'清空'就赋值 None（见 ③）。")

print("\n演示结束：context 在'同一 agent'内只能改值；"
      "加/删字段只是改某个对象，schema 冻结、不持久。")
print("补充：加'额外'属性没事，但 del'已声明'字段会触发框架重新序列化崩溃。")
