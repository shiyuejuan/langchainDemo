# -*- coding: utf-8 -*-
"""60 · context 的「增删改」实跑版

课程 1.2.2 第三种上下文 context（静态运行时 / 工牌）的增、删、改。

核心结论先说：
  context 不是"数据库"，不被 checkpointer 保存、框架也不帮你改它。
  - 增 = 在 @dataclass 定义里加字段 + 构造实例时提供
  - 删 = 从定义里删字段 / 这次 invoke 不传 context（runtime.context 变 None）
  - 改 = 下一次 invoke 传入【新的/改过的】实例；工具里直接改对象属性也能"看见"，
        但那是 Python 对象共享，不是框架记忆——换线程/重新构造实例就丢了。

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
    raise SystemExit("缺少 DEEPSEEK_API_KEY：请确认 Demo/langchainDemo/.env 存在且包含该键")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)


# ─────────────────────────────────────────────────────────────
# 基准工牌：3 个字段
# ─────────────────────────────────────────────────────────────
@dataclass
class AppContext:
    user_id: str
    env: str
    db_host: str


class S(MessagesState):
    pass


@tool
def show_ctx(question: str, runtime: ToolRuntime) -> str:
    """打印并汇报当前 context 的内容。question 是用户的问题。"""
    ctx = runtime.context
    if ctx is None:
        return "【runtime.context 是 None】本次 invoke 没有传 context"
    # region 可能不存在（演示"增"前 vs "增"后），用 getattr 兼容
    region = getattr(ctx, "region", "<尚未定义>")
    return (f"user_id={ctx.user_id}, env={ctx.env}, "
            f"db_host={ctx.db_host}, region={region!r}")


def make_agent(ctx_schema):
    return create_agent(
        model=llm,
        tools=[show_ctx],
        system_prompt="你只能调用 show_ctx 工具获取上下文，并把工具返回的内容原样告诉用户，禁止编造。",
        state_schema=S,
        context_schema=ctx_schema,
        checkpointer=InMemorySaver(),
    )


def ask(agent, text, cfg, context=None):
    """统一调用：context 为哨兵 object 表示"不传"；否则正常传。"""
    kwargs = {"config": cfg}
    if context is not _OMIT:
        kwargs["context"] = context
    return agent.invoke({"messages": [{"role": "user", "content": text}]}, **kwargs)


_OMIT = object()


def banner(t):
    print("\n" + "=" * 64)
    print(t)
    print("=" * 64)


# ════════════════════════════════════════════════════════════
# ① 增（ADD）：在 dataclass 里加一个 region 字段
#    做法 = 改定义 + 构造实例时提供。工具用 getattr 兼容新旧两种 schema。
# ════════════════════════════════════════════════════════════
@dataclass
class AppContextV2:          # ← 在 V1 基础上"增加"了 region
    user_id: str
    env: str
    db_host: str
    region: str              # ← 新增字段


banner("① 增：构造带 region 的新工牌，工具读到了新字段")
agent = make_agent(AppContextV2)
CTX2 = AppContextV2(user_id="u-1001", env="test", db_host="db:5432", region="cn-east")
r = ask(agent, "汇报当前 context。", {"configurable": {"thread_id": "t-add"}}, CTX2)
print("工具回:", r["messages"][-1].content)


# ════════════════════════════════════════════════════════════
# ② 删（DELETE）：这次 invoke 不传 context
#    结果：runtime.context 是 None，工具里必须判空，否则 AttributeError。
# ════════════════════════════════════════════════════════════
banner("② 删：不传 context（_OMIT 表示省略该参数），工具读到 None")
agent = make_agent(AppContext)          # 仍然声明 schema，只是本次不传
r = ask(agent, "汇报当前 context。", {"configurable": {"thread_id": "t-del"}}, _OMIT)
print("工具回:", r["messages"][-1].content)


# ════════════════════════════════════════════════════════════
# ③ 改-A（UPDATE，推荐）：下一次 invoke 传入"改过"的实例
#    框架不会帮你改 context，你只能自己构造新实例再传进去。
# ════════════════════════════════════════════════════════════
banner("③ 改-A：先传 env=test，再传 env=prod（重新构造实例）")
agent = make_agent(AppContext)
cfg = {"configurable": {"thread_id": "t-upd"}}
r1 = ask(agent, "汇报当前 context。", cfg, AppContext("u-1", "test", "db:5432"))
print("第 1 次(env=test):", r1["messages"][-1].content)
r2 = ask(agent, "再汇报一次。", cfg, AppContext("u-1", "prod", "db:5432"))  # 换了实例
print("第 2 次(env=prod):", r2["messages"][-1].content)


# ════════════════════════════════════════════════════════════
# ④ 改-B（工具里直接改对象属性）—— 演示"它能改，但不是框架记忆"
#    同一个可变实例 CTX 被反复传入：工具里改了 env，下一轮传同一实例能看见；
#    但换一个【全新】实例传入，改动立刻消失 → 证明框架没保存它。
# ════════════════════════════════════════════════════════════
banner("④ 改-B：工具里把 env 改成 'MUTATED'，再用'读'工具看是否被'记住'")
agent = make_agent(AppContext)
cfg = {"configurable": {"thread_id": "t-mut"}}
shared = AppContext("u-1", "test", "db:5432")     # 同一个可变实例


@tool
def peek_env(question: str, runtime: ToolRuntime) -> str:
    """读取并返回当前 context 的 env 值。question 是用户的问题。"""
    if runtime.context is None:
        return "context 为空"
    return f"当前 env = {runtime.context.env}"


@tool
def mutate_env(question: str, runtime: ToolRuntime) -> str:
    """在工具内部把 context 的 env 属性改成 'MUTATED'。question 是用户的问题。"""
    if runtime.context is None:
        return "context 为空，无法修改"
    runtime.context.env = "MUTATED"               # 直接改对象属性（不推荐，仅演示）
    return "已把 env 改成 MUTATED"


agent_mut = create_agent(
    model=llm, tools=[peek_env, mutate_env],
    system_prompt="你只能调用 peek_env 或 mutate_env 工具，并把工具返回内容原样告诉用户。",
    state_schema=S, context_schema=AppContext,
    checkpointer=InMemorySaver(),
)

r_a = agent_mut.invoke({"messages": [{"role": "user", "content": "读一下 env。"}]},
                        config=cfg, context=shared)
print("a. 读(同一实例 shared):", r_a["messages"][-1].content, "  ← 期望 test")
r_b = agent_mut.invoke({"messages": [{"role": "user", "content": "把 env 改掉。"}]},
                        config=cfg, context=shared)   # 仍传同一个 shared
print("b. 改(仍传 shared):", r_b["messages"][-1].content)
r_c = agent_mut.invoke({"messages": [{"role": "user", "content": "再读一下 env。"}]},
                        config=cfg, context=shared)   # 仍传同一个 shared
print("c. 再读(仍传 shared):", r_c["messages"][-1].content,
      "  ← 看见 MUTATED：这是 Python 对象共享，不是框架记忆")
r_d = agent_mut.invoke({"messages": [{"role": "user", "content": "再读一下 env。"}]},
                        config=cfg,
                        context=AppContext("u-1", "test", "db:5432"))  # 全新实例
print("d. 读(全新实例):", r_d["messages"][-1].content,
      "  ← 回到 test：证明框架没保存改动")


# ════════════════════════════════════════════════════════════
# ⑤ 关键验证：context 不被 checkpointer 记住
#    同一 thread 第 1 次传 CTX，第 2 次【完全不传】context，
#    看工具读到的是旧值还是 None。
# ════════════════════════════════════════════════════════════
banner("⑤ 关键：同一线程第 2 次不传 context，框架是否'记得'旧值？")
agent = make_agent(AppContext)
cfg = {"configurable": {"thread_id": "t-remember"}}
r1 = ask(agent, "汇报 context。", cfg, AppContext("u-9", "test", "db:5432"))
print("第 1 次(传了):", r1["messages"][-1].content)
r2 = ask(agent, "再汇报一次。", cfg, _OMIT)        # 同一线程，但这次不传
print("第 2 次(同线程·不传):", r2["messages"][-1].content,
      "← None，证明 context 没被 checkpoint 记住")

print("\n全部演示结束。")
