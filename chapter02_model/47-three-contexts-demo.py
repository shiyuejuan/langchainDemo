# -*- coding: utf-8 -*-
"""47 · 三种上下文一锅端：state / store / context

课程 1.2.2「上下文类型及相关的 API」实跑版：
  ① 动态运行时上下文 → state     （可变 · 单次运行）  LangGraph state 对象
  ② 动态跨会话上下文 → store     （可变 · 跨对话）    LangGraph store 对象
  ③ 静态运行时上下文 → context   （静态 · 单次运行）  LangGraph context 对象

玩法：一个工具 my_info(runtime: ToolRuntime) 同时读三种上下文，
用打印证明"工具里三样东西都摸得到"。

依赖环境变量 DEEPSEEK_API_KEY。
"""
import os
from pathlib import Path

from dotenv import load_dotenv  # 自动加载 .env，免去手动 source
from dataclasses import dataclass

# 自动找项目根目录的 .env（chapter02_model 的上一级），没有就静默跳过
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import MessagesState

if not os.getenv("DEEPSEEK_API_KEY"):
    raise SystemExit(
        "缺少 DEEPSEEK_API_KEY：请确认 Demo/langchainDemo/.env 存在且包含该键"
    )

deepseek = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),     # getenv：缺了返回 None，配合上面的友好提示
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)

# ─────────────────────────────────────────────────────────────
# ③ 静态运行时上下文：context_schema —— 启动时传入、运行中不变
#    类比：进场时发的"工牌"，上面写死了你是谁、连的哪个库
# ─────────────────────────────────────────────────────────────
@dataclass
class AppContext:
    """本次运行开始前就定死的元数据"""
    user_id: str
    env: str          # prod / test
    db_host: str      # 模拟"数据库连接"这类静态资源

# ─────────────────────────────────────────────────────────────
# ① 动态运行时上下文：state —— 单次运行内会"演变"的可变数据
#    除默认 messages 外，加一个自定义键 user_name，
#    类比：车间里那块"白板"，运行中谁都能往上写
# ─────────────────────────────────────────────────────────────
class MyState(MessagesState):
    user_name: str   # 自定义键，随运行演变

# ─────────────────────────────────────────────────────────────
# ② 动态跨会话上下文：store —— 跨对话共享的持久数据
#    类比：公司"档案柜"，今天来查、明天来还能查到
# ─────────────────────────────────────────────────────────────
store = InMemoryStore()
# 会话开始前，先往档案柜里放一条用户偏好（namespace, key, value）
store.put(("users", "xiaoming"), "preferences",
          {"language": "中文", "style": "简短", "hobby": "天文"})

# ─────────────────────────────────────────────────────────────
# 关键角色：一个工具同时摸三种上下文
# ─────────────────────────────────────────────────────────────
@tool
def my_info(question: str, runtime: ToolRuntime) -> str:
    """汇报我在工具里能看到的三种上下文。question 是用户的问题。"""
    print("\n" + "=" * 60)
    print("【工具内部实况】三种上下文，工具里全都摸得到：")

    # ① state（动态运行时）—— 单次运行中的白板
    state = runtime.state
    print(f"  ① state   类型={type(state).__name__}")
    print(f"     键有哪些：{list(state.keys())}")
    print(f"     消息条数={len(state.get('messages', []))}，自定义键 user_name={state.get('user_name')!r}")

    # ② store（动态跨会话）—— 跨对话的档案柜
    item = runtime.store.get(("users", "xiaoming"), "preferences")
    print(f"  ② store   类型={type(runtime.store).__name__}")
    print(f"     查档案柜 (users, xiaoming)/preferences → {item.value}")

    # ③ context（静态运行时）—— 进场发的工牌
    ctx = runtime.context
    print(f"  ③ context 类型={type(ctx).__name__}")
    print(f"     工牌内容：user_id={ctx.user_id}, env={ctx.env}, db_host={ctx.db_host}")
    print("=" * 60 + "\n")
    return (
        f"资料汇总 → 白板上的用户名：{state.get('user_name')}；"
        f"档案柜偏好：语言={item.value['language']}，爱好={item.value['hobby']}；"
        f"工牌环境：{ctx.env}（库在 {ctx.db_host}）"
    )

# ─────────────────────────────────────────────────────────────
# 组装：state_schema / context_schema / store / checkpointer 一次挂齐
# ─────────────────────────────────────────────────────────────
agent = create_agent(
    model=deepseek,
    tools=[my_info],
    system_prompt="你是助手。需要环境信息时调用 my_info 工具，然后根据工具结果用一句话回答。",
    state_schema=MyState,            # ① 自定义 state
    context_schema=AppContext,       # ③ 静态 context
    store=store,                     # ② 跨会话 store
    checkpointer=InMemorySaver(),    # 让 state 跨 invoke 续上（45课讲过）
)

CFG = {"configurable": {"thread_id": "t1"}}
CTX = AppContext(user_id="u-1001", env="test", db_host="db-internal:5432")

print("──── 第 1 次 invoke：传入初始 state + 静态 context ────")
# 注意：context 是 invoke 的【顶层关键字参数】，不塞进 config！
r1 = agent.invoke(
    {
        "messages": [{"role": "user", "content": "我是谁？我的偏好和环境是什么？"}],
        "user_name": "小明",                      # ① 自定义 state 键，随输入进来
    },
    config=CFG,
    context=CTX,                                  # ③ 静态 context，运行启动时传入
)
print("最终回答：", r1["messages"][-1].content)

print("\n──── 第 2 次 invoke：同一线程，验证 ①state 白板被带着走 ────")
r2 = agent.invoke(
    {"messages": [{"role": "user", "content": "再汇报一次我是谁，不要调工具，直接答。"}],
     "user_name": "明明"},                        # 白板上的名字被改写了 → 演变
    config=CFG,
    context=CTX,
)
print("最终回答：", r2["messages"][-1].content)
print("第 2 轮 state 消息条数 =", len(r2["messages"]), "（历史续上了）")
