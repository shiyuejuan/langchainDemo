# -*- coding: utf-8 -*-
"""65-store-crud-same-agent.py
同一个 agent 内，store 的增删改（对照 context 60/61、state 62/63）
核心区别：store 是「档案柜」，运行时可自由增删改任意键值，且跨线程/跨会话共享（长期记忆）
不像 state/context 那样 schema 冻结
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit("缺少 DEEPSEEK_API_KEY，请在项目根目录 .env 配置")
llm = ChatOpenAI(model="deepseek-chat", api_key=API_KEY,
                 base_url="https://api.deepseek.com/v1", temperature=0)


# 同一个 store 对象，跨线程/跨会话共享（长期记忆）
store = InMemoryStore()


# 同一 agent 挂 4 个工具，分别负责增/改/查/删
@tool
def upsert_pref(q: str) -> str:
    """把用户 u1 的偏好写入 store（演示【增】+【改】）。q 是用户问题。"""
    ns = ("users", "u1")
    item = store.get(ns, "prefs")          # 先读旧的
    val = dict(item.value) if item else {}  # 局部合并技巧：拿出来→改→整体写回
    val["hobby"] = "天文"
    val["lang"] = "中文"
    store.put(ns, "prefs", val)            # put = 写或覆盖整条
    return f"已写入 store: {val}"


@tool
def read_pref(q: str) -> str:
    """读取 store 里 u1 的偏好（演示【查】get）。q 是用户问题。"""
    ns = ("users", "u1")
    item = store.get(ns, "prefs")
    return f"store 读出: {item.value if item else '无'}"


@tool
def list_prefs(q: str) -> str:
    """列出 store 里 users 命名空间下的所有偏好（演示 search 批量查）。q 是用户问题。"""
    items = store.search(("users",))
    if not items:
        return "store 里 users 下没有任何记录"
    return "store 共有 " + str(len(items)) + " 条: " + \
           "; ".join(f"{it.namespace[-1]}/{it.key}={it.value}" for it in items)


@tool
def del_pref(q: str) -> str:
    """删除 store 里 u1 的偏好（演示【删】delete）。q 是用户问题。"""
    ns = ("users", "u1")
    store.delete(ns, "prefs")
    return "已删除 u1 的 prefs"


agent = create_agent(
    model=llm,
    tools=[upsert_pref, read_pref, list_prefs, del_pref],
    system_prompt="根据用户要求调用对应工具，并原样返回工具结果。",
    state_schema=MessagesState,
    store=store,                 # ← store 通过这里挂进 agent
    checkpointer=InMemorySaver(),
)


def ask(thread: str, msg: str) -> str:
    """用同一个 agent 对象，在不同 thread 上触发不同工具，返回最终文本。"""
    r = agent.invoke({"messages": [{"role": "user", "content": msg}]},
                     config={"configurable": {"thread_id": thread}})
    return r["messages"][-1].content


def banner(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


# ------------------------------------------------------------
# 实验 1：【增】+【查】：在 t_write 线程写入，到【另一个】t_read 线程读取
# ------------------------------------------------------------
banner("① 增 + 查：写在一个线程，在【另一个线程】也能读到（跨线程共享）")
print("t_write 调 upsert_pref:", ask("t_write", "调用 upsert_pref 工具"))
print("t_read (不同线程!) 调 read_pref:", ask("t_read", "调用 read_pref 工具"),
      "← 另一个线程读到了，证明 store 不按 thread 隔离（state 做不到）")
print("   [store 真身] u1/prefs =", store.get(("users", "u1"), "prefs").value)

# ------------------------------------------------------------
# 实验 2：【改】= 同 ns+key 再 put = 覆盖整条（无局部更新）
# ------------------------------------------------------------
banner("② 改：同 ns+key 再 put = 覆盖整条（注意：直接 put 不合并，会丢旧字段）")
# 先用错误示范：直接 put 一个只有 lang 的 dict，hobby 会丢
store.put(("users", "u1"), "prefs", {"lang": "英文"})   # 不合并 → 整条覆盖
print("错误示范(直接 put {lang:英文} 覆盖)后 read_pref:", ask("t_read2", "调用 read_pref 工具"),
      "← hobby 没了，只剩 lang（覆盖整条）")
print("   [store 真身] u1/prefs =", store.get(("users", "u1"), "prefs").value)
# 正确做法：upsert_pref 里先 get 再合并（已演示在①），恢复
print("用 upsert_pref(先 get 合并再 put) 恢复:", ask("t_write3", "调用 upsert_pref 工具"))
print("   [store 真身] u1/prefs =", store.get(("users", "u1"), "prefs").value)

# ------------------------------------------------------------
# 实验 3：【search】批量查
# ------------------------------------------------------------
banner("③ search 批量列出 users 下的所有记录")
print(ask("t_list", "调用 list_prefs 工具"))
print("   [store 真身] search(('users',)) =",
      [(it.namespace, it.key, it.value) for it in store.search(("users",))])

# ------------------------------------------------------------
# 实验 4：【删】delete，删后再查应为空
# ------------------------------------------------------------
banner("④ 删：delete 后，再查应为『无』")
print("t_del 调 del_pref:", ask("t_del", "调用 del_pref 工具"))
print("t_read3 再 read_pref:", ask("t_read3", "调用 read_pref 工具"),
      "← 已删除，证明删成功")
print("   [store 真身] get u1/prefs =", store.get(("users", "u1"), "prefs"),
      "← None 表示已删")

print("\n演示结束：同一个 agent 内，store 增/改/查/删全部运行时动态可控；"
      "且跨线程共享（长期记忆），不受 schema 冻结限制（与 state/context 根本不同）。")
