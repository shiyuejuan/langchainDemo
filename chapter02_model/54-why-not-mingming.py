# -*- coding: utf-8 -*-
"""54 · 验证：第 2 轮 user_name='明明'，state 记的到底是啥，回答为啥还是小明"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import MessagesState

llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url="https://api.deepseek.com/v1", temperature=0)

@dataclass
class AppContext:
    user_id: str; env: str; db_host: str

class MyState(MessagesState):
    user_name: str

store = InMemoryStore()
store.put(("users", "xiaoming"), "preferences", {"language": "中文", "style": "简短", "hobby": "天文"})

@tool
def my_info(question: str, runtime: ToolRuntime) -> str:
    """需要确认白板用户名时调用。"""
    return f"白板用户名={runtime.state.get('user_name')}"

agent = create_agent(model=llm, tools=[my_info],
    system_prompt="需要环境信息时调用 my_info 工具。",
    state_schema=MyState, context_schema=AppContext,
    store=store, checkpointer=InMemorySaver())
CFG = {"configurable": {"thread_id": "t1"}}
CTX = AppContext(user_id="u-1001", env="test", db_host="db-internal:5432")

# 第 1 轮
r1 = agent.invoke({"messages": [{"role":"user","content":"我叫什么？"}], "user_name":"小明"},
                  config=CFG, context=CTX)
print("第 1 轮回答:", r1["messages"][-1].content)
print("  第 1 轮后 state.user_name =", r1["user_name"])

# 第 2 轮：改 user_name=明明，并明确说“不要调工具”
r2 = agent.invoke({"messages": [{"role":"user","content":"再汇报一次我是谁，不要调工具，直接答。"}],
                   "user_name":"明明"}, config=CFG, context=CTX)
print("\n第 2 轮回答:", r2["messages"][-1].content)
print("  第 2 轮后 state.user_name =", r2["user_name"], "  ← 注意：白板确实被改成了 明明")
print("  第 2 轮模型‘看到了’的 messages 角色顺序:")
for m in r2["messages"]:
    tag = {"system":"[系统]","user":"[用户]","ai":"[AI]","tool":"[工具]"}.get(m.type, m.type)
    print(f"    {tag}: {m.content[:28]}")
