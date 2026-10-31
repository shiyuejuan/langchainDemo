"""
28 - ContextEditingMiddleware 演示（上下文编辑中间件，LangChain 1.x + DeepSeek）

课程片段：8、ContextEditingMiddleware —— 上下文编辑中间件，提供上下文管理的一种方式。

源码确认（lc 1.2.12）：
  · ContextEditingMiddleware(*, edits=None, token_count_method='approximate')
      edits: 编辑策略列表（目前只有 ClearToolUsesEdit 一种）
      token_count_method: 'approximate' 粗略估算 / 'model' 用模型 tokenizer 精确算
  · ClearToolUsesEdit(trigger=100_000, clear_at_least=0, keep=3,
                      clear_tool_inputs=False, exclude_tools=(), placeholder='[cleared]')
      trigger          —— 输入 token 超过这个数才触发清理（默认 10 万）
      keep             —— 保留最近 N 条工具结果不动（默认 3）
      exclude_tools    —— 豁免名单（这些工具的结果永不清理）
      placeholder      —— 被清理的结果替换成这个文本（默认 '[cleared]'）
      clear_tool_inputs —— 是否连 AI 消息里的工具调用参数一起清
  · apply 逻辑：tokens > trigger → 找出所有 ToolMessage → 保留最近 keep 条
    → 其余的 content 替换成 placeholder（打上 context_editing.cleared 标记防重复处理）
  · 对齐 Anthropic 的 clear_tool_uses_20250919 策略

demo 验证两件事：
  ① 多轮对话后，旧工具结果是否被替换成占位符
  ② state 里的原始消息是"永久被改"还是"只清发给模型的那份"
（get_time 埋 2099 的技巧再次复用：这里是让工具返回带编号的长文本）
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich import print as rprint
load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


@tool
def search_web(query: str) -> str:
    """联网搜索，返回一段很长的网页内容。"""
    return (f"[{query} 的搜索结果] " + "这是一大段网页正文内容，用来撑大上下文。" * 120)


from langchain.agents.middleware import AgentMiddleware


class PeekMiddleware(AgentMiddleware):
    """偷看主模型实际收到的消息（在内层，看到的是清理后的版本）。"""

    def wrap_model_call(self, request, handler):
        # rprint("当前请求:", request);
        tool_msgs = [m for m in request.messages if type(m).__name__ == "ToolMessage"]
        print(f"   [偷看] 本轮发给模型的 ToolMessage 共 {len(tool_msgs)} 条:")
        for i, m in enumerate(tool_msgs, 1):
            head = str(m.content)[:30]
            print(f"     - 工具结果{i}: {head}...")
        return handler(request)


# trigger 调小到 300（默认 10 万），长文本一次就能超；
# keep=1 → 只保留最近 1 条工具结果；旧的全部换成占位符
mw = ContextEditingMiddleware(
    edits=[ClearToolUsesEdit(trigger=300, keep=1,
                             placeholder="[已清空: 旧搜索结果]")],
)
agent = create_agent(
    model=deepseek,
    tools=[search_web],
    middleware=[mw, PeekMiddleware()],
    checkpointer=InMemorySaver(),   # 多轮对话必需
)
config = {"configurable": {"thread_id": "demo-28"}}

# ── 第 1 轮：搜索"天气"，产生第 1 条工具结果 ──────────────────────
r1 = agent.invoke({"messages": [HumanMessage("帮我搜一下: 北京天气")]},
                  config=config)
print("第 1 轮回答:", r1["messages"][-1].content[:60], "...\n")

# ── 第 2 轮：再搜"股市"，此时历史里已有 2 条工具结果，token 超标 ────
r2 = agent.invoke({"messages": [HumanMessage("再帮我搜一下: A股行情")]},
                  config=config)
print("第 2 轮回答:", r2["messages"][-1].content[:60], "...\n")

# ── 检查 state 里的所有 ToolMessage ─────────────────────────────
state_msgs = agent.get_state(config).values["messages"]
tool_msgs = [m for m in state_msgs if type(m).__name__ == "ToolMessage"]
print("=" * 62)
print(f"state 里共有 {len(tool_msgs)} 条 ToolMessage：")
for i, m in enumerate(tool_msgs, 1):
    tag = "已清理" if str(m.content).startswith("[已清空") else "保留原文"
    print(f"  工具结果{i} [{tag}] -> {str(m.content)[:40]}")
