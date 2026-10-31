"""
31 - rprint 打印 request 的优化格式（LangChain 1.x 调试工具箱）

目标：解决 30 的问题——state/runtime 全展开刷屏。
做法：
  ① 手写 view_* 函数，只挑出调试有用的字段，state/runtime 折叠成摘要
  ② rprint 彩色美化这个"调试视图"字典
  ③ 用 rich.Table 把 messages 列成表格，比嵌套 dict 直观得多

关键字段取舍（中间件调试最常看的就是这几个）：
  model / system_message / tool_choice / tools / messages
  state 只取消息条数，runtime 只取 thread_id / run_id
"""
import os
from dataclasses import fields
from typing import Any

from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents.middleware import ModelRequest, ToolCallRequest

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)
console = Console(width=110)


def view_model_request(req: ModelRequest) -> dict:
    """把 ModelRequest 重组成"调试视图"——只留关键字段，巨型字段折叠。"""
    tool_names = [t.name for t in req.tools if hasattr(t, "name")] if req.tools else []
    state_count = len(req.state.get("messages", [])) if isinstance(req.state, dict) else "?"
    thread_id = (req.runtime.execution_info.thread_id
                 if req.runtime and req.runtime.execution_info else None)
    return {
        "model": type(req.model).__name__,
        "system_message": (req.system_message.content[:40]
                           if req.system_message else None),
        "tool_choice": req.tool_choice,
        "tools": tool_names,
        "messages": [f"{type(m).__name__}: {str(getattr(m, 'content', ''))[:50]}"
                     for m in req.messages],
        "state.messages_count": state_count,
        "runtime.thread_id": thread_id,
    }


def view_tool_request(req: ToolCallRequest) -> dict:
    """把 ToolCallRequest 重组成调试视图。"""
    tc = req.tool_call
    return {
        "tool": req.tool.name if req.tool else None,
        "name": tc.get("name"),
        "args": tc.get("args"),
        "id": tc.get("id"),
    }


def messages_table(req: Any) -> Table:
    """把 messages 列成彩色表格。"""
    table = Table(title="Messages（本轮发给模型的）", show_header=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("type", style="magenta", width=14)
    table.add_column("content", style="green")
    for i, m in enumerate(req.messages):
        table.add_row(str(i), type(m).__name__, str(getattr(m, "content", ""))[:60])
    return table


class OptimizedRequestMiddleware(AgentMiddleware):
    """优化版打印中间件。"""

    def wrap_model_call(self, request: ModelRequest, handler):
        console.rule("[bold cyan]ModelRequest（调试视图）")
        rprint(view_model_request(request))
        console.print(messages_table(request))
        return handler(request)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        console.rule("[bold magenta]ToolCallRequest（调试视图）")
        rprint(view_tool_request(request))
        return handler(request)


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}：晴，22 度"


if __name__ == "__main__":
    deepseek = init_chat_model(
        "openai:deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
    )
    agent = create_agent(
        model=deepseek,
        tools=[get_weather],
        middleware=[OptimizedRequestMiddleware()],
    )
    agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
