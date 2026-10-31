"""
32 - rprint 直接打印 request，得到截图那种"总格式"（LangChain 1.x）

截图格式特征（rich Pretty 对 dataclass/pydantic 的默认渲染）：
    ModelRequest(
        model=ChatOpenAI(...),
        messages=[HumanMessage(...), ...],
        ...
    )
  · 顶层是"类名("，字段名=值，逐层缩进
  · dict 部分仍是花括号 {...}（如 Interrupt 的 value={...}）
  · 字符串带颜色

做法：不再重组 dict，直接把 request 对象交给 rprint。
代价：state/runtime 会全展开，输出长——想要精简就用 31 的 view_* 调试视图。
"""
import os
from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)
console = Console(width=110)


class FullPrettyMiddleware(AgentMiddleware):
    """直接 rprint(request) —— 截图那种类名(字段=值)树状总格式。"""

    def wrap_model_call(self, request, handler):
        console.rule("[bold cyan]ModelRequest 完整 Pretty 格式")
        rprint(request)
        return handler(request)

    def wrap_tool_call(self, request, handler):
        console.rule("[bold magenta]ToolCallRequest 完整 Pretty 格式")
        rprint(request)
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
        middleware=[FullPrettyMiddleware()],
    )
    agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
