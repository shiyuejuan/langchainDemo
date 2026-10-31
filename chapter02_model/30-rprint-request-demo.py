"""
30 - 用 rich 的 rprint 美化打印 request（LangChain 1.x 调试工具箱）

上一段讲了 request 格式化输出，这里升级成 rprint（rich 库）：
  相比裸 print，rprint 自动给嵌套结构上色、语法高亮、深层自动折叠、长度限制，
  特别适合看 ModelRequest / ToolCallRequest 这种带长 messages 的对象。

关键 API：
  from rich import print as rprint           # 直接替换 print
  from rich.pretty import pprint as rpprint   # 美化打印任意对象
  from rich.console import Console
  Console(max_string_length=60, max_depth=2, width=100)  # 防刷屏：字符串最长 60 字符、最深 2 层
  console.rule("标题")                        # 打印带标题的分隔线

实跑展示：同一个 agent，用 rprint 把每次请求的字段漂亮地打出来。
"""
import os
from dataclasses import fields

from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

# 宽 100 的控制台（max_depth 等限制放在 print 调用时传）
console = Console(width=100)

deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


def safe_dict(req) -> dict:
    """把 request 转成字典，但 messages 先裁短、model 换成类型名，避免刷屏。

    注意：不能图省事用 asdict(req) —— 它会递归展开 messages 里的 pydantic
    对象而报错；这里只取标量字段、手写裁剪消息列表。
    """
    d = {f.name: getattr(req, f.name) for f in fields(req)}
    if "messages" in d:
        d["messages"] = [
            f"{type(m).__name__}: {str(getattr(m, 'content', ''))[:40]}"
            for m in d["messages"]
        ]
    if "model" in d:
        d["model"] = type(d["model"]).__name__
    return d


class RichRequestMiddleware(AgentMiddleware):
    """用 rprint 美化打印 request（模型调用 + 工具调用）。"""

    def wrap_model_call(self, request, handler):
        console.rule("[bold red]ModelRequest（模型调用前）")
        console.print(request)
        return handler(request)

    def wrap_tool_call(self, request, handler):
        console.rule("[bold magenta]ToolCallRequest（工具调用前）")
        console.print(safe_dict(request))
        return handler(request)


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}：晴，22 度"


if __name__ == "__main__":
    agent = create_agent(
        model=deepseek,
        tools=[get_weather],
        middleware=[RichRequestMiddleware()],
    )
    agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
