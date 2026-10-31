"""
29 - 中间件里 request 对象的格式化输出（LangChain 1.x）

课程里 wrap_model_call(request, ...) / wrap_tool_call(request, ...) 的 request 到底长啥样，
怎么漂亮地打印出来看？这是给小白调试中间件时的"偷看"工具箱。

源码确认（lc 1.2.12）：
  ModelRequest 字段: model, messages, system_message, system_prompt, tool_choice,
                     tools, response_format, state, runtime, model_settings
  ToolCallRequest 字段: tool_call, tool, state, runtime
  request.override(...) 返回改了某些字段的新 request（不可变更新，原对象不变）

本文件给出 3 种"格式化输出 request"的姿势 + 一个实战打印中间件。
"""
import os
from dataclasses import asdict
from pprint import pformat

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents.middleware import ModelRequest, ToolCallRequest


# ───── 姿势 1：手写"安全裁剪"函数（最可控，推荐）──────────────────
def fmt_messages(messages, limit=40):
    """把消息列表压成可读摘要：类型 + 截断内容。"""
    out = []
    for m in messages:
        name = type(m).__name__
        content = str(getattr(m, "content", ""))[:limit].replace("\n", " ")
        out.append(f"{name}: {content}")
    return out


def fmt_model_request(req: ModelRequest) -> str:
    """把 ModelRequest 格式化成一段易读文本。"""
    tool_names = [t.name for t in req.tools if hasattr(t, "name")] if req.tools else []
    return (
        f"ModelRequest(\n"
        f"  model           = {type(req.model).__name__}\n"
        f"  system_message  = {str(req.system_message.content if req.system_message else None)[:40]}\n"
        f"  tool_choice     = {req.tool_choice}\n"
        f"  tools           = {tool_names}\n"
        f"  response_format = {req.response_format}\n"
        f"  state           = {type(req.state).__name__ if req.state else None}\n"
        f"  messages({len(req.messages)}):\n"
        + "\n".join(f"    {x}" for x in fmt_messages(req.messages))
        + "\n)"
    )


def fmt_tool_request(req: ToolCallRequest) -> str:
    """把 ToolCallRequest 格式化成一段易读文本。"""
    tc = req.tool_call
    return (
        f"ToolCallRequest(\n"
        f"  tool            = {req.tool.name if req.tool else None}\n"
        f"  tool_call.name  = {tc.get('name')}\n"
        f"  tool_call.args  = {tc.get('args')}\n"
        f"  tool_call.id    = {tc.get('id')}\n"
        f"  state           = {type(req.state).__name__}\n"
        f")"
    )


# ───── 姿势 2：用 dataclasses.asdict + pformat（全自动，但 messages 会很长）──
def fmt_asdict(req) -> str:
    d = asdict(req)
    # asdict 会把消息里的大对象全展开，messages 先裁剪
    if "messages" in d:
        d["messages"] = fmt_messages(d.get("messages", []), limit=20)
    return pformat(d, width=80, depth=2)


# ───── 实战：两个打印中间件，挂在真实 agent 上实跑 ─────────────────
class PrintModelRequestMiddleware(AgentMiddleware):
    """在 wrap_model_call 里用姿势 1 打印 ModelRequest。"""

    def wrap_model_call(self, request: ModelRequest, handler):
        print("######## ModelRequest（模型调用前）########")
        print(fmt_model_request(request))
        return handler(request)


class PrintToolRequestMiddleware(AgentMiddleware):
    """在 wrap_tool_call 里用姿势 2 打印 ToolCallRequest。"""

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        print("######## ToolCallRequest（工具调用前）########")
        print(fmt_tool_request(request))
        return handler(request)


if __name__ == "__main__":
    load_dotenv = __import__("dotenv").load_dotenv
    load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)
    import dotenv
    deepseek = init_chat_model(
        "openai:deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
    )

    @tool
    def get_weather(city: str) -> str:
        """查询城市天气。"""
        return f"{city}：晴，22 度"

    agent = create_agent(
        model=deepseek,
        tools=[get_weather],
        middleware=[PrintModelRequestMiddleware(), PrintToolRequestMiddleware()],
    )
    agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
