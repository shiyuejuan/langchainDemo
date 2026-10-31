"""
07 - 演示 ToolStrategy 的 tool_message_content 到底是什么

背景回顾：
- ToolStrategy(ContactInfo) 把 ContactInfo 伪装成一个“隐藏工具”绑给模型；
- 模型调用这个工具填好 name/email/phone 后，框架必须回一条 ToolMessage 给模型
  （和你在 06 天气例子里自己手写 ToolMessage 是同一套机制）；
- 这条由框架自动生成的 ToolMessage 里的“文字内容”，就是 tool_message_content 的值。

本文件做两件事：
1) 传 tool_message_content="已成功抽取信息"，看 ToolMessage 内容变成啥；
2) 不传（默认 None），看框架默认的 ToolMessage 内容又是啥。
"""
import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

# 显式指定 .env 路径（之前实测过，不传路径在部分启动方式下会失败）
load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

# ---------- 1. 定义结构化 Schema ----------
class ContactInfo(BaseModel):
    """用户的联系方式"""
    name: str = Field(description="用户姓名")
    email: str = Field(description="用户邮箱地址")
    phone: str = Field(description="用户的手机号")

# ---------- 2. 初始化 DeepSeek 模型 ----------
model = init_chat_model(
    model="openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

USER_TEXT = "新客户张三，邮箱 zhang@demo.com，手机 13800138000"


def show_result(title: str, agent) -> None:
    """调用 agent 并打印：结构化数据 + 每条消息（重点标出 ToolMessage 内容）"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    result = agent.invoke({"messages": [("user", USER_TEXT)]})

    # 真正抽到的结构化数据（来自模型“调用工具”时传的参数）
    print("\n>>> structured_response（你要的姓名/邮箱/手机）:")
    print(result["structured_response"])

    # 完整消息流，重点看那条 ToolMessage 的文字
    print("\n>>> messages 消息流（看 ToolMessage 的 content）:")
    for i, m in enumerate(result["messages"]):
        t = type(m).__name__
        if t == "ToolMessage":
            # 这一行的 content，就是 tool_message_content 的值！
            print(f"  [{i}] {t}  -> content = {m.content!r}")
        elif hasattr(m, "content"):
            content = m.content
            if isinstance(content, str) and len(content) > 60:
                content = content[:60] + "...(略)"
            print(f"  [{i}] {t}  -> content = {content!r}")
        else:
            print(f"  [{i}] {t}")

    # ---------- 打印完整 response（result 的全部内容，不省略） ----------
    print("\n>>> 完整 response（result 的所有字段与消息，不省略）:")
    print("  result 的 key:", list(result.keys()))
    for i, m in enumerate(result["messages"]):
        t = type(m).__name__
        info: dict = {"type": t}
        if getattr(m, "id", None):
            info["id"] = m.id
        if hasattr(m, "content"):
            info["content"] = m.content
        if getattr(m, "tool_calls", None):
            info["tool_calls"] = m.tool_calls
        if getattr(m, "tool_call_id", None):
            info["tool_call_id"] = m.tool_call_id
        if getattr(m, "name", None):
            info["name"] = m.name
        print(f"  [{i}] {info}")
    print("  structured_response:", result.get("structured_response"))


# ---------- 3. 情况 A：传 tool_message_content ----------
agent_with_msg = create_agent(
    model=model,
    response_format=ToolStrategy(ContactInfo, tool_message_content="已成功抽取信息"),
)
show_result("情况 A：ToolStrategy(ContactInfo, tool_message_content='已成功抽取信息')", agent_with_msg)

# ---------- 4. 情况 B：不传（默认 None），看框架默认内容 ----------
agent_default = create_agent(
    model=model,
    response_format=ToolStrategy(ContactInfo),
)
show_result("情况 B：ToolStrategy(ContactInfo) 不传 tool_message_content（默认 None）", agent_default)

print("\n" + "=" * 70)
print("结论：")
print("  - 那条 ToolMessage 的 content = tool_message_content 的值（情况 A 是 '已成功抽取信息'）")
print("  - 不传时框架用默认文字（情况 B 显示的内容）")
print("  - 真正的姓名/邮箱/手机在 structured_response，不在 ToolMessage 文字里")
print("=" * 70)
