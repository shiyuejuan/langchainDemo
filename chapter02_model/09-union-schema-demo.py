"""
09 - 演示 ToolStrategy 配合 Union：让模型从多种结构里「选一种」返回

Union[ContactInfo, EventDetails] 的含义：
  模型根据输入内容，自动决定调用 ContactInfo 隐藏工具 还是 EventDetails 隐藏工具，
  返回对应结构的数据。

实测结论（DeepSeek 真跑）：
  ToolStrategy 会把 Union 拆成「多个隐藏工具」，每个 schema 一个，模型二选一。
  - 客户信息文本 -> 调用 ContactInfo 工具 -> 返回 name/email/phone
  - 活动信息文本 -> 调用 EventDetails 工具 -> 返回 event_name/date/location
  tool_message_content 在多个工具间共用（两条 ToolMessage 都是 "提取完成!"）。
"""
import os
from dotenv import load_dotenv
from typing import Union
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)


class ContactInfo(BaseModel):
    """用户的联系方式"""
    name: str = Field(description="用户姓名")
    email: str = Field(description="用户邮箱地址")
    phone: str = Field(description="用户的手机号")


class EventDetails(BaseModel):
    """活动详情"""
    event_name: str = Field(description="活动名称")
    date: str = Field(description="活动日期")
    location: str = Field(description="活动地点")


# ---------- 1. 构建（不联网），看 Union 被拆成几个隐藏工具 ----------
ts = ToolStrategy(
    Union[ContactInfo, EventDetails],
    tool_message_content="提取完成!",
    handle_errors=True,
)
print("=== 构建结果 ===")
print("schema_specs 数量（= 隐藏工具数）:", len(ts.schema_specs))
for s in ts.schema_specs:
    print("  隐藏工具:", s.name, "::", s.description)

# ---------- 2. 初始化模型 + 建 agent ----------
model = init_chat_model(
    model="openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)
agent = create_agent(model=model, response_format=ts)


def run(title: str, text: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    r = agent.invoke({"messages": [("user", text)]})
    print("structured_response:", r["structured_response"])
    print("其类型:", type(r["structured_response"]).__name__)
    print("ToolMessage content:",
          [m.content for m in r["messages"] if type(m).__name__ == "ToolMessage"])
    # 看模型到底调了哪个隐藏工具
    for m in r["messages"]:
        if type(m).__name__ == "AIMessage" and getattr(m, "tool_calls", None):
            print("模型调用的工具:", [tc["name"] for tc in m.tool_calls])


run("客户信息输入 -> 期望 ContactInfo",
    "新客户张三，邮箱 zhang@demo.com，手机 13800138000")
run("活动信息输入 -> 期望 EventDetails",
    "公司年会，时间2026年12月20日，地点上海国际会议中心")
