"""
10 - 演示 ToolStrategy 的 handle_errors 传「异常元组」：选择性捕获错误

对应课程截图写法：
    agent = create_agent(
        model=model,
        response_format=ToolStrategy(
            Union[ContactInfo, EventDetails],
            tool_message_content="提取完成!",
            handle_errors=(MultipleStructuredOutputsError, StructuredOutputValidationError),
        ),
    )

handle_errors 的 5 种取值（源码 langchain/agents/factory.py 542-558 行）：
    True   -> 捕获所有错误，用默认模板回给模型让它重试（默认值）
    str    -> 捕获所有错误，用你给的自定义文字让模型重试
    异常类  -> 只捕获这一类异常（含子类），其他异常直接抛出
    异常元组 -> 只捕获元组中任一类型的异常，其他异常直接抛出  <- 本文件演示
    回调函数 -> 调用你的函数，把返回值作为错误消息让模型重试
    False  -> 不捕获，异常直接抛出（agent 崩掉）
"""
import os
from dotenv import load_dotenv
from typing import Union
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.structured_output import (
    ToolStrategy,
    MultipleStructuredOutputsError,      # 模型一次调了多个结构化输出工具时抛出
    StructuredOutputValidationError,     # 工具参数不符合 schema 时抛出
)

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


# ---------- 1. 两个异常类的继承关系 ----------
print("=== 1) 两个异常类的继承链 ===")
print("MultipleStructuredOutputsError <-",
      " <- ".join(c.__name__ for c in MultipleStructuredOutputsError.__mro__[:3][::-1]))
print("StructuredOutputValidationError <-",
      " <- ".join(c.__name__ for c in StructuredOutputValidationError.__mro__[:3][::-1]))

# ---------- 2. 截图写法构建 ----------
print("\n=== 2) handle_errors 传异常元组（截图写法）===")
ts = ToolStrategy(
    Union[ContactInfo, EventDetails],
    tool_message_content="提取完成!",
    handle_errors=(MultipleStructuredOutputsError, StructuredOutputValidationError),
)
print("构建成功，只捕获这两种异常，其他异常直接抛出")

# ---------- 3. 单元素「元组」的隐蔽坑 ----------
print("\n=== 3) 单元素元组的坑 ===")
a = (StructuredOutputValidationError)    # 没逗号 -> 不是元组！就是类本身
b = (StructuredOutputValidationError,)   # 有逗号 -> 才是单元素元组
print("(X)  的类型:", type(a).__name__, "（等于直接传类，效果碰巧相同但不严谨）")
print("(X,) 的类型:", type(b).__name__, "（真元组，规范写法）")

# ---------- 4. 真实跑一次 DeepSeek ----------
print("\n=== 4) 用截图写法真实跑 DeepSeek ===")
model = init_chat_model(
    model="openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)
agent = create_agent(
    model=model,
    response_format=ToolStrategy(
        Union[ContactInfo, EventDetails],
        tool_message_content="提取完成!",
        handle_errors=(MultipleStructuredOutputsError, StructuredOutputValidationError),
    ),
)
r = agent.invoke({"messages": [("user", "新客户张三，邮箱 zhang@demo.com，手机 13800138000")]})
print("structured_response:", r["structured_response"])
print("其类型:", type(r["structured_response"]).__name__)
print("ToolMessage content:",
      [m.content for m in r["messages"] if type(m).__name__ == "ToolMessage"])
