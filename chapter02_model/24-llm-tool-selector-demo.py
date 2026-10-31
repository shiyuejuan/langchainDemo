"""
24 - LLMToolSelectorMiddleware 演示（智能工具筛选中间件，LangChain 1.x + DeepSeek）

课程片段：当工具太多时，用子模型筛选最相关的几个工具。
  · model           —— 用于筛选的子模型（不传 = 用主模型自己筛）
  · max_tools       —— 最多保留几个工具（超出截断，按相关性排序取前 N）
  · always_include  —— 指定的工具永远保留，且【不占 max_tools 名额】

源码确认（lc 1.2.12，tool_selection.py）：
  · 工作时机：wrap_model_call —— 在主模型收到请求【之前】先做一次筛选
  · 筛选流程：取最后一条用户消息 + 工具名清单 → 让筛选模型返回相关工具名
    → 校验（选了不存在的工具名会抛 ValueError）→ 按相关性排序截断到 max_tools
    → 把 always_include 的工具追加回去 → request.override(tools=筛选后的)
  · 注意：筛选本身要额外花一次 LLM 调用！

验证手段：再加一个 PrintToolsMiddleware 打印主模型实际看到的工具列表。
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware
from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Any
from langchain_core.messages import HumanMessage
from langchain.agents.middleware import AgentMiddleware

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


# ── 8 个假工具（不会真执行核心逻辑，只为撑满工具列表）──────────────
@tool
def get_weather(city: str) -> str:
    """查询某个城市今天的天气。"""
    return f"{city}：晴，22 度"

@tool
def get_current_time() -> str:
    """获取当前的日期和时间。"""
    return "2026-10-24 20:30"

@tool
def search_web(query: str) -> str:
    """联网搜索一段文字。"""
    return f"关于 {query} 的搜索结果……"

@tool
def send_email(to: str, subject: str) -> str:
    """给某人发送一封邮件。"""
    return f"已发送给 {to}：{subject}"

@tool
def read_file(path: str) -> str:
    """读取本地文件内容。"""
    return f"{path} 的内容……"

@tool
def write_file(path: str, content: str) -> str:
    """把内容写入本地文件。"""
    return f"已写入 {path}"

@tool
def calculate(expression: str) -> str:
    """计算一个数学表达式的值。"""
    return "42"

@tool
def play_music(song: str) -> str:
    """播放一首歌。"""
    return f"正在播放 {song}"

ALL_TOOLS = [get_weather, get_current_time, search_web, send_email,
             read_file, write_file, calculate, play_music]


class PrintToolsMiddleware(AgentMiddleware):
    """打印主模型实际收到的工具列表——用来'偷看'筛选结果。"""

    def __init__(self, label: str = ""):
        super().__init__()
        self.label = label

    def wrap_model_call(self, request, handler):
        names = [t.name for t in request.tools]
        print(f"   [偷看{self.label}] 主模型收到 {len(names)} 个工具: {names}")
        return handler(request)


class FCSelectable(BaseChatModel):
    """给 DeepSeek 打补丁：结构化输出强制走 function_calling。

    背景：中间件内部用 with_structured_output(json_schema) 让筛选模型
    返回工具清单，但 DeepSeek 中转端点不支持该 response_format，报 400：
    'This response_format type is unavailable now'。
    实测 method="function_calling" 可以正常返回 → 这里包一层强制指定。
    """
    inner: Any

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self.inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return self.inner._llm_type

    def with_structured_output(self, schema, **kwargs):
        return self.inner.with_structured_output(schema, method="function_calling", **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self.inner.bind_tools(tools, **kwargs)


selector_model = FCSelectable(inner=deepseek)


# ── 场景 1：8 个工具，max_tools=3，用主模型自己当筛选员 ─────────────
print("=" * 62)
print("场景 1：问天气，max_tools=3（筛选员 = 主模型 DeepSeek）")
print("=" * 62)
agent1 = create_agent(
    model=deepseek,
    tools=ALL_TOOLS,
    middleware=[
        LLMToolSelectorMiddleware(model=selector_model, max_tools=3),
        PrintToolsMiddleware(label="筛选后"),
    ],
)
r1 = agent1.invoke({"messages": [HumanMessage("北京今天天气怎么样？")]})
print("最终回答:", r1["messages"][-1].content[:100])
print()

# ── 场景 2：always_include 不占 max_tools 名额 ────────────────────
print("=" * 62)
print("场景 2：max_tools=2 + always_include=[play_music]")
print("（play_music 即使没被选中也强制在场，且不占 2 个名额）")
print("=" * 62)
agent2 = create_agent(
    model=deepseek,
    tools=ALL_TOOLS,
    middleware=[
        LLMToolSelectorMiddleware(model=selector_model, max_tools=2,
                                  always_include=["play_music"]),
        PrintToolsMiddleware(label="筛选后"),
    ],
)
r2 = agent2.invoke({"messages": [HumanMessage("帮我放首歌听听")]})
print("最终回答:", r2["messages"][-1].content[:100])
