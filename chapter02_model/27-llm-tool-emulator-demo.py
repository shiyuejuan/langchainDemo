"""
27 - LLMToolEmulator 演示（LLM 工具模拟器，LangChain 1.x + DeepSeek）

课程片段：7、LLMToolEmulator —— 工具尚未开发完成时，用 LLM 模拟工具，方便先测工具调用流程。

源码确认（lc 1.2.12）：
  · 签名：(*, tools=None, model=None)
      tools=None  → 默认模拟【所有】工具
      tools=[...] → 只模拟指定工具（名字字符串或 BaseTool 实例），其余照常真执行
      model       → 用哪个 LLM 来编造结果（默认 = 主模型）
  · 工作原理（wrap_tool_call）：每次工具调用被拦截 →
      该工具在模拟名单 → 构造提示词（工具名+描述+参数）让 LLM"编"一个该工具会返回的结果
      → 直接返回 ToolMessage（短路：真工具根本没执行！）
      不在名单 → 正常 handler 执行
  · 模拟提示词关键句："Generate a realistic response that this tool would return..."

demo 设计：
  query_database 真函数会抛 NotImplementedError（模拟"还没开发完"）
  get_time 真函数返回固定假时间 2099-01-01（用于验证模拟是否生效：
    若拿到 2099 → 真执行了；若拿到别的 → LLM 编的）
"""
import os
from datetime import datetime
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


@tool
def query_database(sql: str) -> str:
    """查询订单数据库并返回结果。输入是一条 SQL 查询语句。"""
    raise NotImplementedError("真实数据库还没接上!")


@tool
def get_time() -> str:
    """获取当前的日期和时间。"""
    return datetime(2099, 1, 1).isoformat()   # 故意的假时间，用于验证


def run(title, middleware):
    print("=" * 62)
    print(title)
    print("=" * 62)
    agent = create_agent(model=deepseek, tools=[query_database, get_time],
                         middleware=middleware)
    try:
        r = agent.invoke({"messages": [HumanMessage("帮我查一下销量最高的 3 个订单，顺便告诉我现在几点")]})
        print("✅ 最终回答:", r["messages"][-1].content[:150])
    except Exception as e:
        print("❌ 出错:", type(e).__name__, "-", str(e)[:100])
    print()


# ── 对照组：没有模拟器 → 未完成的工具真的会被执行 → 报错 ──────────
run("对照组：不加模拟器，未完成的 query_database 真执行 → 出错",
    middleware=[])

# ── 场景 A：只模拟 query_database（get_time 照常真执行）────────────
run("场景 A：模拟器只管 query_database → 编出假订单数据；"
    "get_time 真执行 → 时间应是 2099-01-01",
    middleware=[LLMToolEmulator(tools=["query_database"], model=deepseek)])

# ── 场景 B：tools=None 模拟所有工具 → 时间也该是 LLM 编的 ──────────
run("场景 B：模拟所有工具 → 时间不应该是 2099（是 LLM 编的）",
    middleware=[LLMToolEmulator(model=deepseek)])
