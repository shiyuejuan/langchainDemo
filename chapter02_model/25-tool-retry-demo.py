"""
25 - ToolRetryMiddleware 演示（工具重试中间件，LangChain 1.x + DeepSeek）

课程片段：5、ToolRetryMiddleware 中间件 —— 工具失败自动重试。

源码确认（lc 1.2.12）签名：
  ToolRetryMiddleware(*,
      max_retries=2,          # 最多重试几次（首次 + 重试 = 共试 1+2=3 次）
      tools=None,             # 只对哪些工具生效（None = 所有工具）
      retry_on=(Exception,),  # 遇到哪些异常才重试（元组 或 返回 bool 的函数）
      on_failure='continue',  # 重试全失败后：continue=把错误信息还给模型继续 / error=抛异常
      backoff_factor=2.0,     # 指数退避系数（第 n 次等待 = initial * factor^n）
      initial_delay=1.0,      # 第一次重试前等几秒
      max_delay=60.0,         # 单次等待上限
      jitter=True)            # 随机抖动，避免同一时刻集中重试

demo 把 delay 调到 0.1 秒以便快速跑完。三个场景：
  A: 工具前 2 次失败、第 3 次成功 → 重试机制救回来
  B: 一直失败 + on_failure='continue'（默认）→ 错误信息还给模型，agent 正常回答
  C: 一直失败 + on_failure='error' → 直接抛异常
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

call_counter = {"n": 0}


@tool
def flaky_api(city: str) -> str:
    """查询城市天气的接口（模拟：偶尔抽风，前几次调用会失败）。"""
    call_counter["n"] += 1
    if call_counter["n"] <= 2:
        raise ConnectionError(f"第 {call_counter['n']} 次调用：接口超时!")
    return f"{city}：晴，22 度"


@tool
def broken_api(x: str) -> str:
    """一个永远坏的接口（模拟：无论怎么重试都失败）。"""
    raise RuntimeError("服务彻底挂了，别救了")


def run(title, tool_, middleware, question):
    print("=" * 62)
    print(title)
    print("=" * 62)
    agent = create_agent(model=deepseek, tools=[tool_], middleware=middleware)
    try:
        result = agent.invoke({"messages": [HumanMessage(question)]})
        print("✅ agent 正常结束，最终回答:", result["messages"][-1].content[:80])
    except Exception as e:
        print("❌ 抛出异常:", type(e).__name__, "-", e)
    print()


# ── 场景 A：坏 2 次后第 3 次成功，max_retries=2 刚好够救 ───────────
call_counter["n"] = 0
run("场景 A：前 2 次失败、第 3 次成功 → 重试救回来（共试 3 次）",
    flaky_api,
    [ToolRetryMiddleware(max_retries=2, initial_delay=0.1,
                         backoff_factor=1.0, jitter=False)],
    "北京今天天气怎么样？")

# ── 场景 B：一直失败 + on_failure='continue'（默认）───────────────
run("场景 B：永远失败 + on_failure='continue' → 错误还给模型继续",
    broken_api,
    [ToolRetryMiddleware(max_retries=1, initial_delay=0.1,
                         backoff_factor=1.0, jitter=False)],
    "帮我查一下 x 的数据")

# ── 场景 C：一直失败 + on_failure='error' ─────────────────────────
run("场景 C：永远失败 + on_failure='error' → 直接抛异常",
    broken_api,
    [ToolRetryMiddleware(max_retries=1, initial_delay=0.1,
                         backoff_factor=1.0, jitter=False,
                         on_failure="error")],
    "帮我查一下 x 的数据")
