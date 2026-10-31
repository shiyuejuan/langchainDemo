"""
26 - ModelRetryMiddleware 演示（模型重试中间件，LangChain 1.x + DeepSeek）

课程片段：6、ModelRetryMiddleware 中间件 —— 模型调用失败自动重试。

源码确认（lc 1.2.12）：
  · 签名和 ToolRetryMiddleware 几乎一样，只少了 tools 参数（它天然管的就是模型）：
    (*, max_retries=2, retry_on=(Exception,), on_failure='continue',
     backoff_factor=2.0, initial_delay=1.0, max_delay=60.0, jitter=True)
  · retry_on 不匹配的异常 → 立即失败，不浪费重试次数
  · on_failure='continue' 的特殊点：模型挂了没法"把错误还给模型"，
    所以源码直接造一条说明错误的 AIMessage 当最终回答（工具版是 ToolMessage 还给模型继续）

demo 用上次学过的"包装类"套路：FlakyModel 包住真 deepseek，
前 N 次调用抛异常模拟故障，之后正常委托。
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from typing import Any

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

state = {"n": 0}


class FlakyModel(BaseChatModel):
    """前 fail_times 次调用必失败，之后正常委托给真模型（复用 FCSelectable 的包装套路）。"""
    inner: Any
    fail_times: int

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        state["n"] += 1
        if state["n"] <= self.fail_times:
            raise ConnectionError(f"模拟故障：第 {state['n']} 次模型调用失败")
        return self.inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "flaky-" + self.inner._llm_type


def run(title, fail_times, middleware):
    print("=" * 62)
    print(title)
    print("=" * 62)
    state["n"] = 0
    agent = create_agent(
        model=FlakyModel(inner=deepseek, fail_times=fail_times),
        middleware=middleware,
    )
    try:
        result = agent.invoke({"messages": [HumanMessage("用一句话介绍你自己")]})
        last = result["messages"][-1]
        print("✅ agent 正常结束")
        print("   最终消息类型:", type(last).__name__)
        print("   内容:", str(last.content)[:90])
    except Exception as e:
        print("❌ 抛出异常:", type(e).__name__, "-", e)
    print()


FAST = dict(initial_delay=0.1, backoff_factor=1.0, jitter=False)

# ── 场景 A：坏 2 次后第 3 次成功，max_retries=2 刚好救回 ───────────
run("场景 A：前 2 次失败、第 3 次成功 → 重试救回",
    fail_times=2,
    middleware=[ModelRetryMiddleware(max_retries=2, **FAST)])

# ── 场景 B：永远失败 + 默认 on_failure='continue' ─────────────────
run("场景 B：永远失败 + continue → 不抛异常，返回一条错误说明的 AIMessage",
    fail_times=999,
    middleware=[ModelRetryMiddleware(max_retries=1, **FAST)])

# ── 场景 C：永远失败 + on_failure='error' ─────────────────────────
run("场景 C：永远失败 + error → 直接抛异常",
    fail_times=999,
    middleware=[ModelRetryMiddleware(max_retries=1, **FAST, on_failure="error")])
