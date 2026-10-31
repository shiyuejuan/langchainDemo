"""
23 - ModelFallbackMiddleware 演示（模型故障转移中间件，LangChain 1.x + DeepSeek）

课程片段：ModelFallbackMiddleware —— 当主模型无法访问时，启用备用模型。

源码确认（langchain 1.2.12）：
  · 签名：ModelFallbackMiddleware(first_model, *additional_models)
    —— 传一串"备用模型"，可以是 "openai:xxx" 字符串（内部 init_chat_model 初始化）
       或 BaseChatModel 实例
  · 工作原理（wrap_model_call）：
      1) 先调用 create_agent 里的主模型
      2) 抛异常 → 按顺序逐个尝试备用模型（request.override(model=备用) 换模型重试）
      3) 某个备用成功 → 直接返回它的回答
      4) 全部失败 → 抛"最后一个"异常
  · 注意：主模型不是在这里配的！主模型仍是 create_agent(model=...) 里的那个，
    这个中间件里配的全是"替补队员"。

为了不烧钱也能演示"主模型挂掉"，这里写一个 FailingChatModel（一调用就抛异常的假模型），
替补队员里放真 DeepSeek。
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


class FailingChatModel(BaseChatModel):
    """一调用就抛异常的假模型——用来冒充'挂掉的主模型'。"""
    model_name: str = "always-broken"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise ConnectionError(f"[模拟故障] {self.model_name} 无法访问!")

    @property
    def _llm_type(self) -> str:
        return "failing"


def show(title, agent, question="用一句话介绍你自己"):
    print("=" * 60)
    print(title)
    print("=" * 60)
    try:
        result = agent.invoke({"messages": [HumanMessage(question)]})
        reply = result["messages"][-1]
        print("✅ 拿到了最终回答：", reply.content[:80])
        # response_metadata 里能看出这条回答到底是谁生成的
        meta = reply.response_metadata
        print("   实际生成回答的模型:", meta.get("model_name", meta.get("model", "未知(假模型)")))
    except Exception as e:
        print("❌ 整条链全挂，抛出异常:", type(e).__name__, "-", e)
    print()


# ── 场景 A：主模型挂 → 唯一替补(DeepSeek)顶上 ────────────────────
agent_a = create_agent(
    model=FailingChatModel(model_name="主模型"),
    middleware=[ModelFallbackMiddleware(deepseek)],
)
show("场景 A：主模型挂，1 个替补(DeepSeek) → 应该成功", agent_a)

# ── 场景 B：主模型挂 → 替补1也挂 → 替补2(DeepSeek)顶上 ──────────
agent_b = create_agent(
    model=FailingChatModel(model_name="主模型"),
    middleware=[
        ModelFallbackMiddleware(
            FailingChatModel(model_name="替补1"),
            deepseek,                       # 替补2：真模型
        )
    ],
)
show("场景 B：主模型挂 → 替补1挂 → 替补2(DeepSeek)顶上", agent_b)

# ── 场景 C：全员阵亡 → 抛最后一个异常 ──────────────────────────
agent_c = create_agent(
    model=FailingChatModel(model_name="主模型"),
    middleware=[
        ModelFallbackMiddleware(
            FailingChatModel(model_name="替补1"),
            FailingChatModel(model_name="替补2"),
        )
    ],
)
show("场景 C：主模型+2个替补全挂 → 抛最后那个异常", agent_c)
