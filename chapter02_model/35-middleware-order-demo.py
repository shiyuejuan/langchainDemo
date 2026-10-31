"""
35 - 多个中间件组合及执行顺序（LangChain 1.x + DeepSeek）

课程片段：多个中间件组合及执行顺序

源码确认（lc 1.2.12）：
  · AgentMiddleware 钩子家族：
      生命周期钩子（顺序执行，不是嵌套）：before_agent / after_agent
                                        before_model / after_model
      包裹钩子（洋葱嵌套）：wrap_model_call / wrap_tool_call
  · middleware=[A, B, C] 的顺序规则：
      wrap 型：A 最外层 → C 最内层（贴着模型/工具）
               执行次序 A before → B before → C before → 真调用
                       → C after → B after → A after（洋葱）
      钩子型：before_model 按列表顺序 A→B→C
              after_model 反序 C→B→A —— 同样是洋葱语义！

demo 验证：
  ① 3 个 wrap 中间件 A/B/C，看模型调用和工具调用的进出顺序
  ② 3 个钩子型中间件 before_model/after_model，看是否按列表顺序
"""
import os
from dotenv import load_dotenv
from rich import print as rprint

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}：晴，22 度"


class WrapTrace(AgentMiddleware):
    """wrap 型示踪中间件：进出各打一行。"""

    def __init__(self, tag: str):
        super().__init__()
        self.tag = tag

    @property
    def name(self) -> str:          # create_agent 用 name 判重，必须唯一
        return f"WrapTrace-{self.tag}"

    def wrap_model_call(self, request, handler):
        print(f"  [{self.tag}] model before")
        result = handler(request)
        print(f"  [{self.tag}] model after")
        return result

    def wrap_tool_call(self, request, handler):
        print(f"  [{self.tag}] tool before")
        result = handler(request)
        print(f"  [{self.tag}] tool after")
        return result


class HookTrace(AgentMiddleware):
    """钩子型示踪中间件：before_model/after_model 各打一行。"""

    def __init__(self, tag: str):
        super().__init__()
        self.tag = tag

    @property
    def name(self) -> str:
        return f"HookTrace-{self.tag}"

    def before_model(self, state, runtime):
        print(f"  [{self.tag}] before_model")

    def after_model(self, state, runtime):
        print(f"  [{self.tag}] after_model")


deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


def rule(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


# ── ① wrap 型：洋葱嵌套 ──────────────────────────────────────
rule("① wrap 型（A, B, C 依次入列）—— 观察进/出顺序")
agent1 = create_agent(model=deepseek, tools=[get_weather],
                      middleware=[WrapTrace("A"), WrapTrace("B"), WrapTrace("C")])
agent1.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
print("\n（数一数：before 是 A→B→C，after 是 C→B→A —— 洋葱！）")


# ── ② 钩子型：顺序执行 ────────────────────────────────────────
rule("② 钩子型 before_model/after_model（A, B, C 依次入列）")
agent2 = create_agent(model=deepseek, tools=[get_weather],
                      middleware=[HookTrace("A"), HookTrace("B"), HookTrace("C")])
agent2.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
print("\n（before_model 按 A→B→C，after_model 反序 C→B→A —— 成对钩子也是洋葱语义）")
