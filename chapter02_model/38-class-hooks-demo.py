"""
38 - 基于类实现钩子：两条"固定"规则验证（课程 5.2/2）

规则1：必须继承 AgentMiddleware（框架认类型）
规则2：方法名固定 before_model / after_model / ... （框架认名字的插座）

demo 三连：
  ① 一类挂满全部 6 个钩子，跑一次看完整生命周期时间线
  ② 方法名拼错（before_models）→ 不报错但永不触发（隐形坑！）
  ③ 不继承 AgentMiddleware 的普通类 → create_agent 直接拒绝
"""
import os
from dotenv import load_dotenv

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


# ── ① 挂满全部 6 个钩子的类（方法名全部固定）──────────────────
class LifecycleMW(AgentMiddleware):
    def before_agent(self, state, runtime):
        print("  [before_agent]    整个运行开始（1 次）")

    def before_model(self, state, runtime):
        print("  [before_model]    每次调模型前")

    def wrap_model_call(self, request, handler):
        print("  [wrap_model_call] 包住模型调用 → 进")
        result = handler(request)
        print("  [wrap_model_call] 包住模型调用 ← 出")
        return result

    def wrap_tool_call(self, request, handler):
        print("  [wrap_tool_call]  包住工具调用 → 进")
        result = handler(request)
        print("  [wrap_tool_call]  包住工具调用 ← 出")
        return result

    def after_model(self, state, runtime):
        print("  [after_model]     每次调模型后")

    def after_agent(self, state, runtime):
        print("  [after_agent]     整个运行结束（1 次）")

    def before_models(self, state, runtime):   # ← 故意拼错！
        print("  [before_models]  这个永远不会被打印")


deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


def rule(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


rule("① 挂满 6 钩子跑一次（注意 before_models 拼错了，永远不出现）")
agent = create_agent(model=deepseek, tools=[get_weather],
                     middleware=[LifecycleMW()])
agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})


rule("③ 违反规则1：不继承 AgentMiddleware 会怎样")
class FakeMW:                       # 故意不继承
    def before_model(self, state, runtime):
        print("  不会被调用")

try:
    create_agent(model=deepseek, tools=[get_weather], middleware=[FakeMW()])
except Exception as e:
    print(f"  create_agent 直接拒绝：{type(e).__name__}: {e}")
