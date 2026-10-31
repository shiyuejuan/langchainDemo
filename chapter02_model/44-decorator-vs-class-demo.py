"""
44 - 5.5 装饰器和类的选择（同一个中间件，两种写法对照）

演示同一需求「统计模型被调用了几次 + 打日志」：
  方式A 装饰器：没有 self，想持状态只能靠闭包"凑"——能跑但别扭
  方式B 类：  self 天然持状态——代码更直白

结论：
  · 单钩子 + 无状态 + 一次性逻辑 → 装饰器（轻量）
  · 要状态 / 多钩子共享 / 可复用组件 → 类
  · 装饰器本质 = 自动生成 AgentMiddleware 子类（语法糖）
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, wrap_model_call, before_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}：晴，22 度"


# ===== 方式A：装饰器版 —— 状态只能靠闭包"凑" =====
def make_decorator_mw(label: str):
    calls = {"n": 0}                      # 闭包变量冒充状态

    @wrap_model_call
    def counting_mw(request, handler):
        calls["n"] += 1
        print(f"  [{label}·装饰器] 第 {calls['n']} 次调模型（状态靠闭包凑）")
        return handler(request)

    return counting_mw                    # 返回的是中间件实例


# ===== 方式B：类版 —— self 天然持状态，还能多钩子共享 =====
class CountingMW(AgentMiddleware):
    def __init__(self, label: str):
        super().__init__()
        self.label = label
        self.calls = 0                    # self 就是状态
        self.model_starts = 0             # 供其他钩子共享

    @property
    def name(self):
        return f"CountingMW-{self.label}"

    def before_model(self, state, runtime):        # 多钩子共享 self
        self.model_starts += 1

    def wrap_model_call(self, request, handler):
        self.calls += 1
        print(f"  [{self.label}·类] 第 {self.calls} 次调模型"
              f"（节点进入 {self.model_starts} 次，self 共享）")
        return handler(request)


deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


if __name__ == "__main__":
    print("=" * 60)
    print("方式A 装饰器版（闭包凑状态）")
    print("=" * 60)
    a1 = create_agent(model=deepseek, tools=[get_weather],
                      middleware=[make_decorator_mw("A")])
    a1.invoke({"messages": [HumanMessage("北京天气怎么样？")]})

    print("\n" + "=" * 60)
    print("方式B 类版（self 持状态 + 多钩子共享）")
    print("=" * 60)
    a2 = create_agent(model=deepseek, tools=[get_weather],
                      middleware=[CountingMW("B")])
    a2.invoke({"messages": [HumanMessage("北京天气怎么样？")]})

    print("\n结论：两种写法功能等价。装饰器=语法糖（自动生成 AgentMiddleware 子类）；")
    print("要状态/多钩子共享/复用 → 类；单钩子/无状态/一次性 → 装饰器。")
