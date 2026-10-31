"""
42 - after_model 之后还会执行 wrap_model_call 吗？（模型节点内部慢动作）

结论（实跑验证）：
  · wrap_model_call 包的是「模型调用那一刻」本身：
        before_model
          wrap_model_call ▶ 进
            【真正调 LLM】
          wrap_model_call ◀ 出
        after_model
  · 所以 wrap_model_call 的「出（拆壳）」发生在 after_model 之前；
    after_model 是节点收尾，它之后同一次模型调用不会再跑 wrap_model_call。
  · after_model 之后节点就结束：要么去工具节点、要么结束、要么下一轮重新从 before_model 开始
    （新一轮会有「新的」wrap_model_call，但那不是 after_model 之后紧接的同一段）。
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


class TraceMW(AgentMiddleware):
    @property
    def name(self):
        return "TraceMW"

    def before_model(self, state, runtime):
        print("  【before_model】  ← 模型节点开始")

    def wrap_model_call(self, request, handler):
        print("  【wrap_model_call ▶ 进】  ← 包住 LLM 调用")
        r = handler(request)
        print("  【wrap_model_call ◀ 出】  ← 拆壳（在 after_model 之前）")
        return r

    def after_model(self, state, runtime):
        print("  【after_model】  ← 模型节点收尾")

    def wrap_tool_call(self, request, handler):
        print("  【wrap_tool_call ▶ 进】")
        r = handler(request)
        print("  【wrap_tool_call ◀ 出】")
        return r


deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


if __name__ == "__main__":
    agent = create_agent(model=deepseek, tools=[get_weather], middleware=[TraceMW()])
    print("模型节点内部顺序（每轮标①/②）：\n")
    agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
    print("\n观察：每个模型节点里 wrap 的「出」都比 after_model 早；"
          "after_model 之后只有「工具节点」或「下一轮的 before_model」，"
          "不会再出现一次 wrap_model_call。")
