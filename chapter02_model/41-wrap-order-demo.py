"""
41 - wrap_tool_call 与 wrap_model_call 的调用顺序（洋葱模型延伸）

结论（实跑验证）：
  · 两者各是独立的洋葱层，都按 middleware 列表顺序排层：外层先进/后出
  · 但它们触发在"不同时机"——model 包住模型节点，tool 包住工具节点
  · 一个 agent 循环（模型决定调工具）：
        MODEL 洋葱(外层→内层→模型→内层→外层)
        → TOOL  洋葱(外层→内层→工具→内层→外层)
        → MODEL 洋葱 再来一轮
  · 关键：MODEL 各层先全部跑完，TOOL 各层才开跑（模型节点整段在前，工具节点整段在后）
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


class OrderMW(AgentMiddleware):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    @property
    def name(self):
        return f"OrderMW-{self.tag}"

    def wrap_model_call(self, request, handler):
        print(f"  [{self.tag}] MODEL ▶ before")
        r = handler(request)
        print(f"  [{self.tag}] MODEL ◀ after")
        return r

    def wrap_tool_call(self, request, handler):
        print(f"  [{self.tag}] TOOL  ▶ before")
        r = handler(request)
        print(f"  [{self.tag}] TOOL  ◀ after")
        return r


deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


if __name__ == "__main__":
    agent = create_agent(model=deepseek, tools=[get_weather],
                         middleware=[OrderMW("外层"), OrderMW("内层")])
    print("middleware = [外层, 内层]，各含两种 wrap\n")
    agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
    print("\n读法：外层先包裹→内层包裹→真正调用→内层拆→外层拆；"
          "MODEL 整段跑完，TOOL 整段才开跑。")
