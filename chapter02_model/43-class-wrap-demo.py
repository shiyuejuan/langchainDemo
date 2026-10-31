"""
43 - wrap_model_call / wrap_tool_call 的「类方式」实现

对应 40 课的函数装饰器方式（@wrap_model_call / @wrap_tool_call）。
类方式的独特优势：有 self，可持有跨调用状态、把多个钩子打包成可复用组件。

三种玩法（类方式重写 40 课的三个例子）：
  ① 放行：    log 后 return handler(request)
  ② 改写：    handler(request.override(system_message=...)) 注入指令
  ③ 短路：    不调 handler，自己造结果返回（不执行真工具/真模型）
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}：晴，22 度（这是真工具返回的）"


def _append_sys(sys_msg, extra: str) -> SystemMessage:
    """在原有 system 消息后面追加指令（没有就新建）。"""
    if sys_msg is None:
        return SystemMessage(content=extra)
    return SystemMessage(content=sys_msg.content + extra)


# ===== 玩法① 放行：持有状态（类方式独有优势）=====
class PassMW(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.model_calls = 0   # self 持有跨调用计数

    @property
    def name(self):
        return "PassMW"

    def wrap_model_call(self, request, handler):
        self.model_calls += 1
        print(f"  [类·放行] 第 {self.model_calls} 次调模型，记日志后原样放行")
        return handler(request)   # 原样放行


# ===== 玩法② 改写：注入硬性指令 =====
class InjectMW(AgentMiddleware):
    @property
    def name(self):
        return "InjectMW"

    def wrap_model_call(self, request, handler):
        extra = "\n[附加指令] 你的每句回答结尾必须带【完毕】二字。"
        new_sys = _append_sys(request.system_message, extra)
        print("  [类·改写] 给模型注入硬性指令后放行")
        return handler(request.override(system_message=new_sys))


# ===== 玩法③ 短路：拦截工具，不执行真工具 =====
class FakeToolMW(AgentMiddleware):
    @property
    def name(self):
        return "FakeToolMW"

    def wrap_tool_call(self, request, handler):
        # 注意：wrap_tool_call 的 request.tool_call 是「字典」，不是对象！
        #   取名字用 request.tool_call["name"]，取 id 用 request.tool_call["id"]
        #   而 wrap_model_call 的 request 才是带属性访问（request.messages / request.system_message）
        tc = request.tool_call
        print(f"  [类·短路] 拦截工具 {tc['name']}，不执行真工具，自己造假结果")
        return [ToolMessage(content="（模拟）北京：晴 22 度",
                            tool_call_id=tc["id"])]


deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


def _last_ai(result):
    for m in reversed(result["messages"]):
        if getattr(m, "type", None) == "ai" and m.content:
            return m.content
    return ""


if __name__ == "__main__":
    print("=" * 60)
    print("玩法① 放行（类方式，self 计数）")
    print("=" * 60)
    a1 = create_agent(model=deepseek, tools=[get_weather], middleware=[PassMW()])
    r1 = a1.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
    print("最终回答：", _last_ai(r1))

    print("\n" + "=" * 60)
    print("玩法② 改写（类方式，注入【完毕】指令）")
    print("=" * 60)
    a2 = create_agent(model=deepseek, tools=[get_weather], middleware=[InjectMW()])
    r2 = a2.invoke({"messages": [HumanMessage("你好")]})
    print("最终回答：", _last_ai(r2))

    print("\n" + "=" * 60)
    print("玩法③ 短路（类方式，拦截工具造假结果）")
    print("=" * 60)
    a3 = create_agent(model=deepseek, tools=[get_weather], middleware=[FakeToolMW()])
    r3 = a3.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
    print("最终回答：", _last_ai(r3))
    print("\n证明真工具没被调用：① 上方出现 [类·短路] 拦截打印行；"
          "② 真工具特有的『（这是真工具返回的）』字样没在回答里出现。")
