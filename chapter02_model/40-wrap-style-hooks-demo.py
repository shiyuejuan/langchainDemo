"""
40 - 5.4 Wrap-style hooks 函数用法（@wrap_model_call / @wrap_tool_call）

源码确认（装饰器签名）：
  · @wrap_model_call：裸用 或 (state_schema=, tools=, name=)  —— 没有 can_jump_to！
    函数签名 (request, handler)，必须 return handler(request) 或改写后的结果
  · @wrap_tool_call：裸用 或 (tools=, name=)
    函数签名 (request, handler)，request 是 ToolCallRequest

与 5.3 钩子型函数式的区别：
  · 钩子型 (state, runtime)：站在"车站"上看/发车，可 jump_to 控流程
  · wrap 型 (request, handler)：直接包住"列车运行"，进站前改请求、出站后改结果

demo：
  ① 裸装饰做日志（4 行搞定，对比类式 12 行）
  ② wrap 型真威力：request.override 注入 system_message
     —— 不用跳转，就让回答结尾带上'完毕'（对照 39 课的质检+跳转方案）
  ③ @wrap_tool_call 包住工具调用，看 args
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}：晴，22 度"


# ── ① 裸装饰：4 行日志中间件 ─────────────────────────────────
@wrap_model_call
def log_model(request, handler):
    print(f"  [wrap] → 调模型前：{len(request.messages)} 条消息")
    result = handler(request)
    print(f"  [wrap] ← 模型返回：{type(result).__name__}")
    return result


@wrap_tool_call
def log_tool(request, handler):
    tc = request.tool_call
    print(f"  [wrap-tool] → 执行 {tc.get('name')}，参数 {tc.get('args')}")
    result = handler(request)
    print(f"  [wrap-tool] ← 工具返回：{str(result.content)[:30]}")
    return result


deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


def rule(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


rule("① 裸装饰 @wrap_model_call / @wrap_tool_call（顺便带 name 参数展示）")
print("产物 name:", log_model.name)
agent1 = create_agent(model=deepseek, tools=[get_weather],
                      middleware=[log_model, log_tool])
r1 = agent1.invoke({"messages": [HumanMessage("北京天气怎么样？")]})


# ── ② wrap 型真威力：override 改写请求 ───────────────────────
@wrap_model_call(name="politeness_wrap")     # 带参：自定义中间件名
def add_format_rule(request, handler):
    old = request.system_message.content if request.system_message else ""
    req2 = request.override(system_message=SystemMessage(
        content=(old + "\n附加要求：回答的结尾必须带上'完毕'二字。").strip()))
    return handler(req2)                     # 把改写后的请求交给下一层


rule("② request.override 注入格式要求（不跳转，等效 39 课质检）")
agent2 = create_agent(model=deepseek, tools=[get_weather], middleware=[add_format_rule])
r2 = agent2.invoke({"messages": [HumanMessage("用一句话介绍上海")]})
print("最终回答:", r2["messages"][-1].content)
