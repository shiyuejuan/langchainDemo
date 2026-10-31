"""
37 - Node-style hooks 函数用法（课程 5.3）

不用写类，一个装饰器把普通函数变成中间件（Node-style = 像 LangGraph 节点：
签名 (state, runtime)，返回 None / dict 状态更新 / Command 流程控制）。

两种用法（源码 overload 确认）：
  ① 裸装饰器   @before_model
     直接把函数变成 AgentMiddleware 实例，name 默认取函数名（天然唯一）
  ② 带参装饰器 @before_model(can_jump_to=["end"], ...)
     配置跳转权限后，钩子里 return {"jump_to": "end"} 可直接终止/跳转流程

demo 验证：
  ① 裸装饰器做日志（before/after 各一个），观察 agent 工具循环
  ② 带参装饰器做"停止守卫"：用户说'停止'→ 跳到 end，模型一次都不调
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

model_calls = {"n": 0}


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}：晴，22 度"


# ── 用法①：裸装饰器，普通函数直接变中间件 ────────────────────
@before_model
def log_before_model(state, runtime):
    model_calls["n"] += 1
    print(f"  [函数钩子] 第 {model_calls['n']} 轮调模型前，消息数 {len(state['messages'])}")


@after_model
def log_after_model(state, runtime):
    last = state["messages"][-1]
    head = str(last.content)[:40]
    print(f"  [函数钩子] 模型返回 → {type(last).__name__}: {head}")


deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


def rule(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


rule("用法① 裸装饰器 @before_model / @after_model")
print("产物类型:", type(log_before_model).__name__)
print("中间件 name:", log_before_model.name, "（默认取函数名，天然不会重名）")

agent1 = create_agent(model=deepseek, tools=[get_weather],
                      middleware=[log_before_model, log_after_model])
agent1.invoke({"messages": [HumanMessage("北京天气怎么样？")]})


# ── 用法②：带参装饰器 + jump_to 流程控制 ─────────────────────
@before_model(can_jump_to=["end"])
def exit_guard(state, runtime):
    last = str(state["messages"][-1].content)
    if "停止" in last:
        print("  [停止守卫] 检测到'停止' → jump_to end，模型一次都不会被调")
        return {"jump_to": "end"}
    model_calls["n"] += 1
    return None


rule("用法② 带参装饰器 @before_model(can_jump_to=['end'])")
agent2 = create_agent(model=deepseek, tools=[get_weather], middleware=[exit_guard])

model_calls["n"] = 0
print("正常问题（守卫放行）：")
agent2.invoke({"messages": [HumanMessage("上海天气怎么样？")]})
print(f"模型共跑了 {model_calls['n']} 轮\n")

model_calls["n"] = 0
print("含'停止'的问题（守卫拦截）：")
r = agent2.invoke({"messages": [HumanMessage("停止，不用回答")]})
print(f"模型共跑了 {model_calls['n']} 轮")
print("最终消息类型:", type(r["messages"][-1]).__name__,
      "→ 最后一条还是你的 HumanMessage，模型压根没响应")
