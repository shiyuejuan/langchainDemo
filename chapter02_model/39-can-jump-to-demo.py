"""
39 - 5.3.5 装饰器参数 can_jump_to（跳转白名单）

源码确认（factory.py::_add_middleware_edge）：
  · 不声明 can_jump_to → add_edge 写死默认路由，jump_to 静默无效
  · 声明了 → add_conditional_edges，路由函数实时读 state["jump_to"]
    destinations 白名单 = 默认下一站 + 你声明的目标（'model'/'tools'/'end'）
  · can_jump_to 的值存在钩子方法的 __can_jump_to__ 属性上（装饰器/hook_config 写入）
  · 附带实锤：after_model 边从列表最后一个开始链 → 反序执行（35 课发现的根源）

demo：
  ① 忘写 can_jump_to 的守卫 → jump_to 静默失效（模型照跑）
  ② 声明 ["model"] 的质检钩子 → 回答不合格就附提示跳回 model 重答（防死循环上限 2 次）
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
retry = {"n": 0}


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}：晴，22 度"


deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)


def rule(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


# ── ① 忘写 can_jump_to：jump_to 静默失效 ──────────────────────
@before_model
def naive_guard(state, runtime):
    model_calls["n"] += 1
    if "停止" in str(state["messages"][-1].content):
        print("  [天真守卫] 返回了 jump_to=end（但没声明 can_jump_to！）")
        return {"jump_to": "end"}
    return None


rule("① 未声明 can_jump_to → jump_to 静默失效")
agent1 = create_agent(model=deepseek, tools=[get_weather], middleware=[naive_guard])
agent1.invoke({"messages": [HumanMessage("停止，不用回答")]})
print(f"模型实际跑了 {model_calls['n']} 轮 → 说好的'停止'根本没停（不报错，就是不停）")


# ── ② 声明 can_jump_to=["model"]：质检不合格跳回重答 ──────────
@after_model(can_jump_to=["model"])
def quality_gate(state, runtime):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return None                      # 工具调用轮不质检
    if "完毕" in str(last.content) or retry["n"] >= 2:
        return None                      # 合格，或已达重试上限（防死循环）
    retry["n"] += 1
    print(f"  [质检] 第 {retry['n']} 次不合格（结尾缺'完毕'）→ 附加提示，jump_to model")
    return {"messages": [HumanMessage("请重新回答，并在结尾加上'完毕'二字")],
            "jump_to": "model"}


rule("② 声明 can_jump_to=['model'] → 回答不合格跳回重答")
agent2 = create_agent(model=deepseek, tools=[get_weather], middleware=[quality_gate])
r = agent2.invoke({"messages": [HumanMessage("用一句话介绍北京")]})
print(f"\n质检共打回 {retry['n']} 次")
print("最终回答:", r["messages"][-1].content[:80])
