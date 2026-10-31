"""
22 - ToolCallLimitMiddleware 演示（工具调用限流，LangChain 1.x + DeepSeek）

课程要点：可限制"所有工具"总次数，也可只限"特定工具"；三种退出行为
  · continue（默认）：超限的工具【不再执行】，给模型回一条"超限"ToolMessage，
                      agent 继续跑，模型自己决定怎么办（能力弱可能死循环）
  · end：立即结束（注入提示消息）
  · error：抛 ToolCallLimitExceededError
演示三场景：
  1) 只限 search_web（tool_name="search_web", run_limit=2, continue）
     → 第 3 次调用被拦截，但 agent 仍给出回答
  2) 所有工具 run_limit=2 + end → 提前结束
  3) 只限 search_web + error → 抛异常
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.tools import tool
from langchain.messages import HumanMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)
model = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

@tool
def search_web(query: str) -> str:
    """联网搜索（模拟昂贵 API）。"""
    return f"关于「{query}」的搜索结果：……"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

# ===== 场景 1：只限 search_web，continue（默认行为）=====
agent_c = create_agent(
    model=model,
    tools=[search_web, calculator],
    middleware=[
        ToolCallLimitMiddleware(tool_name="search_web", run_limit=2, exit_behavior="continue"),
    ],
)
print("=" * 64)
print("场景1：tool_name='search_web', run_limit=2, continue（默认）")
print("任务要求搜 4 次——第 3 次起应被拦截，但 agent 仍应给出回答")
r = agent_c.invoke({"messages": [HumanMessage(
    "请用搜索工具依次查询：A、B、C、D 四个关键词，然后告诉我完成情况。")]})
for m in r["messages"]:
    if m.__class__.__name__ == "ToolMessage":
        content = str(m.content)
        tag = " ← 被拦截" if "limit" in content.lower() or "exceed" in content.lower() else ""
        print(f"  [ToolMessage:{m.name}] {content[:70]}{tag}")
print(f"  最终回答：{str(r['messages'][-1].content)[:120]}")

# ===== 场景 2：所有工具 run_limit=2 + end =====
agent_e = create_agent(
    model=model,
    tools=[search_web, calculator],
    middleware=[ToolCallLimitMiddleware(run_limit=2, exit_behavior="end")],
)
print("\n" + "=" * 64)
print("场景2：所有工具 run_limit=2, end")
r2 = agent_e.invoke({"messages": [HumanMessage("搜索 X，再算 1+1，再搜索 Y。")]})
print(f"  最后一条消息：[{r2['messages'][-1].__class__.__name__}] "
      f"{str(r2['messages'][-1].content)[:80]}")

# ===== 场景 3：只限 search_web + error =====
agent_x = create_agent(
    model=model,
    tools=[search_web],
    middleware=[ToolCallLimitMiddleware(tool_name="search_web", run_limit=1, exit_behavior="error")],
)
print("\n" + "=" * 64)
print("场景3：tool_name='search_web', run_limit=1, error")
try:
    agent_x.invoke({"messages": [HumanMessage("搜索 A，再搜索 B。")]})
    print("  （未抛异常？）")
except Exception as e:
    print(f"  抛出异常：{type(e).__name__}：{str(e)[:90]}")
