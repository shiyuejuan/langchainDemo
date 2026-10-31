"""
14 - 演示 create_agent 的 middleware 中间件

middleware 是 LangChain 1.x agent 的"中间件/钩子"：在 agent 执行流程的特定时机
（模型调用前/后、agent 运行前/后、工具调用前/后）插入你的自定义逻辑。
常见现成中间件（langchain.agents.middleware 模块里都有）：
    model_retry / tool_retry   重试
    model_call_limit / tool_call_limit   限流（限制调用次数）
    pii                        PII 脱敏/检测
    human_in_the_loop          人工审核介入
    summarization              长对话摘要

课程片段：create_agent(model="deepseek-v4-flash", middleware=[]) 是空中间件（默认行为）。

下面用 before_model / after_model 钩子打印日志，直观展示 middleware 夹在 agent 执行里。
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

model = init_chat_model(
    model="openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

# 计数器，统计模型被调用了几次
counter = {"n": 0}


@before_model
def log_before(state, runtime):
    """模型调用前触发：打印一条日志（也可在这里改写 state）。"""
    counter["n"] += 1
    print(f"[middleware before_model] 第 {counter['n']} 次即将调用模型")


@after_model
def log_after(state, runtime):
    """模型调用后触发：打印当前已有几条消息。"""
    print(f"[middleware after_model] 模型已返回，当前消息数: {len(state['messages'])}")


# 课程的空 middleware=[] 换成装了真实钩子的列表
agent = create_agent(
    model=model,
    middleware=[log_before, log_after],
)

print("=== 带中间件的 agent 实跑 ===")
r = agent.invoke({"messages": [("user", "你好，用一句话介绍北京")]})
print("\n最终回答:", r["messages"][-1].content)
print(f"\n(中间件统计：模型共被调用 {counter['n']} 次)")
