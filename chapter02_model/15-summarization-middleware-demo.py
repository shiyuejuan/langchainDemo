"""
15 - 演示 SummarizationMiddleware（历史消息自动摘要中间件）

作用：当对话变长、接近 token 上限时，自动把较早的消息压成一条摘要，
保留最近的 keep 条，省 token、防超出模型上下文窗口。
文档：保留 AI/Tool 消息对，避免破坏工具调用上下文。

课程片段的两个问题（已修正）：
  1) SummarizationMiddleware(...) 只是创建了实例，却没放进 middleware=[...]，
     所以根本不生效！必须 middleware=[SummarizationMiddleware(...)]。
  2) model="deepseek-v4-flash" 不是真实模型，且引用的 model 变量需先定义。
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

model = init_chat_model(
    model="openai:deepseek-chat",
    profile={"max_input_tokens": 128_000},   # 加这项：fraction 触发需模型上下文上限来算比例
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

# 创建摘要中间件：消息数>6 或 token>100 时触发，保留最近 2 条，自定义提示
summary_mw = SummarizationMiddleware(
    model=model,
    trigger=[("tokens", 100), ("messages", 6), ("fraction", 0.0001)],
    keep=("messages", 2),
    summary_prompt="对历史消息摘要，消息列表如下\n{messages}",
)

# 关键：必须放进 middleware 列表，否则不生效
agent = create_agent(model=model, middleware=[summary_mw])

# 构造一段较长的对话历史（9 条消息，超过 6 条触发阈值）
history = []
for i in range(4):
    history.append(("user", f"第{i + 1}轮：北京今天天气如何"))
    history.append(("assistant", f"第{i + 1}轮回答：北京今天晴，温度约25度"))
history.append(("user", "请总结我们前面聊了什么"))

print("输入消息数:", len(history))
r = agent.invoke({"messages": history})
msgs = r["messages"]
print("触发摘要后，输出消息数:", len(msgs))
print("\n输出消息类型序列（看是否被压缩）:")
for m in msgs:
    t = type(m).__name__
    content = m.content
    if isinstance(content, str) and len(content) > 50:
        content = content[:50] + "...(略)"
    print(f"  {t}: {content!r}")
