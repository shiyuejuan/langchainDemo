"""
12 - 对比 stream_mode="updates" 与 stream_mode="messages"

两种模式都是 agent.stream() 的"边跑边汇报"，但汇报粒度不同：
- "updates"：每个 chunk = {节点名: 该节点产生的状态更新}（粗粒度，几个大块）
- "messages"：每个 chunk = (单条消息的流式片段, 元数据)（细粒度，token 级，适合打字机效果）

另外纠正一种常见语法错误（把 print 写进 stream 参数、漏掉 stream 闭括号）：
    错误: for chunk in agent.stream({...}, stream_mode="messages"  print(chunk) ): print("-"*50)
    正确: for chunk in agent.stream({...}, stream_mode="messages"):
              print(chunk); print("-"*50)
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return f"{city} 今天晴，25度"


model = init_chat_model(
    model="openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)
agent = create_agent(model=model, tools=[get_weather])

question = "北京天气怎么样？"

print("===== updates 模式（粗粒度，按节点聚合）=====")
cnt = 0
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="updates",
):
    cnt += 1
    nodes = list(chunk.keys()) if isinstance(chunk, dict) else chunk
    print(f"chunk {cnt}: 类型={type(chunk).__name__}, 节点={nodes}")
print(f"updates 共 {cnt} 个 chunk\n")

print("===== messages 模式（细粒度，token 级，截断前 30 个）=====")
cnt = 0
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="messages",
):
    cnt += 1
    msg = chunk[0]
    content = getattr(msg, "content", "")
    if isinstance(content, str) and content:
        # 有文字 -> 这就是"打字机"逐字流出的部分
        print(f"chunk {cnt}: {type(msg).__name__} 文本片段={content[:30]!r}")
    else:
        # 工具调用阶段：content 为空，但带 tool_call_chunks
        tag = "工具调用片段" if getattr(msg, "tool_call_chunks", None) else "空"
        print(f"chunk {cnt}: {type(msg).__name__} content=空 [{tag}]")
    if cnt >= 30:
        print("   ... messages 模式是 token 级流式，后面还有更多，已截断")
        break
print(f"(已演示前 {cnt} 个；messages 模式实际 chunk 数量远多于 updates)")
