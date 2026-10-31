"""
11 - 演示 agent.stream(..., stream_mode="updates") 流式输出

对应课程片段：
    for chunk in customer_service_agent.stream(
        {"messages": [{"role": "user", "content": "查询客户ID为 CUST123456 的完整信息和可用优惠"}]},
        stream_mode="updates",
    ):
        rprint(chunk)
        print("-" * 50)

要点：
- .stream() 流式调用：模型/工具逐步产出，逐块返回（不必等全部跑完）
- stream_mode="updates"：每个 chunk = {节点名: 该节点产生的「状态更新」}
- 输入消息支持三种写法：字典 {"role":..,"content":..} / 元组 ("user",..) / HumanMessage 对象
- rprint 来自 rich 库，彩色美化打印（需 pip install rich）
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from rich import print as rprint  # rich 的彩色 print（与课程 rprint 用途一致）

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
print("===== stream_mode='updates' 流式输出 =====")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="updates",
):
    rprint(chunk)
    # 摘要：看清每个 chunk 是哪个节点、产出了什么类型的消息
    node = next(iter(chunk.keys()))
    updates = chunk[node]
    msgs = updates.get("messages", []) if isinstance(updates, dict) else []
    kinds = [type(m).__name__ for m in msgs]
    print(f"   → 节点 '{node}' 更新了 {len(msgs)} 条消息: {kinds}")
    print("-" * 50)
