"""
46 - 第09章：上下文工程（Context Engineering）

核心命题：模型只知道你给它看的东西。回答质量 = f(上下文质量)。

四大策略（选做对的事）：
  写 Write    ：把内容存到上下文之外（checkpointer/memory/便签）
  选 Select   ：把对的内容拉进上下文（system 注入/RAG/挑工具）
  压 Compress ：历史太长就压缩（Summarization/ClearToolUses，28 课已学）
  隔 Isolate  ：拆子 agent，各自只带自己的上下文

本 demo 一个问题测三种上下文（温度=0，保证可复现）：
  ① 无资料：模型编/说不知道        → 上下文缺失 = 答案缺失
  ② 对资料：答对"保修 12 个月"      → 上下文给对 = 答案对
  ③ 错资料：照样答"保修 3 个月"     → 模型只信你给的上下文（垃圾进垃圾出）
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

QUESTION = "智学笔的保修期是多久？"


def ask(prompt: str, label: str):
    agent = create_agent(model=deepseek, tools=[], system_prompt=prompt)
    r = agent.invoke({"messages": [HumanMessage(QUESTION)]})
    answer = r["messages"][-1].content
    print(f"\n[{label}]")
    print(f"  上下文（system）前 60 字：{prompt[:60]}...")
    print(f"  回答：{answer}")
    return answer


if __name__ == "__main__":
    no_knowledge = "你是智学笔官方客服，请简洁回答用户问题。"

    right_knowledge = """你是智学笔官方客服，只依据以下资料回答：
【产品资料】
- 产品：智学笔 X1，AI 智能手写笔
- 保修：整机保修 12 个月，电池保修 6 个月
- 充电：Type-C，1 小时充满，续航 30 天
资料里没有的，回答"暂时没有相关信息"。"""

    wrong_knowledge = """你是智学笔官方客服，只依据以下资料回答：
【产品资料】
- 产品：智学笔 X1，AI 智能手写笔
- 保修：整机保修 3 个月
资料里没有的，回答"暂时没有相关信息"。"""

    print("=" * 60)
    print("上下文工程对照实验：同一个问题 × 三种上下文")
    print("=" * 60)
    ask(no_knowledge, "① 无资料（Select 缺失）")
    ask(right_knowledge, "② 对资料（Select 正确）")
    ask(wrong_knowledge, "③ 错资料（垃圾进垃圾出）")

    print("\n结论：模型没有'事实'，只有'上下文'。")
    print("上下文工程 = 系统地决定每次调模型时给它看什么：写/选/压/隔。")
