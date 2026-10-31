"""
18 - HITL 完整可运行代码（把课程截图那几行补全）

截图里只有“审批恢复 + 打印”的局部片段，缺了：
  · 各种 import（含 from langgraph.types import Command）
  · 模型 / 工具 / agent / checkpointer / middleware 的搭建
  · 第一次 invoke 触发中断
  · 中断检测（本版本是返回式 result["__interrupt__"]）
  · decisions 的定义
这里全部补齐，直接 python 18-hitl-complete.py 即可运行。
"""
import os
from dotenv import load_dotenv

# 1) 模型 + 工具 + 中间件 需要的导入
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool

# 2) HITL 必备：checkpointer（存状态，才能中断/恢复）+ Command（恢复指令）
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command   # ← 截图用到的 Command 必须引入！

# ====== 模型（DeepSeek，OpenAI 兼容协议）======
load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)
model = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

# ====== 工具 ======
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return f"{city} 今天晴，25度"

@tool
def send_email_tool(to: str, body: str) -> str:
    """向指定收件人发送邮件（危险操作，需审批）。"""
    return f"已向 {to} 发送邮件：{body}"

# ====== checkpointer + HITL 中间件 ======
checkpointer = InMemorySaver()
hitl = HumanInTheLoopMiddleware(
    interrupt_on={"get_weather": True, "send_email_tool": True},  # 这两个工具调用前需确认
    description_prefix="中断啦",
)

agent = create_agent(
    model=model,
    tools=[get_weather, send_email_tool],
    checkpointer=checkpointer,      # 必须：中断后靠它 resume
    middleware=[hitl],              # 必须：把中间件实例装进列表
)

# 同一个 thread_id：第一次调用和第二次恢复必须用同一个
config = {"configurable": {"thread_id": "demo-1"}}

# ====== 第 1 步：第一次调用，模型要调 get_weather → 被 HITL 中断 ======
result = agent.invoke(
    {"messages": [("user", "帮我查一下北京天气")]},
    config=config,
)

# 本 langgraph 版本：中断是“返回式”，写在 result["__interrupt__"] 里
interrupts = result.get("__interrupt__") if isinstance(result, dict) else None

# ========== ↓↓↓ 以下就是你截图里的那段（已补全上下文）↓↓↓ ==========
if interrupts:
    # 把中断里“要调哪个工具、参数是什么”打印出来给人看
    for it in interrupts:
        req = it.value if hasattr(it, "value") else it
        ars = req.get("action_requests") if isinstance(req, dict) else req.action_requests
        for ar in ars:
            name = ar.get("name") if isinstance(ar, dict) else ar.name
            args = ar.get("args") if isinstance(ar, dict) else ar.args
            print(f"[待审批] 工具={name} 参数={args}")

    # 人的决定：批准。本版本 resume 必须是这个结构
    decisions = {"decisions": [{"type": "approve"}]}   # 想拒绝就改 "reject"

    # 审批通过 → 从断点恢复（必须同一 thread_id）
    resumed_response = agent.invoke(
        Command(resume=decisions),
        config=config,  # 必须是同一个 thread_id
    )

    print("==== 审批后继续执行 ====")
    for msg in resumed_response["messages"]:
        msg.pretty_print()
# ========== ↑↑↑ 截图片段到此 ↑↑↑ ==========
else:
    print("没有触发中断，完整回答：", result["messages"][-1].content)
