"""
16 - 演示 HumanInTheLoopMiddleware（人工介入 / 审批中间件）+ checkpointer

作用：agent 要调用"需确认"的工具时，先暂停，等人批准(approve)或拒绝(reject)再继续。
      常用于发邮件、删数据、转账等危险操作前的人工审核。
      checkpointer=InMemorySaver() 是 LangGraph 的"记忆 + 可中断/恢复"机制，
      必须配合它，中断后才能在同一个 thread 上 resume 续跑。

课程片段的两个问题（已修正）：
  1) HumanInTheLoopMiddleware(...) 实例没放进 middleware=[...] 列表，等于不生效。
  2) 中断后恢复时，resume 必须传 {"decisions": [{"type": "approve"/"reject"}]}。
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph.errors import GraphInterrupt

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return f"{city} 今天晴，25度"


@tool
def get_news(topic: str) -> str:
    """查询指定主题的新闻。"""
    return f"关于 {topic} 的新闻：今日无重大事件。"


@tool
def read_email_tool(email_id: str) -> str:
    """读取指定邮件的内容。"""
    return f"邮件 {email_id}：你好，这是一封测试邮件。"


@tool
def send_email_tool(to: str, body: str) -> str:
    """向指定收件人发送邮件。"""
    return f"已向 {to} 发送邮件：{body}"


model = init_chat_model(
    model="openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

checkpointer = InMemorySaver()

# 人工介入配置：天气/新闻需确认；读邮件不确认；发邮件需确认且自定义描述
hitl = HumanInTheLoopMiddleware(
    interrupt_on={
        "get_weather": True,
        "get_news": True,
        "read_email_tool": False,
        "send_email_tool": {"allowed_decisions": ["approve", "reject"], "description": "发送邮件中断啦"},
    },
    description_prefix="中断啦",
)

agent = create_agent(
    model=model,
    tools=[get_weather, get_news, read_email_tool, send_email_tool],
    checkpointer=checkpointer,
    middleware=[hitl],          # 关键：必须装进中间件列表
)

config = {"configurable": {"thread_id": "demo-1"}}

print("=== 第 1 步：调用 agent（模型要调 get_weather，应被 HITL 中断）===")
result = agent.invoke({"messages": [("user", "帮我查一下北京天气")]}, config=config)

# 兼容两种中断表达：抛出 GraphInterrupt 异常，或返回含 __interrupt__ 的状态
interrupts = None
if isinstance(result, dict) and "__interrupt__" in result:
    interrupts = result["__interrupt__"]
    print(f"触发中断（返回式）！共 {len(interrupts)} 个")
elif isinstance(result, dict):
    print("未中断。消息流:")
    for m in result["messages"]:
        tc = getattr(m, "tool_calls", None)
        print("  ", type(m).__name__, repr(getattr(m, "content", "")[:30]),
              ("tool_calls=" + str(tc) if tc else ""))

if interrupts is not None:
    for it in interrupts:
        req = it.value if hasattr(it, "value") else it
        # LangGraph 把 HITLRequest 序列化成 dict，做兼容
        ars = req.get("action_requests") if isinstance(req, dict) else req.action_requests
        rcs = req.get("review_configs") if isinstance(req, dict) else req.review_configs
        for ar, rc in zip(ars, rcs):
            name = ar.get("name") if isinstance(ar, dict) else ar.name
            args = ar.get("args") if isinstance(ar, dict) else ar.args
            allowed = rc.get("allowed_decisions") if isinstance(rc, dict) else rc.allowed_decisions
            desc = ar.get("description") if isinstance(ar, dict) else ar.description
            print(f"  工具={name} 参数={args}")
            print(f"  允许决策={allowed} 描述={desc!r}")

    print("\n=== 第 2 步：人工 approve 恢复执行 ===")
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}), config=config
    )
    print("最终回答:", result["messages"][-1].content)

    # 与课程截图一致：把恢复后的完整消息流逐条漂亮打印
    print("\n==== 审批后继续执行 ====")
    for msg in result["messages"]:
        msg.pretty_print()
