"""
17 - TodoListMiddleware 演示（LangChain 1.x + DeepSeek）

middleware=[TodoListMiddleware()] 到底做了什么 —— 用真实代码证实：
  (A) 它会【自动】给 agent 注入 write_todos 工具（tools= 里不用手写）
  (B) 它会往 agent 图里挂一个 after_model 钩子节点
  (C) 多步骤任务下，agent 会先 write_todos 规划，再逐步执行
  (D) 跑完可在 result["todos"] 拿到带状态的待办清单
"""
import os
import tempfile
import json
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import HumanMessage

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env")
model = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

WS = tempfile.mkdtemp(prefix="todo-demo-")

# 预先放一个"有 bug"的文件，制造一个真正的多步骤修复任务
with open(os.path.join(WS, "app.py"), "w", encoding="utf-8") as f:
    f.write(
        "def compute(a, b):\n"
        "    # 故意写错的减法\n"
        "    return a + b\n"
    )

@tool
def list_files(path: str = ".") -> str:
    """列出工作区下的文件/目录。"""
    full = os.path.join(WS, path)
    if not os.path.isdir(full):
        return f"不是目录: {path}"
    return "\n".join(sorted(os.listdir(full))) or "(空)"

@tool
def read_file(path: str) -> str:
    """读取工作区里的文件内容。"""
    full = os.path.join(WS, path)
    try:
        with open(full, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {e}"

@tool
def write_file(path: str, content: str) -> str:
    """向工作区写入一个文件。"""
    full = os.path.join(WS, path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {path}"

@tool
def run_tests() -> str:
    """运行测试（演示桩：始终通过）。"""
    return "全部测试通过。"

tools = [list_files, read_file, write_file, run_tests]

# ============ 重点：中间件放进 middleware=[...] ============
mw = TodoListMiddleware()
agent = create_agent(
    model=model,
    tools=tools,
    middleware=[mw],
    system_prompt=(
        "你是一个代码修复助手。遇到多步骤任务时，先使用 write_todos 制定待办事项;"
        "然后读取文件、修复代码并运行测试。工作全部在工作区下进行。"
    ),
)

print("=" * 64)
print("(A) 注入证据：中间件实例自带的 tools 里就有 write_todos")
print("    ", [t.name for t in mw.tools if isinstance(t, BaseTool)])

print("\n(B) 注入证据：agent 图里多了中间件钩子节点")
print("    ", [n for n in agent.get_graph().nodes.keys() if "Todo" in n])

def agent_tool_names(agent):
    # 在编译后的图里搜 write_todos 字样，证明已绑进模型可调用工具
    txt = repr(agent.get_graph().nodes)
    return "write_todos" in txt
print("     agent 可用工具中包含 write_todos ?", agent_tool_names(agent))

print("\n(C) write_todos 工具要求的参数（agent 要照这个结构传）：")
wt = next(t for t in mw.tools if isinstance(t, BaseTool) and t.name == "write_todos")
print("    ", json.dumps(wt.args, ensure_ascii=False))

# ============ 一个明显更复杂的多步骤任务 ============
task = (
    "工作区里有个 app.py，其中 compute(a,b) 故意写错了（应该是减法却写成加法）。"
    "请按步骤完成：1) 读 app.py 定位 bug；2) 修复 compute 为减法；"
    "3) 写 test_app.py 用断言覆盖；4) 运行测试确认通过；5) 写 README.md 说明用法。"
)
result = agent.invoke({"messages": [HumanMessage(task)]})

print("\n" + "=" * 64)
print("(D) agent 实际调用的工具顺序：")
called = []
for m in result["messages"]:
    for tc in getattr(m, "tool_calls", []) or []:
        called.append(tc["name"])
print("    ", called)
used_todo = "write_todos" in called
print("    → write_todos 被自动调用了吗？", used_todo)

print("\nresult['todos']（带状态的待办清单，模型没用时为 None）：")
print(json.dumps(result.get("todos"), ensure_ascii=False, indent=2, default=str))

print("\n" + "=" * 64)
print("工作区最终文件：", list_files.invoke({}))
print("\nagent 最终回答：")
print(result["messages"][-1].content)
