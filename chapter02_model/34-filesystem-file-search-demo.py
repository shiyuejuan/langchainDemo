"""
34 - FilesystemFileSearchMiddleware 演示（本地文件检索，LangChain 1.x + DeepSeek）

课程片段：9、FilesystemFileSearchMiddleware —— 给 Agent 装上本地文件搜索眼

源码确认（lc 1.2.12）：
  · FilesystemFileSearchMiddleware(*, root_path, use_ripgrep=True, max_file_size_mb=10)
  · 它是"加工具"型中间件：不 wrap 任何调用，而是 __init__ 里造两个 @tool
    挂到 self.tools，create_agent 自动合并进 agent 工具列表
  · glob_search(pattern, path="/")        按文件名模式找文件（pathlib.glob）
  · grep_search(pattern, path="/", include=None,
                output_mode="files_with_matches"|"content"|"count")
    内容搜索：优先 ripgrep(rg --json, 30s 超时)，没装 rg 自动降级纯 Python 正则
  · 沙箱：agent 看到的是"虚拟路径" /xxx（相对 root_path）
    _validate_and_resolve_path 禁止 .. 和 ~ 穿越，越界直接 ValueError
  · max_file_size_mb 只在 Python 搜索路径生效（ripgrep 路径不查大小）

demo 验证：
  ① 注入工具清单
  ② 不经模型直接调工具（确定性看清输入输出）
  ③ 沙箱防穿越：path 传 "../" 被拒
  ④ 真实 agent 提问让其自选 grep_search
"""
import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware
from langchain_core.messages import HumanMessage
from rich import print as rprint
load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

SANDBOX = Path("/Users/shiyuejuan/Desktop/Demo/langchainDemo/chapter02_model/file_search_sandbox")

# ── 自建房文件沙箱（幂等：先清再建）──────────────────────────────
if SANDBOX.exists():
    shutil.rmtree(SANDBOX)
SANDBOX.mkdir(parents=True)
files = {
    "notes/todo.md": "- 复习 LangChain 中间件\n- 买牛奶\n- 部署 redis 缓存服务\n",
    "src/main.py":   "import os\nimport redis\n\nconn = redis.Redis(host='localhost')\nprint('connect ok')\n",
    "src/utils.py":  "def add(a, b):\n    return a + b\n\n\ndef now():\n    from datetime import datetime\n    return datetime.now()\n",
    "docs/deploy.md": "# 部署手册\n生产环境 redis 地址: redis://prod:6379\n密码见内部文档\n",
}
for rel, content in files.items():
    p = SANDBOX / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

print("rg 是否可用:", shutil.which("rg") or "否（走 Python 降级）")


def rule(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


# ── ① 注入的工具清单 ─────────────────────────────────────────
mw = FilesystemFileSearchMiddleware(root_path=str(SANDBOX))
rule("① 注入的工具清单")
print("self.tools 名字:", [t.name for t in mw.tools])
print("glob_search 描述:", mw.glob_search.description[:60], "...")
print("grep_search 描述:", mw.grep_search.description[:60], "...")


# ── ② 直接调工具（确定性强，不经模型）─────────────────────────
rule("② 直接调 glob_search / grep_search（看清输入输出）")
print("glob **/*.py:\n", mw.glob_search.invoke({"pattern": "**/*.py"}))
print("glob **/*.md:\n", mw.glob_search.invoke({"pattern": "**/*.md"}))
print("grep redis (content):\n", mw.grep_search.invoke({"pattern": "redis", "output_mode": "content"}))
print("grep redis (count):\n", mw.grep_search.invoke({"pattern": "redis", "output_mode": "count"}))
print("grep python def (include=*.py, content):\n",
      mw.grep_search.invoke({"pattern": r"def \w+", "include": "*.py", "output_mode": "content"}))


# ── ③ 沙箱防穿越 ──────────────────────────────────────────────
rule("③ 沙箱防护：越界路径 / 非法正则")
print("glob path='../' ->", repr(mw.glob_search.invoke({"pattern": "*.py", "path": "../"})))
print("grep 非法正则 '[' ->", repr(mw.grep_search.invoke({"pattern": "["})))


# ── ④ 真实 agent 自选工具 ────────────────────────────────────
rule("④ 真实 agent 提问")
deepseek = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)
agent = create_agent(model=deepseek, tools=[], middleware=[mw])
result = agent.invoke({"messages": [HumanMessage(
    "帮我查一下：项目里哪些文件用到了 redis？分别在第几行？把行内容也给我看看")]})
rprint("result======\n",result["messages"])
# 列出模型实际调用了哪些工具
used = []
for m in result["messages"]:
    if hasattr(m, "tool_calls") and m.tool_calls:
        for tc in m.tool_calls:
            used.append((tc["name"], tc["args"]))
print("模型实际调用的工具:", used)
print("\n最终回答:\n", result["messages"][-1].content)

shutil.rmtree(SANDBOX)  # 演示完清理沙箱
