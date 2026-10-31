"""
34b - 检索文件名称(glob) vs 检索文件内容(grep) 对照演示
"""
import os, shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents.middleware import FilesystemFileSearchMiddleware
from rich import print as rprint
load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

SANDBOX = Path("/Users/shiyuejuan/Desktop/Demo/langchainDemo/chapter02_model/file_search_sandbox")
if SANDBOX.exists():
    shutil.rmtree(SANDBOX)
SANDBOX.mkdir(parents=True)
files = {
    "notes/todo.md": "- 复习 LangChain 中间件\n- 买牛奶\n- 部署 redis 缓存服务\n",
    "src/main.py":   "import os\nimport redis\n\nconn = redis.Redis(host='localhost')\nprint('connect ok')\n",
    "src/utils.py":  "def add(a, b):\n    return a + b\n",
    "docs/deploy.md": "# 部署手册\n生产环境 redis 地址: redis://prod:6379\n",
}
for rel, content in files.items():
    p = SANDBOX / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

mw = FilesystemFileSearchMiddleware(root_path=str(SANDBOX))


def rule(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


rule("场景 A：我只记得文件名后缀是 .py，想把这些文件找出来")
print("→ 用 glob_search：它只看【文件名】，完全不打开文件")
rprint("mw=====\n",mw.glob_search)
print(mw.glob_search.invoke({"pattern": "**/*.py"}))


rule("场景 B：我想找'内容里写了 redis'的文件，但不知道在哪个文件")
print("→ 用 grep_search(output_mode=files_with_matches)：它只看【内容】，不关心文件名")
print(mw.grep_search.invoke({"pattern": "redis", "output_mode": "files_with_matches"}))


rule("对照：同样是这 4 个文件，两种检索'看到的世界'完全不同")
print("glob 命中（按名字）:", [p for p in mw.glob_search.invoke({"pattern": "**/*"}).split("\n")])
print("grep 命中（按内容）:", [p for p in mw.grep_search.invoke({"pattern": "redis", "output_mode": "files_with_matches"}).split("\n")])


rule("grep content 模式：进一步给出 行号 + 那行原文")
print(mw.grep_search.invoke({"pattern": "redis", "output_mode": "content"}))

shutil.rmtree(SANDBOX)
