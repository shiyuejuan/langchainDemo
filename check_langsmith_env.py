#!/usr/bin/env python3
"""
LangSmith + 模型通道 自检脚本
用法：
    cd langchain1.2_tutorial
    python chapter03_langsmith/check_langsmith_env.py

它会依次检查：
    1) .env 里 4 个 LANGSMITH_* 变量是否齐全
    2) LangSmith 服务通不通、key 有没有权限（会真的查一次项目列表）
    3) 模型通道（DeepSeek 官网 / CloseAI 中转）能不能正常回话
每一项都会打印 ✅ / ❌，最后给出结论。
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 始终加载项目根目录的 .env（不管从哪个目录运行）
ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
print(f"📄 读取配置：{ENV_PATH}\n")
load_dotenv(ENV_PATH, override=True)


def mask(v):
    if not v:
        return "（空）"
    return v[:4] + "..." + v[-4:] if len(v) > 8 else "***"


ok_count = 0
fail = []

# ---------- 1. 环境变量 ----------
print("=" * 60)
print("第 1 步：检查 LANGSMITH_* 环境变量")
print("=" * 60)
need = ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT"]
for k in need:
    v = os.getenv(k)
    if k == "LANGSMITH_API_KEY":
        print(f"  {k:22} = {mask(v)}")
    else:
        print(f"  {k:22} = {v!r}")
    if not v:
        fail.append(f"{k} 未配置")
if (os.getenv("LANGSMITH_TRACING") or "").lower() != "true":
    fail.append("LANGSMITH_TRACING 不是 true —— 不会上报任何数据")
if not fail:
    ok_count += 1
    print("  ✅ 4 个变量齐全且开关已打开")

# ---------- 2. LangSmith 连通性 ----------
print("\n" + "=" * 60)
print("第 2 步：检查 LangSmith 服务与 API Key 权限")
print("=" * 60)
try:
    from langsmith import Client

    client = Client()
    projects = list(client.list_projects(limit=5))
    print(f"  ✅ 连接成功，账号下已有 {len(projects)} 个项目：")
    for p in projects:
        print(f"     - {p.name}")
    target = os.getenv("LANGSMITH_PROJECT")
    names = [p.name for p in projects]
    if target and target not in names:
        print(f"  ℹ️  配置的项目 {target!r} 还没创建 —— 第一次上报时会自动创建，属正常")
    ok_count += 1
except Exception as e:
    msg = str(e)
    print(f"  ❌ 失败：{type(e).__name__}")
    if "403" in msg:
        print("     原因：API Key 无效 / 已被删除 / 不属于当前工作区")
        print("     解决：登录 https://smith.langchain.com → Settings → API Keys → 重新创建")
    elif "401" in msg:
        print("     原因：未携带或错误的 API Key，检查 .env 的 LANGSMITH_API_KEY")
    else:
        print(f"     原始信息：{msg[:200]}")
    fail.append("LangSmith 连接失败")

# ---------- 3. 模型通道 ----------
print("\n" + "=" * 60)
print("第 3 步：检查模型通道（会真实调用一次，消耗极少 token）")
print("=" * 60)
from langchain.chat_models import init_chat_model

channels = [
    ("DeepSeek 官网", {
        "model": "deepseek-v4-flash", "model_provider": "deepseek",
        "api_key": os.getenv("DEEPSEEK_API_KEY"), "base_url": os.getenv("DEEPSEEK_BASE_URL"),
    }),
]
any_model_ok = False
for label, kw in channels:
    try:
        m = init_chat_model(**kw)
        r = m.invoke("只回复两个字：正常", config={"run_name": f"自检-{label}", "max_tokens": 16})
        print(f"  ✅ {label}：{r.content[:20]!r}")
        any_model_ok = True
    except Exception as e:
        msg = str(e)
        hint = ""
        if "401" in msg or "invalid" in msg.lower() or "无效" in msg:
            hint = "（key 失效，请到对应平台重新生成）"
        print(f"  ❌ {label}：{type(e).__name__}{hint}")
if any_model_ok:
    ok_count += 1
else:
    fail.append("所有模型通道都不可用")

# ---------- 结论 ----------
print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print(f"通过 {ok_count}/3 项检查")
if fail:
    for f in fail:
        print(f"  ⚠️  {f}")
    print("\n处理完以上问题后，重新运行本脚本；全部 ✅ 再去跑 notebook，")
    print("LangSmith 后台才会出现 trace。")
    sys.exit(1)
else:
    print("🎉 一切正常，可以打开 01-langsmith-test.ipynb 开始实验了。")
