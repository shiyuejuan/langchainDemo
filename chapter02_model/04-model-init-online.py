"""
04-model-init-online.py
第一次用「在线大模型」做 invoke。

为什么改过：原脚本用 OpenRouter 的免费模型（nvidia/nemotron-3.5-lightning:free），
但本账号 OpenRouter 返回 402（无额度），且免费模型高峰期经常 429 限流、部分模型
长时间挂起 → 表现就是「没输出」或抛 TimeoutError / RateLimitError。
DeepSeek 官方接口在本项目已验证可用（约 1.6s 返回，且自动上报 LangSmith 链路），
因此默认走 DeepSeek。OpenRouter 的写法保留在文件底部，充值后可切换。
"""
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
import os

load_dotenv(override=True)


def build_llm():
    # —— 主用：DeepSeek 官方（本项目已验证可用，自动上报 LangSmith）——
    return ChatDeepSeek(
        model="deepseek-chat",                    # 对话模型；推理模型用 "deepseek-reasoner"
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),  # .env 里是 https://api.deepseek.com
        # 防止「卡住没输出」：30 秒超时 + 只重试 1 次，快速失败并打印原因
        timeout=30,
        max_retries=1,
    )


llm = build_llm()
print(">>> 正在调用 DeepSeek（自动上报 LangSmith，控制台会打印链路链接）...", flush=True)

try:
    response = llm.invoke("1 + 1 = ?")
    print(response)
except Exception as e:
    # 友好报错：不再丢一堆 traceback，直接说清原因
    print(f"\n调用失败：{type(e).__name__}")
    print(f"  原因：{e}")
    print("  排查清单：")
    print("  1) .env 里 DEEPSEEK_API_KEY 是否复制完整、没有多余换行；")
    print("  2) 网络能否访问 api.deepseek.com；")
    print("  3) 模型名是否有效（deepseek-chat / deepseek-reasoner）。")


# ============================================================================
# 备用方案：OpenRouter（一个 key 调 462 个模型，但本账号 402 无额度，先注释）
# 充值后：把上面 build_llm() 改成返回下面这个即可。
# ----------------------------------------------------------------------------
# from langchain_openrouter import ChatOpenRouter
#
# def build_llm():
#     return ChatOpenRouter(
#         # 免费模型（不用充值；高峰可能 429，重跑即可）：
#         #   "nvidia/nemotron-3.5-lightning:free"
#         #   "google/gemma-4-31b-it:free"
#         # 付费模型（需先在 openrouter.ai/credits 充值）：
#         #   "deepseek/deepseek-v4.1-flash"
#         #   "openai/gpt-4o-mini"
#         model="nvidia/nemotron-3.5-lightning:free",
#         api_key=os.getenv("OPENROUTER_API_KEY"),
#         # app_title 必须显式关掉：它的默认值会让库传一个 SDK 不认的 x_title 参数，
#         # 直接报 TypeError: unexpected keyword argument 'x_title'（第三方包兼容 bug）
#         app_title=None,
#         timeout=30,
#         max_retries=1,
#     )
# ============================================================================
