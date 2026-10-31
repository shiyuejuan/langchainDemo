"""
13 - 演示 init_chat_model 的 profile 参数（模型能力档案）

课程原代码用私有网关 gpt-5.4-mini + CLOSEAI_*，这里改为你机器上可用的 DeepSeek。

profile 是什么：
    LangChain 1.x 的 Beta 特性，一个 TypedDict（total=False），描述模型能力，
    字段包括 max_input_tokens / tool_calling / structured_output / temperature ... 等。
    框架在决定策略（例如结构化输出用 function_calling 还是 json_schema）时会参考它。
    - 不传：框架尝试自动推断（按模型名查表 / 调 API）。
    - 手动传：覆盖或明确指定，避免框架误判你的模型能力。
注意：profile 是 total=False 的 TypedDict，任何字段都可能缺失，访问要用 .get()。
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv("/Users/shiyuejuan/Desktop/Demo/langchainDemo/.env", override=True)

custom_profile = {
    "max_input_tokens": 128_000,   # 声明该模型最多支持 12.8 万输入 token
}

# 课程写法（私有网关，你机器上不可用）：
#   model = init_chat_model(
#       model="gpt-5.4-mini", model_provider="openai",
#       profile=custom_profile,
#       api_key=os.getenv("CLOSEAI_API_KEY"),
#       base_url=os.getenv("CLOSEAI_BASE_URL"),
#   )
# 你的 DeepSeek 改造版：
model = init_chat_model(
    model="openai:deepseek-chat",   # DeepSeek 走 OpenAI 兼容协议
    profile=custom_profile,         # 手动声明模型能力档案（经 **kwargs 透传到底层）
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

print("=== profile 内容（total=False，用 .get 访问）===")
print("max_input_tokens:", model.profile.get("max_input_tokens"))
print("整个 profile:", model.profile)

print("\n=== 实跑一次确认模型可用 ===")
resp = model.invoke("一句话介绍下你自己")
print(resp.content)
