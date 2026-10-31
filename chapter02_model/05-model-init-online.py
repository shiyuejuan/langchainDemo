from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

# 加载配置文件
load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 获取大模型
# 说明：原写法读取 CLOSEAI_API_KEY / CLOSEAI_BASE_URL，但 .env 里没有这两个变量，
# 导致 api_key=None 报 openai.OpenAIError: Missing credentials。
# 这里改用本项目已验证可用的 DeepSeek 配置；init_chat_model 的 "openai:" 前缀
# 表示走 OpenAI 兼容协议，配合 base_url 即可调用 DeepSeek。
model = init_chat_model(
    model="openai:deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=30,
    max_retries=1,
)

print(model.invoke("一句话介绍下你自己"))
