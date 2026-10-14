from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
import os

load_dotenv(override=True)
api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_BASE_URL")

llm_deepseek = ChatDeepSeek(
    model="deepseek-flash",  # 实测只有 deepseek-flash / deepseek-v4-pro 两个可用模型名
    api_key=api_key,
    api_base=api_base,
)

response = llm_deepseek.invoke("请用一句话介绍你自己")

print(response)