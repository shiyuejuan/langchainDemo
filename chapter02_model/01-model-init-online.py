from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
import os

load_dotenv(override=True)
api_key = os.getenv("DEEPSEEK_API_KEY")
# 注意：.env 里的变量名是 DEEPSEEK_BASE_URL（不是 DEEPSEEK_API_BASE，读错会拿到 None，
# 报 pydantic ValidationError: api_base Input should be a valid string, input value=None）
# 兜底默认官方地址，即使 .env 没配也能跑
api_base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

llm_deepseek = ChatDeepSeek(
    model="deepseek-flash",  # 实测只有 deepseek-flash / deepseek-v4-pro 两个可用模型名
    api_key=api_key,
    api_base=api_base,
)

response = llm_deepseek.invoke("请用一句话介绍你自己")

print(response)