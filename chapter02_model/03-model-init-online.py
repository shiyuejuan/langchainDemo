from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv(override=True)
api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE")

llm_deepseek = ChatOpenAI(
    model="deepseek-flash",  # 实测只有 deepseek-flash / deepseek-v4-pro 两个可用模型名
    api_key=api_key,
    base_url=api_base,
)

response = llm_deepseek.invoke("1 + 1 = ?")

print(response)