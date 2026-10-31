"""
20 - 详解 PIIMiddleware 的 detector 参数（纯 Python，不调模型）

detector 回答的问题是："中间件怎么知道哪一段是敏感信息？"
它有两种写法：
  A. 正则字符串：detector=r"sk-[a-zA-Z0-9]+"
  B. 函数：返回一个 list[PIIMatch]，每个元素是 {type, value, start, end}
另外：当传入内置类型（email/credit_card/ip/mac_address/url）且 detector=None 时，
     用中间件自带的检测器。
"""
import re
from langchain.agents.middleware.pii import PIIMatch

# ===== 形式 B：自定义函数 detector（课程的 detect_phone_number 就是这种）=====
def detect_phone_number(text: str) -> list[PIIMatch]:
    # 11 位手机号：1 开头，第二位是 3-9，后面 9 位数字
    return [
        {"type": "phone_number", "value": m.group(), "start": m.start(), "end": m.end()}
        for m in re.finditer(r"1[3-9]\d{9}", text)
    ]

sample = "联系我：手机 13812345678，备用 15900001234。"

print("=== 形式 B：函数 detector 返回的原始结构 ===")
matches = detect_phone_number(sample)
print(matches)
print("类型：", type(matches), "，每个元素是 dict，含 type / value / start / end")
print("start/end 是'在第几个字符到第几个字符'，中间件靠它精准替换文本")

# ===== 形式 A：正则字符串 detector（课程的 api_key 就是这种）=====
pattern = r"sk-[a-zA-Z0-9]+"
print("\n=== 形式 A：正则字符串 detector ===")
print("正则", pattern, "在文本里匹配到的片段：")
for m in re.finditer(pattern, "key 是 sk-abc123XYZ456，另一个 sk-SECRET789"):
    print(f"  value={m.group()}  start={m.start()}  end={m.end()}")

# ===== detector=None + 内置类型：用中间件自带检测器 =====
print("\n=== detector=None + 内置类型（email）===")
print("此时中间件用自己的邮箱识别规则，无需你写正则/函数")
print("例：user@example.com 会被内置检测器命中并按 strategy 处理")
print("内置类型只有 5 种：email / credit_card / ip / mac_address / url")
print("phone_number 不在这 5 种里 → 必须自己写 detector（形式 A 或 B）")
