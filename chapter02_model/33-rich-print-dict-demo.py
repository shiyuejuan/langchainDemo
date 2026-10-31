"""
33 - rich 打印字典（LangChain 1.x 调试工具箱）

演示 rich 打印 dict 的几种姿势 + 常见坑。
环境：rich 15.0.0
关键：长度/深度控制要交给 Pretty(max_string=, max_depth=)，
      Console.print() 本身不带这些参数。
"""
from rich import print as rprint
from rich.pretty import Pretty, pprint
from rich.console import Console

user = {
    "name": "小明",
    "age": 18,
    "active": True,
    "tags": ["vip", "newbie"],
    "profile": {"city": "北京", "level": 3},
}

big = {
    "request": {
        "model": "deepseek-chat",
        "messages": [{"role": "user",
                      "content": "这是一段很长的用户消息，" * 8}],
        "tools": ["get_weather", "search_web"],
    },
    "meta": {"ok": True, "code": 200},
}


def sep(title: str):
    rprint(f"\n[bold cyan]{title}[/bold cyan]")


# ① 对照：裸 print vs rprint
sep("① 对照：内置 print(字典) vs rprint(字典)")
print(user)          # 丑：纯文本、无类型颜色、长内容挤一行
rprint(user)         # 美：key 上色、值按类型上色、自动整理


# ② 函数式 pprint（rich 自带，替代标准库 pprint）
sep("② rich.pretty.pprint(字典)")
pprint(user)


# ③ 窄屏幕自动折行：用 Console(width=...) 控制总宽度
sep("③ Console(width=60).print(Pretty(big)) —— 窄屏自动折行")
Console(width=60).print(Pretty(big))


# ④ 限制字符串长度：Pretty(max_string=20)
sep("④ 限制字符串长度 Pretty(max_string=20)")
rprint(Pretty(big, max_string=20))


# ⑤ 限制嵌套深度：Pretty(max_depth=1) 只展开一层
sep("⑤ 限制嵌套深度 Pretty(max_depth=1)")
rprint(Pretty(big, max_depth=1))


# ⑥ 完整打印（嵌套 + 长字符串，自动换行不乱截断）
sep("⑥ 完整 rprint(big) —— 嵌套字典 + 长字符串")
rprint(big)


# ⑦ 坑：千万别套 f-string
sep("⑦ 坑：rprint(f\"{user}\") 退化成普通字符串，丢失美化")
rprint(f"{user}")

sep("⑧ 正确姿势：rprint(对象本身)，不要 f-string 包")
rprint(user)
