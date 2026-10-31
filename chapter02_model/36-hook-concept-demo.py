"""
36 - 什么是 hook（钩子函数）？用 20 行迷你框架看穿本质

钩子三要素：
  ① 框架在主流程里预留"空位"（时机）
  ② 你在子类里重写填入自己的逻辑
  ③ 你从不手动调用它——框架跑到那个时机自动调

这正是 AgentMiddleware 的 before_model / wrap_model_call 等的工作方式。
"""
import io
import contextlib


class MiniAgent:
    """迷你框架：主流程 run() 是框架写死的，钩子是留给你的空位。"""

    def before_run(self):     # ← 钩子：默认空实现（你不用管就等于没挂）
        pass

    def after_run(self):      # ← 钩子
        pass

    def run(self):            # ← 主流程：这才是"调用者"，你动不了它
        self.before_run()     # 时机① 框架自动调你的钩子
        print("  主流程：干活中……")
        self.after_run()      # 时机② 框架自动调你的钩子


class MyAgent(MiniAgent):
    """你只需要继承 + 重写钩子，主流程一行没改。"""

    def before_run(self):
        print("  [钩子] 开工前：检查工具箱")

    def after_run(self):
        print("  [钩子] 收工后：写工作日志")


class LazyAgent(MiniAgent):
    """不重写任何钩子 → 主流程照常跑，钩子位置等于不存在。"""
    pass


if __name__ == "__main__":
    print("① 挂了钩子的子类（框架自动在时机点调你的实现）：")
    MyAgent().run()

    print("\n② 什么都没重写的子类（空钩子 = 无感，主流程不受影响）：")
    LazyAgent().run()
