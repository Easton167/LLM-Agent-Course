# region：导入库
import os
import pathlib
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
# endregion


# region：定义工具
# ---------- 1. 定义工具（就是普通 Python 函数） ----------
def get_product_price(product: str) -> float:
    """查询商品价格。参数: product(商品名)"""
    return {"laptop": 1200, "mouse": 150, "keyboard": 90}.get(product, 0)


def apply_discount(price: float, tier: str) -> float:
    """根据折扣等级计算最终价格。参数: price(原价), tier(bronze/silver/gold)"""
    rates = {"bronze": 7, "silver": 9, "gold": 8}
    return round(float(price) * (rates.get(tier, 0) / 10), 2)


TOOLS = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}
# endregion

# region：构造 ReAct Prompt
# ---------- 2. 构造 ReAct Prompt（Agent 的"灵魂"全在这里） ----------
TOOL_DESC = "\n".join(f"- {name}: {fn.__doc__}" for name, fn in TOOLS.items())

PROMPT = f"""你可以使用以下工具回答问题：
{TOOL_DESC}

请严格按以下格式输出（每轮一次）：
Thought: 你的思考
Action: 工具名，必须是 [{", ".join(TOOLS)}] 之一
Action Input: 参数，用逗号分隔，例如 laptop 或 1200, gold
Observation: （这一行由系统填充，你不要写）

可以重复多轮 Thought/Action/Action Input/Observation。
当你得到答案后，用：
Thought: 我已经知道答案了
Final Answer: 最终答案

规则：价格必须调用 get_product_price 获取，折扣必须调用 apply_discount，禁止自己算。

Question: {{question}}
Thought:"""
# endregion

# region：LLM 封装
# ---------- 3. LLM 封装 ----------
llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)
# endregion


# region：Agent loop
def run_agent(question: str, max_steps: int = 10) -> str:
    scratchpad = ""  # 记录历史：LLM 输出 + 工具结果，不断追加

    for step in range(1, max_steps + 1):
        print(f"\n--- 第 {step} 轮 ---")

        # (1) 拼接完整 prompt 调用 LLM，用 stop 阻止它自己编 Observation
        full_prompt = PROMPT.format(question=question) + scratchpad
        output = llm.invoke(full_prompt, stop=["\nObservation"]).content
        print(output)

        # (2) 先看有没有最终答案
        if m := re.search(r"Final Answer:\s*(.+)", output):
            return m.group(1).strip()

        # (3) 解析 Action 和 Action Input
        action = re.search(r"Action:\s*(.+)", output)
        action_input = re.search(r"Action Input:\s*(.+)", output)
        if not (action and action_input):
            return "解析失败：LLM 没有按格式输出"

        tool_name = action.group(1).strip()
        # 参数用逗号分割，并去掉可能的 key= 前缀和引号
        args = [
            a.split("=", 1)[-1].strip().strip("'\"")
            for a in action_input.group(1).split(",")
        ]

        # (4) 调用工具拿到 Observation
        if tool_name not in TOOLS:
            observation = f"错误：工具 {tool_name} 不存在"
        else:
            observation = str(TOOLS[tool_name](*args))
        print(f"Observation: {observation}")

        # (5) 把本轮 LLM 输出 + Observation 追加到历史，进入下一轮
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    return "超过最大轮次仍未得到答案"


# endregion

# region：主函数
if __name__ == "__main__":
    answer = run_agent("What is the price of a laptop after applying a gold discount?")
    print(f"\n{'=' * 50}")
    print(f"Final Answer: {answer}")
# endregion
