import os
import pathlib

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from langgraph.graph import MessagesState

# region: 加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")
# endregion

# region: 初始化大模型
llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)
# endregion

# region: 系统消息
SYSYEM_MESSAGE = """
你是一位乐于助人的助手，能够利用工具来回答问题。
"""
# endregion

# region: 工具函数


@tool
def get_product_price(product: str) -> float:
    """查询商品价格。参数: product(商品名)."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1200, "mouse": 150, "keyboard": 90}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """根据折扣等级计算最终价格。参数: price(原价), discount_tier(bronze/silver/gold)"""
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 7, "silver": 9, "gold": 8}
    discount = discount_percentages.get(discount_tier, 0)
    return round(float(price) * (discount / 10), 2)


# endregion

# region: 为大模型绑定工具
tools = [get_product_price, apply_discount]
llm_with_tools = llm.bind_tools(tools)
# endregion


# region: 运行推理节点
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    运行推理节点。
    """
    response = llm_with_tools.invoke(
        [{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]]
    )
    return {"messages": [response]}


# endregion

# region: 工具节点
tool_node = ToolNode(tools)
# endregion
