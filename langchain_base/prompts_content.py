import os
import pathlib

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# 获取项目目录
path = pathlib.Path(__file__).parent.parent

# 加载 .env 文件
load_dotenv(dotenv_path=path / ".env")

# 使用 MessagesPlaceholder 预留动态位置
# 这样可以在运行时传入历史记录或额外的动态提示列表
template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个产品运营专家，名字是 {name}。"),
        ("system", "当前策略重点：{strategy_focus}。"),
        MessagesPlaceholder(variable_name="history"),  # 动态插入历史或特定消息
        ("human", "{user_input}"),
    ]
)

llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)

chain = template | llm

# 动态传入上下文参数
response = chain.invoke(
    {
        "name": "张三",
        "strategy_focus": "强调节能环保，受众为年轻群体",
        "history": [HumanMessage(content="之前提到过吹风机非常耐用。")],
        "user_input": "请为吹风机写一段卖点描述，限20字内。",
    }
)

# 传入的提示词内容
"""
System：
你是一个产品运营专家，
名字是 张三。
------------------------
System：
当前策略重点：
强调节能环保，
受众为年轻群体。
------------------------
Human：
之前提到过吹风机非常耐用。
------------------------
Human：
请为吹风机写一段卖点描述，
限20字内。
"""

print(response.content)
