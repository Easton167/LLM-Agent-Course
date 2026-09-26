# region: 导入库
import os
import pathlib

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# endregion

# region: 加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")
# endregion


# region: 定义工具
@tool
def search_weather(query: str) -> str:
    """Search for weather information related to the query."""
    return f"The weather in {query} is sunny"


# endregion

# region: 初始化模型
llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# endregion

# region: 创建工具列表
tools = [search_weather]
# endregion

# region: 创建Agent对象
agent = create_agent(model=llm, tools=tools)
# endregion


# region: 主函数
def main():
    response = agent.invoke(
        {"messages": HumanMessage(content="What is the weather in shanghai?")}
    )
    print(response["messages"][-1].content)


# endregion

# region: 运行程序
if __name__ == "__main__":
    main()
# endregion
