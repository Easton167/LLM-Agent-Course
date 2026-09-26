# region: 导入库
import os
import pathlib

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

# endregion

# region: 获取环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")
# endregion

# region: 初始化Tavily客户端
tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
# endregion


# region: 定义工具
@tool
def search_weather(query: str) -> str:
    """Search the weather for the query"""

    return f"The weather in {query} is sunny"


@tool
def tavily_search(query: str) -> str:
    """Search the web for the query"""
    response = tavily_client.search(query)
    return response.get("results")


# endregion

# region: 初始化LLM
llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# endregion

# region: 创建Agent
tools = [search_weather, tavily_search]
agent = create_agent(model=llm, tools=tools)
# endregion


# region: 主函数
def main():
    response = agent.invoke(
        {
            "messages": HumanMessage(
                content="上海天气如何？同时帮我找两个大模型开发工程师的招聘信息。"
            )
        }
    )
    print(response.get("messages")[-1].content)


# endregion

# region: 运行主函数
if __name__ == "__main__":
    main()
# endregion
