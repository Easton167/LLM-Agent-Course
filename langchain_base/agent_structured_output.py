# region: 导入库
import os
import pathlib

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

# endregion

# region: 加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")

# endregion


# region: 定义数据结构
class ContactInfo(BaseModel):
    """Contact information for a person."""

    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    birthday: str = Field(description="The birthday of the person")


# endregion

# region: 初始化搜索引擎工具
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# endregion


# region: 定义搜索引擎工具
@tool
def tavily_search(query: str) -> str:
    """Search the web for the query"""
    response = tavily_client.search(query)
    return response.get("results")


# endregion

# region: 初始化模型
llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
    extra_body={"enable_thinking": False},
)
# endregion

# region: 工具列表
tools = [tavily_search]
# endregion

# region: 创建Agent
agent = create_agent(model=llm, tools=tools, response_format=ToolStrategy(ContactInfo))
# endregion

# region: 执行Agent
result = agent.invoke({"messages": HumanMessage(content="查询一下迈克尔乔丹的信息")})

print(result["structured_response"])
# endregion
