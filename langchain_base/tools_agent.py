import os
import pathlib

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI

path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")


@tool
def search_weather(query: str) -> str:
    """Search the web for information about the query"""
    return f"The weather in {query} is sunny"


llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

tools = [search_weather]

agent = create_agent(model=llm, tools=tools)


def main():
    response = agent.invoke(
        {"messages": HumanMessage(content="What is the weather in beijing?")}
    )
    print(response)


if __name__ == "__main__":
    main()
