import os
import pathlib

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 自动找到项目根目录
path = pathlib.Path(__file__).parent.parent

# 加载 .env
load_dotenv(path / ".env")

llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.0,
)

system_msg = SystemMessage("你是一个产品运营AI助手.")
human_msg = HumanMessage("你好，你是谁?")

messages = [system_msg, human_msg]
response = llm.invoke(messages)  # Returns AIMessage
print(response.content)
