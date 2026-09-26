# region: 导入库
import os
import pathlib

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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
    phone: str = Field(description="The phone number of the person")


# endregion

# region: 初始化模型
# 确保使用支持 Tool Calling/Function Calling 的模型版本
llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
    extra_body={"enable_thinking": False},
)
# endregion

# region: 绑定结构化输出
# method="function_calling"，通过工具调用返回结构化输出，更稳定且无需修改 prompt
structured_llm = llm.with_structured_output(ContactInfo, method="function_calling")
# endregion

# region: 执行并获取结果
prompt = "Extract user info from: John Doe, john@example.com, (555) 123-4567"
result = structured_llm.invoke(prompt)
print(result)
# result 现在是一个 ContactInfo 类的实例
print(f"Name: {result.name}")
print(f"Email: {result.email}")
print(f"Phone: {result.phone}")

# endregion
