import os
import pathlib

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# region: 加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")
# endregion

# region: 初始化评分模型
llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
    extra_body={"enable_thinking": False},
)
# endregion


# region: 定义评分结构
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# endregion

# region: 文档相关性评分链
structured_llm_grader = llm.with_structured_output(
    GradeDocuments, method="function_calling"
)

system = (
    "You are a grader assessing relevance of a retrieved document to a user question. "
    "If the document contains keyword(s) or semantic meaning related to the question, "
    "grade it as relevant. "
    "Give a binary score 'yes' or 'no' to indicate whether the document is relevant."
)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
# endregion
