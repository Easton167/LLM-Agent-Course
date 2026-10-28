import os
import pathlib

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# region: 加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")
# endregion

# region: 初始化回答模型
llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
    extra_body={"enable_thinking": False},
)
# endregion

# region: 回答链
template = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Context: {context}\n"
    "\n"
    "Answer:"
)

prompt = ChatPromptTemplate.from_template(template)

generation_chain = prompt | llm | StrOutputParser()
# endregion


# region: 生成回答节点
def generate(state):
    """根据问题和检索到的文档生成回答。"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join([doc.page_content for doc in documents])
    generation = generation_chain.invoke({"question": question, "context": context})
    return {"generation": generation}


# endregion
