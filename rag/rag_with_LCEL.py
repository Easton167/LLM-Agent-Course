import os
import pathlib
from operator import itemgetter

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# region：加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")
# endregion

print("正在初始化组件...")

# region：初始化大模型
llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)
# endregion

# region：初始化向量模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False,
)
# endregion

# region：创建云端 Chroma 客户端
client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)
# endregion

# region：连接云端向量库（指定集合名称）
vectorestore = Chroma(
    client=client,
    collection_name="wangyangming",
    embedding_function=embeddings,
)
# endregion

# region：创建检索器
vector_retriever = vectorestore.as_retriever(search_kwargs={"k": 3})
# endregion

# region：创建中文提示词模板
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严谨的问答助手。请只根据下面的上下文回答问题，不要编造内容。\n\n上下文：\n{context}",
        ),
        ("human", "问题：{question}\n\n请给出详细、准确的中文回答。"),
    ]
)
# endregion


# region：格式化上下文
def format_docs(docs):
    """把检索到的文档拼接成一个上下文字符串。"""
    return "\n\n".join(doc.page_content for doc in docs)


# endregion

# ============================================================================
# 实现 1：不用 LCEL 的简单写法
# ============================================================================


def retrieval_chain_without_lcel(query: str):
    """
    不使用 LCEL 的检索链。

    流程：
    1. 检索相关文档
    2. 格式化上下文
    3. 组装提示词
    4. 调用大模型
    """
    docs = vector_retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content


# ============================================================================
# 实现 2：使用 LCEL 的写法
# ============================================================================


def create_retrieval_chain_with_lcel():
    """
    使用 LCEL 创建检索链。
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | vector_retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


# region: 主函数
if __name__ == "__main__":
    print("开始检索...")

    # 中文问题示例
    query = "王阳明出生于什么时候？"

    result_without_lcel = retrieval_chain_without_lcel(query)

    retrieval_chain = create_retrieval_chain_with_lcel()
    result_with_lcel = retrieval_chain.invoke({"question": query})

    print("\n不用 LCEL 的回答：")
    print(result_without_lcel)

    print("\n使用 LCEL 的回答：")
    print(result_with_lcel)
# endregion
