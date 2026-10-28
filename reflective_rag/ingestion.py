import os
import pathlib

from dotenv import load_dotenv

# region: 加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")
# WebBaseLoader 导入时就会检查这个变量
os.environ.setdefault("USER_AGENT", "langchain-agent-course/reflective-rag")
# endregion

import chromadb  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_community.document_loaders import WebBaseLoader  # noqa: E402
from langchain_openai import OpenAIEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

# region: 初始化向量模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False,
    chunk_size=10,
)
# endregion

# region: 连接本地向量库
persist_dir = pathlib.Path(__file__).parent / "chroma_db"
client = chromadb.PersistentClient(path=str(persist_dir))
vectorstore = Chroma(
    client=client,
    collection_name="rag_collection",
    embedding_function=embeddings,
)
# endregion

# region: 入库（集合为空时才抓取、切块、写入）
if vectorstore._collection.count() == 0:
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    print("ingesting data...")
    docs_content = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs_content for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_list)
    doc_splits = [doc for doc in doc_splits if doc.page_content.strip()]

    for doc in doc_splits:
        doc.metadata = {
            key: value
            for key, value in (doc.metadata or {}).items()
            if isinstance(value, (str, int, float, bool))
        }
    vectorstore.add_documents(doc_splits)
    print("ingestion complete")
# endregion

# region: 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# endregion
