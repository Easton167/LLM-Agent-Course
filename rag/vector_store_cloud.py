import os
import pathlib

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# region 加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")
# endregion

if __name__ == "__main__":
    # region 加载文件
    print("正在加载中文文档...")
    document = TextLoader(path / "doc" / "王阳明.txt", encoding="utf-8").load()
    # endregion

    # region 切块
    print("正在切分文档...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    print(f"共生成 {len(texts)} 个文本块")
    docs = [t.page_content for t in texts]
    metas = [t.metadata for t in texts]
    # endregion

    # region 向量化
    # 非 OpenAI 服务商需关闭 check_embedding_ctx_length，避免按 OpenAI token 规则预切
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",
        openai_api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        check_embedding_ctx_length=False,
    )
    print("正在生成向量...")
    # DashScope 兼容接口对 embedding 的 batch size 通常有限制
    vecs = embeddings.embed_documents(docs, chunk_size=10)
    # endregion

    # region 存储
    print("正在写入向量库...")
    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
    )
    try:
        collection = client.get_collection(name="wangyangming")
    except Exception:
        collection = client.create_collection(name="wangyangming")

    collection.add(
        documents=docs,
        embeddings=vecs,
        ids=[str(i) for i in range(len(docs))],
        metadatas=metas,
    )
    print(f"已在 wangyangming 中写入 {len(docs)} 条文档")
    # endregion
