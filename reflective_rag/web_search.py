import os
import pathlib
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from state import GraphState
from tavily import TavilyClient

# region: 加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")
# endregion

# region: 初始化搜索客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
# endregion


# region: 网络搜索节点
def web_search(state: GraphState) -> dict[str, Any]:
    """本地文档不够用时，按问题去网上补材料。"""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = list(state.get("documents") or [])

    response = tavily_client.search(query=question, max_results=3)
    results = response.get("results") or []
    joined = "\n\n".join(
        item.get("content", "") for item in results if item.get("content")
    )
    documents.append(Document(page_content=joined, metadata={"source": "tavily"}))
    return {"documents": documents, "question": question}


# endregion
