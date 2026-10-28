from typing import Any

from ingestion import retriever
from state import GraphState


# region: 向量检索节点
def retrieve(state: GraphState) -> dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


# endregion
