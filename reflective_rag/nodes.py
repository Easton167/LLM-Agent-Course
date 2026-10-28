from typing import Any

from retrieval_grader import retrieval_grader
from state import GraphState


# region: 文档相关性过滤节点
def grade_documents(state: GraphState) -> dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    If any document is not relevant, or nothing relevant remains, run web search.
    """
    print("check documents relevance")
    question = state["question"]
    documents = state["documents"]

    filtered_documents = []
    web_search = False

    for document in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": document.page_content}
        )
        grade = (getattr(score, "binary_score", None) or "").strip().lower()
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_documents.append(document)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True

    if not filtered_documents:
        web_search = True

    return {
        "documents": filtered_documents,
        "question": question,
        "web_search": web_search,
    }


# endregion
