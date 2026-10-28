from typing import Any, TypedDict


# region: 图状态
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: retrieved documents
    """

    question: str
    generation: str
    web_search: bool
    documents: list[Any]


# endregion
