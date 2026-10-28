import pathlib

from chains import generate
from dotenv import load_dotenv
from nodes import grade_documents
from retrieve import retrieve
from state import GraphState
from web_search import web_search

from langgraph.graph import END, StateGraph

# region: 加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")
# endregion

# region: 节点名称
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
WEB_SEARCH = "web_search"
GENERATE = "generate"
# endregion


# region: 条件边：相关文档是否够用来回答
def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print("---DECISION: DOCUMENTS NOT RELEVANT ENOUGH, INCLUDE WEB SEARCH---")
        return WEB_SEARCH
    print("---DECISION: GENERATE---")
    return GENERATE


# endregion

# region: 组装状态图
builder = StateGraph(GraphState)
builder.add_node(RETRIEVE, retrieve)
builder.add_node(GRADE_DOCUMENTS, grade_documents)
builder.add_node(WEB_SEARCH, web_search)
builder.add_node(GENERATE, generate)

builder.set_entry_point(RETRIEVE)
builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)
builder.add_edge(WEB_SEARCH, GENERATE)
builder.add_edge(GENERATE, END)

builder.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE,
    },
)
# endregion

# region: 编译图
app = builder.compile()
print(app.get_graph().draw_mermaid())
# endregion
