import pathlib
from tkinter import LAST

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from nodes import run_agent_reasoning, tool_node

from langgraph.graph import END, MessagesState, StateGraph

# region: 加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")
# endregion

# region: 定义节点
AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1
# endregion


# region: 定义条件
def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


# endregion

# region: 定义状态图
state_graph = StateGraph(MessagesState)

state_graph.set_entry_point(AGENT_REASON)
state_graph.add_node(AGENT_REASON, run_agent_reasoning)
state_graph.add_node(ACT, tool_node)

state_graph.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})
# endregion

# region: 添加边
state_graph.add_edge(ACT, AGENT_REASON)
# endregion

# region: 编译状态图
graph = state_graph.compile()
graph.get_graph().draw_mermaid_png(output_file_path="mermaid.png")
# endregion

# region: 运行应用
if __name__ == "__main__":
    print("Hello Langgraph Agent!")

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the price of a laptop after applying a gold discount?"
                )
            ]
        }
    )
    print(result["messages"][LAST].content)
# endregion
