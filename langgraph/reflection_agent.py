import os
import pathlib
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages

from langgraph.graph import END, StateGraph

# region: 加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")

# endregion


# region: 定义状态
class MessageGraph:
    messages: Annotated[list[BaseMessage], add_messages]


# endregion

# region: 定义节点
REFLECT = "reflect"
GENERATE = "generate"
# endregion

# region: 定义提示词
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名小红书科技博主的写作助手，负责写出优秀的小红书笔记。"
            "请根据用户需求生成最好的一版笔记。"
            "如果用户给出了修改意见，请在上一版基础上改写，不要原样重复旧版本。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名小红书爆款内容评审，正在给一篇笔记提修改意见。"
            "请给出具体、可执行的建议，涵盖标题吸引力、开头钩子、篇幅、口语感、互动引导。"
            "只输出修改意见，不要直接重写整篇笔记。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
# endregion

# region: 定义LLM
llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)
# endregion

# region: 定义链
generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
# endregion

# region: 定义节点函数


def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


# endregion


# region: 定义条件判断
def should_continue(state: MessageGraph):
    if len(state["messages"]) > 3:
        return END
    return REFLECT


# endregion

# region: 定义状态图
builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)
builder.add_conditional_edges(
    GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT}
)
# endregion

# region: 添加边
builder.add_edge(REFLECT, GENERATE)
# endregion

# region: 编译图
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="reflection_agent.png")
# endregion

# region: 运行
if __name__ == "__main__":
    print("LangGraph 反思智能体：小红书笔记改写")
    inputs = {
        "messages": [
            HumanMessage(
                content="""把这段草稿改成更适合发小红书的笔记：

                        LangChain 新出的 Tool Calling 真的被低估了。

                        等了很久，终于能用统一方式，在不同模型上做函数调用，写 Agent 简单很多。

                        我根据他们最新的博客做了一期视频。
                        """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)
# endregion
