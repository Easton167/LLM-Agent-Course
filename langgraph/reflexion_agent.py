import datetime
import os
import pathlib
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from tavily import TavilyClient

from langgraph.graph import END, START, MessagesState, StateGraph

# region: 加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")
# endregion

# region: 最大迭代次数
MAX_ITERATIONS = 2
# endregion


# region: 定义数据结构
class Reflection(BaseModel):
    """对答案的反思评价"""

    missing: str = Field(description="指出回答中缺失的关键信息")
    superfluous: str = Field(description="指出回答中多余的无关内容")


class AnswerQuestion(BaseModel):
    """回答用户问题"""

    answer: str = Field(description="250字左右的详细回答")
    reflection: Reflection = Field(description="对初始回答的反思与批判")
    search_queries: list[str] = Field(description="1-3个用于优化答案的搜索查询语句")


class ReviseAnswer(AnswerQuestion):
    """修订原始回答"""

    references: list[str] = Field(description="支撑修订答案的参考资料链接")


# endregion

# region: 定义大模型
# 调用通义千问大模型
llm = ChatOpenAI(
    model="qwen3.6-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)
# parser = JsonOutputToolsParser(return_id=True)
# parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion, ReviseAnswer])
# endregion

# region: 定义提示词
# 核心研究员提示模板（中文）
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一名资深行业研究员。
            当前时间：{time}

            工作要求：
            1. {first_instruction}
            2. 严格反思并批判你的回答，力求极致优化
            3. 推荐搜索查询语句，用于补充信息、完善回答""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "严格按照要求的格式回答用户问题"),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

# 答案修订指令（中文核心要求）
revise_instructions = """根据搜索到的新信息修订之前的回答：
    1. 结合反思结果，补充回答中缺失的关键信息
    2. 必须添加数字引用标注，保证内容可验证
    3. 回答底部添加「参考文献」板块（不计入字数），格式：
        - [1] https://example.com
        - [2] https://example.com
    4. 删除冗余内容，严格控制回答字数不超过250字
"""

# 两个派生提示词：只差第一条工作要求
first_responder_prompt = actor_prompt_template.partial(
    first_instruction="提供250字左右的详细回答"
)
revisor_prompt = actor_prompt_template.partial(first_instruction=revise_instructions)
# endregion

# region: 定义链
# 首次回答执行链
first_responder = first_responder_prompt | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="auto"
)

# 答案修订执行链
revisor = revisor_prompt | llm.bind_tools(tools=[ReviseAnswer], tool_choice="auto")
# endregion

# region: 定义搜索工具
tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))


def run_queries(search_queries: list[str], **kwargs):
    """批量执行搜索查询"""
    return tavily_client.batch_search(search_queries)


# 工具节点
execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
# endregion


# region: 定义节点函数
def draft_node(state: MessagesState):
    """生成初始回答"""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def revise_node(state: MessagesState):
    """根据搜索结果修订答案"""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


# endregion


# region: 定义条件判断
def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """根据迭代次数判断是否继续搜索"""
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    num_iterations = count_tool_visits
    if num_iterations >= MAX_ITERATIONS:
        return END
    else:
        return "execute_tools"


# endregion

# region: 定义状态图
builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)

# 构建流程链路
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
# endregion

# region: 编译图
graph = builder.compile()

# 生成工作流流程图
print("=== LangGraph 工作流 Mermaid 图 ===")
graph.get_graph().draw_mermaid_png(output_file_path="reflexion_agent.png")
# endregion

# region: 运行
if __name__ == "__main__":
    print("\n=== 开始执行中文任务 ===")
    res = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请介绍人工智能在智慧医疗领域的应用，并列举国内该领域已获得融资的初创企业",
                }
            ]
        }
    )
    # 输出最终优化后的答案
    last_message = res["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("\n=== 最终回答 ===")
        print(last_message.tool_calls[0]["args"]["answer"])
# endregion
