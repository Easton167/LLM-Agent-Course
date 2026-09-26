import os
import pathlib

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 自动找到项目根目录
path = pathlib.Path(__file__).parent.parent

# 加载 .env
load_dotenv(path / ".env")


def main():
    print("Hello，langchain-course!")

    # 准备数据
    information = """
        王阳明（王守仁）
        一、基本生平
        王守仁（1472—1529），字伯安，号阳明，浙江余姚人，明代顶级思想家、军事家、教育家，陆王心学集大成者，后世称阳明先生。 因曾筑室于会稽山阳明洞，自号阳明子；官至南京兵部尚书，封新建伯，谥号文成，又称王文成公。
        关键人生节点
        少年立志：年少便不满足科举辞章，立志做 “第一等事 —— 读书做圣贤”。
        格竹悟道：年轻时笃信朱熹理学，对着竹子 “格物穷理” 七天，毫无所得，埋下心学伏笔。
        龙场悟道（心学诞生） 因上书得罪宦官刘瑾，被贬贵州龙场驿。蛮荒困苦中一夜顿悟：心即理，万事真理不在外物，而在本心。
        平乱立功
        平定江西匪患；
        35 天平定宁王朱宸濠叛乱（千古奇功）；
        广西平定思田土司之乱、八寨盗贼。 文臣带兵，百战百胜，是中国少有的 “立德、立功、立言” 三不朽圣人。
        晚年讲学：随处办学，传播心学，57 岁病逝于归途。
        二、心学四大核心思想
        1. 心即理
        万事万物的道理，不在外界事物里，而在人的本心。 不必向外苦苦求索天理，反观内心就能找到善恶、是非标准。
        2. 知行合一
        知和行本是一体，知而不行等于未知。
        只懂道理不去做，不算真正明白；
        真正明白道理，自然会付诸行动。 打破当时读书人 “空谈义理、脱离实践” 的弊病。
        3. 致良知
        良知是人天生自带的是非之心，人人皆有，不分圣贤凡夫。致良知就是时时反省、去除私欲，让本心良知完全显露，依良知行事便是圣人之道。 阳明晚年将所有学问浓缩为这三个字。
        4. 存天理，去人欲（新解读）
        不同于朱熹 “灭人欲”，王阳明不否定正常欲望，只去除过度、自私、贪婪的私欲，守住本心良知。
        三、四句教（心学总纲）
        晚年对弟子总结，为王门核心教法：
        四、主要著作
        《传习录》：弟子记录其讲学语录，心学第一经典；
        《王文成公全书》：收录奏疏、书信、诗文；
        《大学问》：以心学重新解读《大学》。
    """

    # 使用 ChatPromptTemplate 定义角色和任务
    summary_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位专业的历史文学顾问，请根据用户提供的资料进行整理。"),
            (
                "human",
                (
                    f"请根据以下资料：\n{information}\n\n"
                    "请完成以下任务：\n"
                    "1. 撰写一段简短的生平总结。\n"
                    "2. 列出两个关于此人的有趣轶事。"
                ),
            ),
        ]
    )

    # 初始化大模型
    llm = ChatOpenAI(
        model="qwen3.6-flash",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )

    # 构建链
    chain = summary_prompt_template | llm

    # 执行链
    response = chain.invoke(input={"information": information})
    # information->ChatPromptTemplate->Messages->ChatModel->AIMessage

    print(response.content)

    # 获取 token 使用量（注意：不同模型供应商的返回结构可能不同）
    usage = response.response_metadata.get("token_usage", {})
    print(f"\nTotal Tokens: {usage.get('total_tokens', '未知')}")


if __name__ == "__main__":
    main()
