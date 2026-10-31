import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


llm = ChatOpenAI(
    model="deepseek-v4-flash-0731",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("YT_URL"),
)


def main():

    response = llm.invoke("Hello, 你是谁?")
    print(response.content)


if __name__ == "__main__":
    main()
