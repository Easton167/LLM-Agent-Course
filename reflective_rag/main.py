import pathlib

from dotenv import load_dotenv
from graph import app

# region: 加载环境变量
path = pathlib.Path(__file__).parent.parent
load_dotenv(path / ".env")
# endregion

# region: 运行
if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "what is agent memory?"}))
# endregion
