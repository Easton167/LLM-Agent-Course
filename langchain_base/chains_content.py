from langchain_core.runnables import RunnableLambda

# 把普通函数变成 Runnable
# 示例1
# upper = RunnableLambda(lambda x: x.upper())

# result = upper.invoke("hello")
# print(result)  # HELLO


# 示例2
step1 = RunnableLambda(lambda x: x.strip())
step2 = RunnableLambda(lambda x: x.lower())
step3 = RunnableLambda(lambda x: x + "!")

pipeline = step1 | step2 | step3

print(pipeline.invoke("  Hello World  "))
