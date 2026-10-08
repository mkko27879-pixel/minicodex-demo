from pathlib import Path
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.core.config import settings
from llm import llm
from tools import apply_patch, get_time, list_files, read_file, recall_memory, run, search


DB_URI=settings.DB_URI
prompt=(Path(__file__).parent/"system.md").read_text(encoding="utf-8")

SUMMARY_PROMPT="""你是一个对话摘要助手。请阅读以下对话历史，并按照以下要求进行精简总结：
                            1. 【用户画像】：用户的姓名、喜好、身份等基本信息。
                            2. 【当前任务】：用户正在咨询或解决的具体问题。
                            3. 【已达成的共识】：AI 已经给出的结论或确认的事实。
                            4. 【待办事项】：还未解决的问题。
                            
                            要求：
                            - 严格控制字数在 300 字以内。
                            - 使用第三人称（如“用户说”、“AI回答”）。
                            - 绝对不允许丢失上述4个维度的关键信息。
                            对话历史：
                            
{messages}
                            </messages>"""

# 注意：pool / checkpointer / agent 都不能在模块顶层构造。
# AsyncPostgresSaver.__init__ 内部会调用 asyncio.get_running_loop()，
# 在导入期（没有运行中的事件循环）构造会直接抛 RuntimeError。
# 因此统一在 async 的 startup 阶段构造，并由 main.py 放进 app.state 供路由取用。
_pool=None
_checkpointer=None
_agent=None


async def init_agent():
    """
    在运行中的事件循环里构建连接池、checkpointer 和 agent，并建表。
    只能在应用 startup（async 上下文）里调用。
    """
    global _pool,_checkpointer,_agent
    # 延迟导入，保证 graph 模块被导入时不触发任何连接/构造函数
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    #异步连接池
    _pool=AsyncConnectionPool(
            conninfo=DB_URI,
            open=False,
            max_size=10,
            kwargs={"autocommit":True,"row_factory":dict_row}
        )
    await _pool.open()
    _checkpointer=AsyncPostgresSaver(_pool)
    await _checkpointer.setup() #建表


    _agent=create_agent(
        model=llm,
        tools=[read_file, list_files, search, apply_patch, run, recall_memory, get_time],
        checkpointer=_checkpointer,
        system_prompt=prompt,
        middleware=[
            SummarizationMiddleware(
                model=llm,
                trigger=("tokens",4000),  # 在 4000 个 token 时触发摘要
                keep=("messages",20),  # 摘要后保留最近 20 条消息
                summary_prompt=SUMMARY_PROMPT,  # 可选
            )]
    )


async def close_agent():
    """关闭连接池，把连接还回去"""
    global _pool
    if _pool is not None:
        await _pool.close()


def get_agent():
    """取当前 agent 实例（供 app.state 使用）"""
    return _agent


def get_checkpointer():
    """取当前 checkpointer 实例（供 app.state 使用）"""
    return _checkpointer











if __name__ == "__main__":
    print("-----------")
    # 用真实连接跑一次独立冒烟：init_agent 必须在事件循环里调用
    import asyncio

    async def _run():
        await init_agent()
        try:
            config={"configurable":{"thread_id":"test_user_id"}}
            graph_input={"messages":[{"role":"user","content":"现在时间是多少？"}]}
            async for chunk in get_agent().astream(graph_input,config=config,stream_mode="messages"):
                msg_chunk,metadata=chunk
                if metadata.get("langgraph_node")=="model" and msg_chunk.content:
                    print(msg_chunk.content,end="")
        finally:
            await close_agent()

    asyncio.run(_run())
