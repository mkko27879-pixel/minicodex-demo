"""长期记忆工具（改造三）：让模型按需检索跨会话记住的用户/项目事实。

为什么做成工具而不是每轮自动注入：
  - 省 token：只在模型觉得需要时才查，不做无差别注入；
  - 不污染短期历史：工具结果进的是 ToolMessage，不会像前置 SystemMessage 那样
    被 checkpointer 写进 thread 历史、越聊越多。
"""
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.services.memory import search_memory
from tools.schemas import SearchMemoryInput


@tool(args_schema=SearchMemoryInput)
async def recall_memory(query: str, config: RunnableConfig) -> str:
    """检索长期记忆：用户的身份与开发偏好、项目环境事实、用户的通用指令。

    当你不确定用户之前是否交代过偏好或要求时（比如该用什么测试框架、
    有没有约定必须跑什么命令），先用这个工具查一下，不要凭空猜。
    """
    # user_id / repo_id 由服务端通过 config 注入；模型看不到、也不需要知道。
    # （RunnableConfig 是 LangChain 的注入参数，不会出现在工具入参 schema 里。）
    cfg = config.get("configurable", {}) or {}
    user_id = cfg.get("user_id")
    if not user_id:
        return "当前会话没有 user_id，无法检索长期记忆"
    text = await search_memory(user_id=user_id, query=query, repo_id=cfg.get("repo_id"))
    return text or "没有找到相关的长期记忆"
