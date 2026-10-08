from typing import AsyncIterable

from fastapi import APIRouter, Depends, Query, Request
from langchain_core.messages import HumanMessage

from app.services.memory import add_memory, current_repo_id
from schemas.base import ChatRequest, ChatResponse, HistoryResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

router=APIRouter()


# 这两个 provider 是"依赖注入"的入口：agent / checkpointer 只在 lifespan 的
# async 上下文里构造（见 graph.init_agent），不能模块级导入，只能在请求期从
# app.state 取。好处是路由签名不再暴露 Request，测试也能用
# app.dependency_overrides 直接替换成假对象。
def get_agent(request: Request):
    return request.app.state.agent


def get_checkpointer(request: Request):
    return request.app.state.checkpointer






@router.post("/chat")
async def get_req(request: ChatRequest, agent=Depends(get_agent)):
    # 长期记忆已做成工具（recall_memory），由模型按需检索，这里不再自动注入系统消息；
    # 只把 user_id / repo_id 放进 config，工具从那里取（改造二/三）。
    user_id=request.user_id or request.thread_id
    repo_id=current_repo_id()
    config={"configurable":{"thread_id":request.thread_id,"user_id":user_id,"repo_id":repo_id}}
    # 只传本轮新消息，历史由 checkpointer 按 thread_id 维护
    res= await agent.ainvoke({"messages":[HumanMessage(content=request.message)]},config=config)
    ai_rely=res["messages"][-1].content
    await add_memory(user_id,[
        {"role":"user","content":request.message},
        {"role":"assistant","content":ai_rely}
    ],repo_id=repo_id)
    return ChatResponse(message=ai_rely)




@router.post("/chat/stream", response_class=EventSourceResponse)
async def stream_chat(request: ChatRequest, agent=Depends(get_agent)) -> AsyncIterable[ServerSentEvent]:
    """
    流式接口（支持SSE）
    """
    # 同上：记忆检索交给 recall_memory 工具，config 里只带 user_id / repo_id
    user_id=request.user_id or request.thread_id
    repo_id=current_repo_id()
    config={"configurable":{"thread_id":request.thread_id,"user_id":user_id,"repo_id":repo_id}}

    # 收集本轮回复，流结束后沉淀到长期记忆
    collected=[]
    try:
        async for msg_chunk, metadata in agent.astream(
                {"messages":[HumanMessage(content=request.message)]}, config=config, stream_mode="messages"
        ):
            if metadata.get("langgraph_node") == "model" and msg_chunk.content:
                collected.append(msg_chunk.content)
                yield ServerSentEvent(event="token", data=msg_chunk.content)
        # 写长期记忆，放在 done 之前且被 try 包住，写库失败不会影响已输出的回复
        await add_memory(user_id,[
            {"role":"user","content":request.message},
            {"role":"assistant","content":"".join(collected)}
        ],repo_id=repo_id)
        yield ServerSentEvent(event="done", raw_data="[DONE]")
    except Exception as e:
        print(e)
        yield ServerSentEvent(event="error", data=str(e))
        yield ServerSentEvent(event="done", raw_data="[DONE]")



@router.get("/messages")
async def get_history(thread_id: str =Query(...), agent=Depends(get_agent))->HistoryResponse:
    config={"configurable":{"thread_id":thread_id}}
    state= await agent.aget_state(config)
    if not state or not state.values.get("messages"):
        return HistoryResponse(history=[])
    messages=state.values["messages"]
    history_list=[]
    for mes in messages:
        if mes.type =="tool":
            continue
        # 跳过摘要中间件写回 state 的占位消息
        if mes.additional_kwargs.get("lc_source") == "summarization":
            continue
        # 跳过注入的长期记忆（它被当作前置 SystemMessage 写进了 state，不是真实对话）
        if mes.type=="system" and str(mes.content).startswith("关于当前用户的已知信息"):
            continue
        role="assistant" if mes.type=="ai" else ("user" if mes.type=="human" else "system")
        history_list.append({"role":role,"content":mes.content})

    return HistoryResponse(history=history_list)


@router.delete("/messages")
async def delete_history(thread_id: str=Query(...), checkpointer=Depends(get_checkpointer)):
    await checkpointer.adelete_thread(thread_id)
    return {"state":"success","message":f"历史记录 {thread_id} 已删除"}



