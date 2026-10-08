
from typing import Literal
from pydantic import BaseModel,Field


class Message(BaseModel):
    role: Literal["user","assistant","system"] =Field(default="user",description="用户或助手")
    content: str =Field(...,description="消息内容")

class ChatRequest(BaseModel):
    message: str
    thread_id: str =Field(...,description="线程ID")
    user_id: str | None =Field(default=None,description="用户ID，长期记忆按它归档；不传则退化成 thread_id")


class ChatResponse(BaseModel):
    message: str


class HistoryResponse(BaseModel):
    history: list[Message]


