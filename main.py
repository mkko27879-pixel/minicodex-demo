import asyncio
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.api import api_router
from graph import init_agent, close_agent, get_agent, get_checkpointer


def selector_loop_factory(use_subprocess: bool = False) -> "asyncio.AbstractEventLoop":
    """
    uvicorn 自定义事件循环工厂。

    psycopg 的 async 模式不支持 Windows 默认的 ProactorEventLoop，而 uvicorn
    在 Windows 上**硬编码**返回 ProactorEventLoop（忽略 event loop policy），
    导致连接池建不起连接、启动时 PoolTimeout。这里强制返回 SelectorEventLoop。

    用法：uvicorn --loop main:selector_loop_factory main:app

    注意：uvicorn 对自定义 loop 字符串，会把导入到的对象直接当"零参调用返回
    loop 实例"的工厂，所以这里本身就要返回实例，而不是再返回一个工厂函数。
    """
    return asyncio.SelectorEventLoop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_agent()      # 启动:在事件循环里建连接池 + checkpointer + agent + 建表
    app.state.agent = get_agent()               # 暴露给路由
    app.state.checkpointer = get_checkpointer()
    yield
    await close_agent()
app = FastAPI(lifespan=lifespan)
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
  # 关闭:把连接还回去





app.include_router(api_router, prefix="/api")
