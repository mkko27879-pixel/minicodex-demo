from datetime import datetime
from langchain_core.tools import tool



@tool
def get_time() -> str:
    """
    获取当前时间
    """
    now=datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
