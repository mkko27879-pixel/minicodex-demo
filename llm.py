
from langchain.chat_models import init_chat_model

from app.core.config import settings





llm=init_chat_model(
    model="deepseek-flash",
    api_key=settings.DEEPSEEK_API_KEY,
)
