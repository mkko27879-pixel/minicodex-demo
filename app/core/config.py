from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 定义环境变量的名称
    DB_URI: str
    DASHSCOPE_API_KEY: str
    DEEPSEEK_API_KEY: str
    #如何读取文件中的环境变量
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,# 忽略大小写

    )
settings=Settings()
