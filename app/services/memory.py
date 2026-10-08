"""长期记忆：mem0 + pgvector。

三处加固：
  - 改造二：按 (user_id, repo_id) 隔离，A 项目的偏好不串到 B 项目；另留一个全局作用域。
  - 改造一：源头用 custom_instructions 约束提取范围，读取时再用【白名单】兜底。
  - 读取走阈值，避免把不相关的记忆也塞进去。
"""
import hashlib
import re
from pathlib import Path

from mem0 import AsyncMemory
from app.core.config import settings

# mem0 默认把"历史库"放在用户目录（mem0_dir），该位置在受限环境下不可写，
# 会导致 add_memory 抛 "attempt to write a readonly database"。
# 这里显式指向项目内可写目录，保证读写都在仓库内。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DB_PATH = str(DATA_DIR / "mem0_history.db")

# DashScope 的 text-embedding-v4 默认输出 1024 维向量。
# 这里必须让 embedder 的维度、pgvector 建表时的维度保持一致，
# 否则写库会报 "expected 1536 dimensions, not 1024"。
EMBEDDING_DIMS = 1024

# —— 改造二：repo 隔离 ——
# 记忆按 (user_id, repo_id) 隔离：A 项目的偏好不会串到 B 项目。
# 额外留一个 GLOBAL 作用域，装"跨仓库的通用事实"（如"改完必须跑测试"），任何仓库都查得到。
GLOBAL_SCOPE = "global"

# —— 改造三：读取阈值 ——
# pgvector 的 score = 1 - 余弦距离，所以这是"至少要有多像"。
# mem0 默认 0.1，过松，会召回一堆不相关的；这里调到 0.5 起步，按真实数据再校准。
SCORE_THRESHOLD = 0.5


def current_repo_id() -> str:
    """算当前仓库的稳定标识。

    优先用 git remote（换路径克隆也能对上）；本项目没有 remote，
    所以退化为项目根路径的 hash。刻意不带分支——偏好应跨分支共享。
    """
    return hashlib.sha1(str(PROJECT_ROOT).encode("utf-8")).hexdigest()[:12]


# —— 改造一：读取时【白名单】过滤 ——
# 一条记忆必须"长得像"三类允许内容之一才保留，其余一律丢弃。
# 为什么用白名单而不是黑名单：黑名单（列 def/class/.py）会漏掉
# "用户把 max_size 改成了 20" 这种没有代码特征词的过期状态；
# 白名单宁可错杀，也不让易变内容长期留存。代价是句式特殊但合法的记忆会被误丢。
_ALLOWED_MEMORY_PATTERNS = (
    # 主语是"用户"，讲身份 / 开发偏好 / 通用指令
    re.compile(
        r"用户[^。；;\n]{0,20}"
        r"(是|叫|从事|来自|住|姓名|名字|偏好|习惯|喜欢|倾向|常用|爱用|不喜欢|讨厌"
        r"|擅长|熟悉|要求|必须|禁止|不要|务必|需要|约定|规定)"
    ),
    # 主语是"本项目"等，讲环境 / 技术栈事实
    re.compile(
        r"(本项目|这个项目|项目|仓库|工程|环境)[^。；;\n]{0,20}"
        r"(使用|采用|基于|依赖|管理|配置|约定|要求|部署|运行|用)"
    ),
)


def _is_allowed_memory(text: str) -> bool:
    """白名单判定：只有像"用户偏好 / 项目环境 / 通用指令"的记忆才保留。"""
    return any(p.search(text) for p in _ALLOWED_MEMORY_PATTERNS)


config = {
    "history_db_path": HISTORY_DB_PATH,
    # 改造一（源头约束）：mem0 默认用英文 prompt 提取，这里强制中文 + 限定提取范围。
    # 注意：这是 prompt，属于"尽力而为"；读取时还有 _is_allowed_memory 白名单兜底。
    "custom_instructions": (
        "1. 提取结果必须用简体中文，统一以“用户”或“本项目”作主语。"
        "2. 严禁提取任何具体代码结构：不要记录类名、函数名、变量名、文件名、行号或代码片段。"
        "3. 只提取这三类信息："
        "（a）用户的身份与开发偏好，如“用户习惯用 Python 3.10”“用户偏好 pytest”；"
        "（b）环境与项目事实，如“本项目用 Poetry 管理依赖”；"
        "（c）用户的通用指令，如“用户要求改完代码必须跑全量测试”。"
        "4. 不属于以上三类的，一律不提取。"
    ),
    "llm": {
        "provider": "deepseek",
        "config": {
            "model": "deepseek-chat",
            "api_key": settings.DEEPSEEK_API_KEY,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-v4",
            "api_key": settings.DASHSCOPE_API_KEY,
            "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            # 显式指定维度，mem0 才会把 dimensions 参数透传给 DashScope
            "embedding_dims": EMBEDDING_DIMS,
        },
    },
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "collection_name": "my_agent_memory",
            "connection_string": settings.DB_URI,
            "embedding_model_dims": EMBEDDING_DIMS,
        },
    },
}

# 初始化 AsyncMemory 对象
# from_config 是同步的，内部只是解析配置；真正的 add/search 都是协程，需要 await
memory = AsyncMemory.from_config(config)


async def search_memory(user_id: str, query: str, repo_id: str | None = None) -> str:
    """按 user + repo 检索长期记忆，返回可直接拼进提示词的文本；没有命中返回空串。

    检索范围（改造二）：当前仓库的记忆 + 全局记忆；别的仓库的记忆查不到。
    """
    rid = repo_id or current_repo_id()
    filters = {
        "user_id": user_id,
        # 本仓库 + 全局，用 $or 组合；A 项目的记忆不会出现在 B 项目
        "$or": [{"repo_id": rid}, {"repo_id": GLOBAL_SCOPE}],
    }
    results = await memory.search(query=query, filters=filters, threshold=SCORE_THRESHOLD)
    items = [item["memory"] for item in results.get("results", [])]
    # 改造一（读取兜底）：只保留命中白名单的记忆
    kept = [m for m in items if _is_allowed_memory(m)]
    if not kept:
        return ""
    return "；".join(kept)


async def add_memory(user_id: str, messages: list, repo_id: str | None = None):
    """把本轮对话沉淀成长期记忆，按 repo_id 归档（改造二）。"""
    metadata = {"repo_id": repo_id or current_repo_id()}
    await memory.add(messages=messages, user_id=user_id, metadata=metadata)
