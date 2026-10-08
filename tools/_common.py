"""工具共用的工作区路径与“垃圾文件”过滤规则。

集中放在这里，避免每个工具各写一份、彼此还不一致（C5）。
"""
from pathlib import Path

# 工作区根目录 = 项目根（tools/ 的上一级）。
# 用 __file__ 推导，而不是 os.getcwd()——后者的值取决于进程从哪个目录启动，不可靠。
WORKDIR = Path(__file__).resolve().parent.parent

# 遍历/搜索时跳过的目录：体积大、与源码无关，或存放敏感/运行期数据
JUNK_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".idea", ".vscode", "node_modules", "dist", "build", "data",
}

# 遍历/搜索时跳过的文件：以密钥文件为主，避免把 API Key 泄进结果
JUNK_FILES = {".env", ".env.local", ".env.development", ".env.production", ".DS_Store"}

# 二进制/非文本后缀：读进来没意义，还会解码报错
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".tar", ".gz",
    ".exe", ".dll", ".so", ".pyc", ".bin", ".db", ".sqlite",
    ".xlsx", ".xls", ".docx", ".pptx", ".woff", ".woff2", ".ttf",
}


def resolve_in_workdir(path: str) -> Path:
    """把相对路径解析为工作区内的绝对路径，并确保没有越界。

    - 拒绝绝对路径（统一要求相对工作区根目录）；
    - 用 resolve() 展开 ".."，再做前缀校验，防止 path traversal。
    越界时抛 ValueError，由调用方转成给模型看的错误文本。
    """
    if not path or not path.strip():
        raise ValueError("路径不能为空")
    raw = Path(path.strip())
    if raw.is_absolute():
        raise ValueError("不接受绝对路径，请给相对工作区根目录的路径")
    target = (WORKDIR / raw).resolve()
    try:
        target.relative_to(WORKDIR)
    except ValueError:
        raise ValueError(f"路径越界，只能访问工作区内：{path}") from None
    return target


def is_junk(name: str, is_dir: bool) -> bool:
    """判断一个条目是否该被遍历/搜索跳过。

    规则：目录命中 JUNK_DIRS（含 *.egg-info）；文件命中 JUNK_FILES 或二进制后缀。
    """
    if is_dir:
        return name in JUNK_DIRS or name.endswith(".egg-info")
    if name in JUNK_FILES:
        return True
    return Path(name).suffix.lower() in BINARY_SUFFIXES
