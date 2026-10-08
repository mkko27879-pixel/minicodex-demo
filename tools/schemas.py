from pydantic import BaseModel, ConfigDict, Field
class ReadFileInput(BaseModel):
    """读取单个文本文件的内容，可按行号范围截取。"""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1, description="文件路径，相对工作区根目录，例如 'app/core/config.py'。禁止绝对路径")
    start_line: int | None = Field(default=1, ge=1, description="起始行号（从 1 开始，含）。默认 1，即从文件开头读")
    end_line: int | None = Field(default=None, ge=1, description="结束行号（含）。不传则读到文件末尾")


class ListFilesInput(BaseModel):
    """列出某个目录下的文件和子目录。"""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(default=".", description="要列出的目录，相对工作区根目录，默认 '.' 表示工作区根")


class SearchInput(BaseModel):
    """按关键词或正则在目录内搜索文本，返回命中的文件与行号。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="搜索关键词；默认按字面量匹配，regex=true 时才当正则")
    path: str = Field(default=".", description="搜索起始目录，相对工作区根目录，默认 '.'")
    glob: str | None = Field(default=None, description="按文件名过滤，例如 '*.py'。不传则搜索全部文本文件")
    max_results: int = Field(default=10, ge=1, le=500, description="最多返回多少条命中，默认 10，避免刷屏")
    regex: bool = Field(default=False, description="是否把 query 当正则表达式。默认 False，按普通文本搜索")


class PatchEdit(BaseModel):
    """一处精确替换：把 old_text 换成 new_text。"""

    model_config = ConfigDict(extra="forbid")

    old_text: str = Field(..., min_length=1, description="要被替换的原文，需能在文件中匹配到（匹配允许忽略首尾空白差异）")
    new_text: str = Field(..., description="替换后的新内容；传空字符串表示删除这段原文")


class ApplyPatchInput(BaseModel):
    """对文件做替换式编辑（apply_patch）。一处或多处替换按顺序应用。"""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1, description="要修改的文件路径，相对工作区根目录")
    edits: list[PatchEdit] = Field(..., min_length=1, description="按顺序应用的一处或多处替换")


class RunCommandInput(BaseModel):
    """在工作区执行命令并返回输出。"""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(..., min_length=1, description="要执行的命令。出于安全考虑，当前仅允许以 'pytest' 开头的测试命令")


class SearchMemoryInput(BaseModel):
    """检索长期记忆（跨会话记住的用户/项目事实）。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="要检索的内容，例如“用户偏好用什么测试框架”")
