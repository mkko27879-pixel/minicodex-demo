"""search 工具：在工作区内搜索文本（默认按字面量，可选正则）。"""
import os
import re
from pathlib import Path

from langchain_core.tools import tool

from tools._common import WORKDIR, is_junk, resolve_in_workdir
from tools.schemas import SearchInput


def _iter_files(root: Path):
    """递归遍历 root 下的文本文件，跳过垃圾目录与垃圾文件。"""
    # path 直接指向某个文件时，就只搜这一个
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        # 就地裁剪 dirnames 实现剪枝：命中的目录不再往下走，比全遍历再过滤快
        dirnames[:] = [d for d in dirnames if not is_junk(d, True)]
        for name in filenames:
            if is_junk(name, False):
                continue
            yield Path(dirpath) / name


# 搜索工具(全局搜索)
@tool(args_schema=SearchInput)
def search(query: str, path: str = ".", glob: str | None = None,
           max_results: int = 10, regex: bool = False) -> str:
    """按关键词（默认字面量，可选正则）搜索文本，返回命中的「文件:行号: 内容」。"""
    # —— 安全校验 ——
    try:
        target = resolve_in_workdir(path)
    except ValueError as e:
        return f"搜索失败：{e}"

    if not target.exists():
        return f"目录不存在：{path}"

    # —— 编译匹配模式 ——
    if regex:
        # 显式要求正则时才当正则；写错就明确报错，别静默搜不到
        try:
            pattern = re.compile(query)
        except re.error as e:
            return f"正则表达式不合法：{e}"
    else:
        # 默认按字面量匹配（re.escape）：避免 '.'、'(' 等被当正则，导致误匹配或直接报错（A2）
        pattern = re.compile(re.escape(query))

    # —— 搜索截断：命中数达到 max_results 就停，如果太多则只取前若干条 ——
    hits = []
    for file in _iter_files(target):
        if glob and not file.match(glob):     # 按文件名过滤，如 '*.py'
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except Exception:
            continue                          # 二进制 / 编码异常，跳过
        # 相对工作区的路径，模型拿到后能直接喂给 read_file
        shown = file.relative_to(WORKDIR).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{shown}:{lineno}: {line.strip()}")
                if len(hits) >= max_results:
                    return "\n".join(hits) + f"\n...（已达上限 {max_results} 条，可能还有更多，请缩小范围）"

    # —— 返回搜索结果：没有命中给提示，有命中直接返回 ——
    return "\n".join(hits) if hits else f"未找到匹配：{query}"
