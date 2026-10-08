"""文件工具：read_file（读）、list_files（列目录）、apply_patch（写）。"""
import os
from pathlib import Path

from langchain_core.tools import tool

# 路径校验与“垃圾文件”过滤规则统一从 _common 引入，避免各工具各写一份
from tools._common import WORKDIR, is_junk, resolve_in_workdir
from tools.schemas import ApplyPatchInput, ListFilesInput, ReadFileInput

# read_file 一次最多返回多少行，避免把整个大文件塞进上下文（C3）
MAX_READ_LINES = 400


@tool(args_schema=ReadFileInput)
def read_file(path: str, start_line: int | None = 1, end_line: int | None = None) -> str:
    """读取工作区内某个文本文件的内容，返回带行号的文本，可用行号范围截取。"""
    # —— 路径安全校验 ——
    try:
        target = resolve_in_workdir(path)
    except ValueError as e:
        return f"读取失败：{e}"

    if not target.exists():
        return f"文件不存在：{path}"
    if not target.is_file():
        return f"不是文件（可能是目录）：{path}"

    # —— 读取文件 ——
    try:
        # 用 splitlines() 而不是 readlines()：不保留行尾的 \n，便于统一拼接行号
        lines = target.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return f"{path} 不是 UTF-8 文本文件（可能是二进制），暂不支持读取"
    except Exception as e:
        return f"读取 {path} 出错：{e}"

    # —— 计算要返回的行范围 ——
    total = len(lines)
    start = max(1, start_line or 1)                    # 行号从 1 开始；不传默认从头读
    end = total if end_line is None else min(end_line, total)

    if start > total:
        return f"起始行 {start} 超过文件总行数（共 {total} 行）"
    if end < start:
        return f"结束行 {end} 小于起始行 {start}"

    # —— 大文件截断：一次别读太多行 ——
    truncated = False
    if end - start + 1 > MAX_READ_LINES:
        end = start + MAX_READ_LINES - 1
        truncated = True

    # 行号从 start 开始编号（注意不是从 1），并保留原始缩进——代码缩进有意义，不能 strip
    body = "\n".join(f"{start + i} {line}" for i, line in enumerate(lines[start - 1:end]))

    header = f"# {path}（第 {start}-{end} 行，共 {total} 行）"
    if truncated:
        # C2：说清楚是“从第 start 行起”的 N 行，而不是含糊的“前 N 行”
        header += f"\n# 注意：内容过多，本次只返回从第 {start} 行起的 {MAX_READ_LINES} 行，可用 start_line/end_line 继续读"
    return f"{header}\n{body}"


@tool(args_schema=ListFilesInput)
def list_files(path: str = ".") -> str:
    """列出工作区内某个目录下的文件和子目录（只列一层）。"""
    # —— 路径安全校验 ——
    try:
        target = resolve_in_workdir(path)
    except ValueError as e:
        return f"列出失败：{e}"

    if not target.exists():
        return f"目录不存在：{path}"
    if not target.is_dir():
        return f"不是目录：{path}"

    # —— 遍历目录 ——
    # 用 os.scandir：比 iterdir() 更快，DirEntry 自带的类型信息也不用反复 stat。
    # 只列一层：要看更深的内容，让模型再对子目录调一次 list_files 往下钻。
    # 在 with 块内就把需要的属性取出来，不依赖 DirEntry 离开上下文后的生命周期。
    try:
        with os.scandir(target) as it:
            raw = [(e.is_dir(), e.name, e.path) for e in it]
    except Exception as e:
        return f"读取目录 {path} 出错：{e}"

    # —— 过滤垃圾文件 ——
    # 规则统一在 _common.is_junk：垃圾目录、密钥文件、二进制后缀都会被跳过
    entries = [(d, n, p) for (d, n, p) in raw if not is_junk(n, d)]

    # 目录排在文件前面，同类按名字排序：多次调用结果稳定、好读
    entries.sort(key=lambda t: (not t[0], t[1].lower()))

    # —— 截断，防止条目太多 ——
    MAX_ENTRIES = 200
    truncated = len(entries) > MAX_ENTRIES
    entries = entries[:MAX_ENTRIES]

    # —— 返回相对路径 ——
    # 用相对工作区根的路径（'/' 分隔），模型拿到的就是它后续能直接喂给 read_file 的那种路径。
    lines = []
    for is_dir, name, p in entries:
        try:
            shown = Path(p).relative_to(WORKDIR).as_posix()
        except ValueError:
            shown = name
        lines.append(f"{'DIR ' if is_dir else 'FILE'} {shown}")

    if not lines:
        return f"{path}：（空目录）"
    header = f"# {path}（{len(lines)} 项）"
    if truncated:
        header += f"\n# 注意：条目过多，只显示前 {MAX_ENTRIES} 项"
    return header + "\n" + "\n".join(lines)


def _edit_pair(edit):
    """取出一处替换的 (old_text, new_text)。

    兼容 edit 以 pydantic 模型或 dict 传入——取决于 LangChain 校验 args_schema 后的形态，
    两种都接受，避免因框架细节而失败。
    """
    if isinstance(edit, dict):
        return edit["old_text"], edit["new_text"]
    return edit.old_text, edit.new_text


def _find_unique(content: str, old: str):
    """在 content 中唯一定位 old，返回 ((start, end), None) 或 (None, 错误原因)。

    两级匹配：
    1. 精确子串匹配（最快最准）。
    2. 退化匹配：忽略每行首尾空白后逐行比对，容忍模型输出里的缩进/空行轻微差异。
    匹配到多处时要求模型提供更长上下文——宁可不改，也不能改错地方。
    """
    # 1) 精确匹配
    idx = content.find(old)
    if idx != -1:
        if content.find(old, idx + 1) != -1:
            return None, "匹配到多处，请提供更长的上下文以唯一定位"
        return (idx, idx + len(old)), None

    # 2) 忽略首尾空白的逐行匹配
    targets = [l.strip() for l in old.splitlines()]
    if not targets:
        return None, "old_text 不能是空内容"

    content_lines = content.splitlines(keepends=True)
    # 预计算每一行在整个字符串里的起始偏移
    offsets = []
    pos = 0
    for l in content_lines:
        offsets.append(pos)
        pos += len(l)

    n = len(targets)
    hits = []
    for i in range(len(content_lines) - n + 1):
        if all(content_lines[i + j].strip() == targets[j] for j in range(n)):
            hits.append(i)

    if not hits:
        return None, "在文件中找不到这段原文"
    if len(hits) > 1:
        return None, "匹配到多处（宽松匹配），请提供更长的上下文"
    i = hits[0]
    start = offsets[i]
    end = offsets[i + n - 1] + len(content_lines[i + n - 1])
    return (start, end), None


@tool(args_schema=ApplyPatchInput)
def apply_patch(path: str, edits: list) -> str:
    """对文件做替换式编辑：按顺序把每处 old_text 换成 new_text。

    所有替换先在内存里全部成功，才写回磁盘——任意一处失败就整体不落盘（原子性），
    避免"改了一半"把文件搞坏。
    """
    # —— 路径安全校验 ——
    try:
        target = resolve_in_workdir(path)
    except ValueError as e:
        return f"编辑失败：{e}"

    if not target.exists():
        return f"文件不存在：{path}"
    if not target.is_file():
        return f"不是文件：{path}"

    try:
        content = target.read_text(encoding="utf-8")
    except Exception as e:
        return f"读取 {path} 出错：{e}"

    # —— 依次应用替换，全部在内存里完成 ——
    for i, edit in enumerate(edits, start=1):
        old_text, new_text = _edit_pair(edit)
        span, err = _find_unique(content, old_text)
        if err:
            return (
                f"第 {i} 处替换失败：{err}\n"
                f"文件未做任何修改（本次编辑整体回滚），请调整位置或上下文后重试。"
            )
        s, e = span
        content = content[:s] + new_text + content[e:]

    # —— 全部成功才写回 ——
    try:
        target.write_text(content, encoding="utf-8")
    except Exception as e:
        return f"写入 {path} 出错：{e}"

    return f"已修改 {path}：成功应用 {len(edits)} 处替换"
