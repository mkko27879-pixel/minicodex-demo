"""run 工具：在工作区执行受限命令，并只把有用的输出返回给大模型。"""
import os
import re
import shlex
import subprocess
import sys

from langchain_core.tools import tool

from tools._common import WORKDIR
from tools.schemas import RunCommandInput

# 白名单：当前只放行 pytest 测试命令
ALLOWED_COMMAND = "pytest"

# 超时熔断：命令最多跑多久，超了直接掐掉
TIMEOUT_SECONDS = 60

# 输出上限：失败时最多回传多少字符，避免一次报错把上下文撑爆
MAX_OUTPUT_CHARS = 4000

# 这些环境变量可能藏着密钥，执行外部命令前从子进程环境里剔除
SENSITIVE_ENV_KEYS = ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "DB_URI", "OPENAI_API_KEY")

# pytest 的汇总行形如 "3 passed in 0.12s"，用它更精确地抓结果，避免误抓含 "error" 的 warning 行（C4）
SUMMARY_RE = re.compile(r"\b\d+\s+(passed|failed|error|errors|skipped|deselected)\b")


def _clean_env() -> dict:
    """复制一份环境变量，剔除可能含密钥的项。

    为什么要做：被测代码理论上可能打印环境变量；剔除后即便它这么做，
    也不会把 API Key / 数据库连接串带进给大模型的输出里。
    """
    env = dict(os.environ)
    for key in SENSITIVE_ENV_KEYS:
        env.pop(key, None)
    return env


def _filter_output(output: str, returncode: int) -> str:
    """过滤命令输出，只留对大模型有用的部分。

    - 成功：pytest 会打一大串通过信息，对大模型没价值，只回一行汇总；
    - 失败：pytest 的关键报错在末尾（FAILURES / short test summary），所以保留尾部。
    """
    if returncode == 0:
        # 用 SUMMARY_RE 精确匹配汇总行，例如 "3 passed in 0.12s"
        lines = [l.strip() for l in output.splitlines() if l.strip()]
        summary = next((l for l in reversed(lines) if SUMMARY_RE.search(l)), "")
        return f"命令执行成功。{summary}" if summary else "命令执行成功（无输出）"

    # 失败：保留尾部（报错在这里），过长则截断前面的内容
    if len(output) > MAX_OUTPUT_CHARS:
        output = "...（输出过长，已截断前面的内容）\n" + output[-MAX_OUTPUT_CHARS:]
    return f"命令失败（退出码 {returncode}）：\n{output}"


# 运行命令工具，验证代码是否安全
@tool(args_schema=RunCommandInput)
def run(command: str) -> str:
    """在工作区执行命令并返回输出。当前仅允许 pytest 测试命令。"""
    # A1：先把反斜杠统一成正斜杠。shlex 在 POSIX 模式下会把 '\' 当转义符，
    #     导致 'tests\test_x.py' 被解析成 'teststest_x.py'；Windows 下正斜杠同样可用。
    cmd = command.strip().replace("\\", "/")

    # —— 白名单校验 ——
    # 用 == 而不是 startswith，避免 'pytest; ...' 这种前缀相同的绕过写法
    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        return f"命令解析失败：{e}"
    if not parts or parts[0] != ALLOWED_COMMAND:
        return f"拒绝执行：当前只允许 {ALLOWED_COMMAND} 命令，收到的是：{cmd}"

    # 用当前解释器以 `-m pytest` 运行，原因有二：
    #   1) 即使 pytest 不在 PATH 上也能找到它；
    #   2) shell=False 直接执行参数数组，从根本上杜绝命令注入（无需再过滤 ; && | 等符号）。
    argv = [sys.executable, "-m", "pytest", *parts[1:]]

    # —— 剔除环境变量 ——
    env = _clean_env()

    try:
        # —— 超时熔断 + 路径限制 ——
        proc = subprocess.run(
            argv,
            cwd=WORKDIR,          # 路径限制：固定在工作区根目录执行
            env=env,              # 不带敏感环境变量
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",     # 输出含非法字节也不崩
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"命令超时（超过 {TIMEOUT_SECONDS} 秒）：{cmd}"
    except Exception as e:
        return f"执行失败：{e}"

    # —— 输出过滤：成功不发日志给大模型，失败只发关键报错 ——
    return _filter_output((proc.stdout or "") + (proc.stderr or ""), proc.returncode)
