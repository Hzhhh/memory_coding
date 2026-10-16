# -*- coding: utf-8 -*-
"""Search 侧判别 + 题型条件化打包(spec §5)。

一次 LLM 调用完成:查询题型识别 + 候选三分类(ACTIONABLE/CONTEXT/IRRELEVANT);
随后纯代码打包:repo 概览卡 → 行动卡(带原文摘录配对)→ 语境材料补足 → 预算裁剪。

合规红线:LLM 输出只影响"选哪些/什么序/多少条",返回的每条 content 均为已存
储记录,本模块零生成、零改写。任何失败 → 原样返回输入(降级 = v0.4 行为)。
"""
from __future__ import annotations

import os

from . import llm

ENABLED = os.environ.get("JUDGE", "0") == "1"
MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-4o")
TIMEOUT = float(os.environ.get("JUDGE_TIMEOUT", "45"))
TOP_N = int(os.environ.get("JUDGE_TOP_N", "60"))          # 参与判别的候选数
SNIPPET = int(os.environ.get("JUDGE_SNIPPET", "500"))     # 候选展示截断
TARGET_BUG = int(os.environ.get("JUDGE_TARGET_BUG", "40"))
TARGET_FEATURE = int(os.environ.get("JUDGE_TARGET_FEATURE", "75"))
TARGET_OTHER = int(os.environ.get("JUDGE_TARGET_OTHER", "60"))
TOKEN_BUDGET = int(os.environ.get("PACK_TOKEN_BUDGET", "55000"))
MAX_EXCERPTS = int(os.environ.get("PACK_MAX_EXCERPTS", "10"))
EXCERPT_CHARS = int(os.environ.get("PACK_EXCERPT_CHARS", "900"))

_SYS = (
    "You judge memories for a coding agent. Given a NEW engineering task and numbered "
    "memory candidates retrieved from past work, decide for each candidate whether it "
    "would actually help solve the new task.\n"
    "Classification:\n"
    "- A (ACTIONABLE): directly useful — same bug pattern, same root cause, same "
    "API/module to change, same fix strategy, or the exact feature pattern to follow.\n"
    "- C (CONTEXT): orienting material — repo conventions, architecture, test setup, "
    "related module knowledge.\n"
    "- I (IRRELEVANT): everything else. IMPORTANT: a memory from the same repository "
    "that solves a DIFFERENT, unrelated problem is I, not A — same-repo is not "
    "same-relevant. Judge by the task to accomplish, not keyword overlap.\n"
    'Respond with ONLY JSON: {"qtype": "bug"|"feature"|"other", '
    '"judged": [{"n": <candidate number>, "cls": "A"|"C"|"I"}]}. '
    "qtype: does the task ask to fix/debug existing behavior (bug) or add/implement "
    "new capability (feature)? Classify every candidate you can."
)


def _tag(item: dict) -> str:
    kind = item.get("kind", "chunk")
    if kind == "profile":
        return "REPO-PROFILE"
    if kind == "card":
        return f"CARD-{item.get('meta', {}).get('type', 'DEBUG').upper()}"
    return "RAW"


def _est_tokens(s: str) -> int:
    return len(s) // 4 + 8


def judge_and_pack(
    query: str,
    options: list[str] | None,
    items: list[dict],
    fetch_excerpts=None,
) -> list[dict]:
    """items: 检索层输出(含 kind/meta);返回打包后的最终列表。永不抛出。"""
    if not ENABLED or not llm.available() or not items:
        return items
    try:
        return _pack(query, options or [], items, _classify(query, options, items),
                     fetch_excerpts)
    except Exception as e:
        print(f"[judge] fallback to recall order: {e}")
        return items


def _classify(query: str, options: list[str], items: list[dict]) -> dict:
    head = items[:TOP_N]
    lines = []
    for n, it in enumerate(head):
        body = it["content"].replace("\n", " ¶ ")[:SNIPPET]
        lines.append(f"[{n}] ({_tag(it)}) {body}")
    opt = ""
    if options:
        opt = "\nAnswer options (if a candidate's evidence helps choose among them, "
        "that candidate is A):\n" + "\n".join(f"- {o[:200]}" for o in options[:8])
    user = (
        f"New task:\n{query[:3000]}{opt}\n\nMemory candidates:\n" + "\n---\n".join(lines)
    )
    out = llm.chat_json(MODEL, _SYS, user, max_tokens=1800, temperature=0.0,
                        timeout=TIMEOUT, retries=1)
    if not isinstance(out, dict):
        return {}
    out.setdefault("qtype", "other")
    out.setdefault("judged", [])
    return out


def _pack(query: str, options: list[str], items: list[dict],
          verdict: dict, fetch_excerpts) -> list[dict]:
    qtype = verdict.get("qtype", "other")
    if qtype not in ("bug", "feature", "other"):
        qtype = "other"
    head = items[:TOP_N]
    tail = items[TOP_N:]  # 未判别部分保持 RRF 序补位

    cls: dict[int, str] = {}
    order_a: list[int] = []
    for j in verdict.get("judged", []):
        try:
            n, c = int(j["n"]), str(j["cls"]).upper()[:1]
        except Exception:
            continue
        if 0 <= n < len(head) and c in ("A", "C", "I"):
            if c == "A" and n not in cls:
                order_a.append(n)
            cls[n] = c

    target = {"bug": TARGET_BUG, "feature": TARGET_FEATURE,
              "other": TARGET_OTHER}[qtype]

    picked: list[dict] = []
    picked_ids: set[str] = set()

    def take(it: dict) -> bool:
        if it["id"] in picked_ids:
            return False
        picked.append(it)
        picked_ids.add(it["id"])
        return True

    # 1) repo 概览卡置顶(存在即取,不判别——它对任何任务都是定向材料)
    for it in head:
        if it.get("kind") == "profile":
            take(it)
            break

    # 2) 行动卡按判别顺序;bug 型给行动卡配原文摘录(卡+证据配对,answer 可回退原文)
    n_excerpt = 0
    for n in order_a:
        it = head[n]
        if not take(it):
            continue
        if qtype == "bug" and fetch_excerpts and n_excerpt < MAX_EXCERPTS:
            for ex in fetch_excerpts(it):
                if ex["id"] not in picked_ids:
                    take(ex)
                    n_excerpt += 1

    # 3) 语境材料(C 类)补足
    for n, it in enumerate(head):
        if len(picked) >= target:
            break
        if cls.get(n, "C") == "C":
            take(it)

    # 4) 未判别/尾部按 RRF 原序补位(feature 型刻意保持宽上下文)
    for n, it in enumerate(head):
        if len(picked) >= target:
            break
        if cls.get(n, "C") == "I":  # 判为 I 的丢弃,其余(含未判别)补位
            continue
        take(it)
    for it in tail:
        if len(picked) >= target:
            break
        take(it)

    # 5) token 预算裁剪(按序保留前缀——与平台截断规则同构)
    out: list[dict] = []
    used = 0
    for it in picked:
        t = _est_tokens(it["content"])
        if used + t > TOKEN_BUDGET:
            break
        out.append(it)
        used += t
    return out if out else items[: min(target, len(items))]
