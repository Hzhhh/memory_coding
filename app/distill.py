# -*- coding: utf-8 -*-
"""Add 侧经验卡蒸馏:工程轨迹 -> 自足可执行的 DEBUG/PATTERN 经验卡。

设计要点(spec §3/§4):
- Pass 1:强模型把一次 Add 的 messages 蒸馏成 0~3 张卡(JSON schema 约束);
- Pass 2:代码级事实校验——文件路径必须能在源文本中回溯,防幻觉行号/路径;
          不合格卡重生成一次,再不合格直接丢弃(原块保留,卡是增量不是替代);
- profile 卡:域级 repo 概览,由该域全部卡合成,每次 Add 后幂等重建;
- 全程硬超时(HARD_CAP),超时即放弃蒸馏,调用方只保留原块。
"""
from __future__ import annotations

import os
import time

from . import llm

ENABLED = os.environ.get("DISTILL", "0") == "1"
MODEL = os.environ.get("DISTILL_MODEL", "openai/gpt-4o")
TIMEOUT = float(os.environ.get("DISTILL_TIMEOUT", "75"))
HARD_CAP = float(os.environ.get("DISTILL_HARD_CAP", "90"))  # 单次 Add 蒸馏总预算(秒)
MSG_CHARS = int(os.environ.get("DISTILL_MSG_CHARS", "2800"))     # 单消息截断
TOTAL_CHARS = int(os.environ.get("DISTILL_TOTAL_CHARS", "14000"))  # 整次输入截断

_SYS_PASS1 = (
    "You are a senior software engineer building long-term memory for a coding agent. "
    "Convert the raw engineering trajectory below (issue text, discussions, patches, "
    "error stacks, test output) into EXPERIENCE CARDS that a mid-level engineer can act "
    "on DIRECTLY, without reading the raw trajectory.\n"
    "Rules:\n"
    "- Extract 0-3 cards; one card per independent problem/solution in the trajectory. "
    "Skip boilerplate, repo setup, and CI noise.\n"
    "- Card types: DEBUG = fixing a defect (symptom/root_cause/fix); "
    "PATTERN = implementing a feature or refactor (approach/conventions).\n"
    "- Each card must be fully SELF-CONTAINED: no pronouns, no references to 'the issue' "
    "or 'above'; name the repo/module/files explicitly.\n"
    "- Every file path must appear VERBATIM in the trajectory. NEVER invent paths, line "
    "numbers, or commands. If unsure, omit the field.\n"
    "- 'fix' states what to change and why it works, not the full diff.\n"
    "- Plain English, 150-400 tokens per card.\n"
    'Respond with ONLY JSON: {"cards": [{"type": "DEBUG"|"PATTERN", "task": str, '
    '"symptom": str, "root_cause": str, "fix": str, "files": [str], "verify": str, '
    '"pattern": str, "api": str}]}. Omit non-applicable fields.'
)

_SYS_PROFILE = (
    "You maintain a one-card REPOSITORY PROFILE for a coding agent's memory. "
    "Given experience cards extracted from past work in one repository, produce a "
    "single compact profile card (<=250 tokens) a coding agent can use as orienting "
    "context: what the repo is, language/framework, layout, conventions, how to run "
    "tests. Never invent facts not implied by the cards.\n"
    'Respond with ONLY JSON: {"repo": str, "conventions": str, "test": str}.'
)


def _render(messages: list[dict]) -> str:
    parts: list[str] = []
    total = 0
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = (m.get("content") or "")[:MSG_CHARS]
        piece = f"[{role}]\n{content}"
        if total + len(piece) > TOTAL_CHARS:
            piece = piece[: max(0, TOTAL_CHARS - total)]
        parts.append(piece)
        total += len(piece)
        if total >= TOTAL_CHARS:
            break
    return "\n\n".join(parts)


def _norm_file(f: str) -> str:
    f = (f or "").strip().strip("`\"'")
    for pre in ("a/", "b/"):
        if f.startswith(pre):
            f = f[len(pre):]
    return f.strip()


def _file_traceable(f: str, source: str) -> bool:
    """路径或其 basename 在源文本中出现(大小写不敏感)即视为可回溯。"""
    f = _norm_file(f)
    if not f:
        return False
    low = source.lower()
    if f.lower() in low:
        return True
    base = f.rsplit("/", 1)[-1]
    return bool(base) and base.lower() in low


def _validate(card: dict, source: str) -> tuple[dict, list[str]]:
    """返回(修正后的卡, 问题列表)。可修复问题就地修复,不可修复记入问题。"""
    problems: list[str] = []
    ctype = str(card.get("type", "")).upper()
    if ctype not in ("DEBUG", "PATTERN"):
        ctype = "DEBUG" if card.get("root_cause") or card.get("symptom") else "PATTERN"
    card["type"] = ctype

    for key in ("task", "symptom", "root_cause", "fix", "pattern", "api", "verify"):
        v = card.get(key)
        card[key] = str(v).strip() if isinstance(v, str) else ""

    if not card["task"]:
        problems.append("missing task")
    if ctype == "DEBUG":
        if not card["root_cause"] and not card["fix"]:
            problems.append("debug card without root_cause/fix")
        if not card["fix"]:
            problems.append("debug card without fix")
    else:
        if not card["pattern"] and not card["fix"]:
            problems.append("pattern card without pattern/fix")

    files = card.get("files")
    files = [f for f in (files if isinstance(files, list) else []) if isinstance(f, str)]
    keep = []
    for f in files:
        nf = _norm_file(f)
        if nf and _file_traceable(nf, source):
            keep.append(nf)
    if files and not keep:
        problems.append("files_untraceable")
    card["files"] = sorted(set(keep))

    # verify 命令宽松校验:主要 token 至少一半能在源文本找到,否则置空(卡保留)
    if card["verify"]:
        toks = [t for t in card["verify"].split() if len(t) > 3]
        hit = sum(1 for t in toks if t.lower() in source.lower())
        if toks and hit * 2 < len(toks):
            card["verify"] = ""
    return card, problems


def format_card(card: dict, when: str) -> str:
    lines = [f"[TASK] {card['task']}"]
    if card["type"] == "DEBUG":
        if card.get("symptom"):
            lines.append(f"[SYMPTOM] {card['symptom']}")
        if card.get("root_cause"):
            lines.append(f"[ROOT_CAUSE] {card['root_cause']}")
    else:
        if card.get("pattern"):
            lines.append(f"[PATTERN] {card['pattern']}")
        if card.get("api"):
            lines.append(f"[API] {card['api']}")
    if card.get("fix"):
        lines.append(f"[FIX] {card['fix']}")
    if card.get("files"):
        lines.append("[FILES] " + "; ".join(card["files"]))
    if card.get("verify"):
        lines.append(f"[VERIFY] {card['verify']}")
    if when:
        lines.append(f"[WHEN] {when}")
    return "\n".join(lines)


def distill_messages(messages: list[dict]) -> list[dict] | None:
    """messages -> 校验通过的卡列表;失败/无卡返回 None 或空列表。"""
    if not ENABLED or not llm.available() or not messages:
        return None
    deadline = time.time() + HARD_CAP
    source = "\n".join((m.get("content") or "") for m in messages)
    rendered = _render(messages)
    if len(rendered) < 200:  # 过短轨迹(寒暄/空内容)不值得蒸馏
        return None

    out = llm.chat_json(
        MODEL, _SYS_PASS1, rendered, max_tokens=2200, temperature=0.0,
        timeout=TIMEOUT, retries=1,
    )
    cards = out.get("cards") if isinstance(out, dict) else None
    if not isinstance(cards, list) or not cards:
        return None

    ok: list[dict] = []
    bad: list[dict] = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        c, probs = _validate(c, source)
        (bad if probs else ok).append(c)

    if bad and time.time() < deadline - 20:  # 有预算才重生成,否则直接丢弃
        hints = [
            "- a card referenced file paths that do NOT exist in the trajectory; "
            "use only paths that appear verbatim" for c in bad
            if not any(_file_traceable(f, source) for f in c.get("files", []) if f)
        ]
        note = ("\nThe previous attempt had problems:\n" + "\n".join(hints[:3])
                if hints else "")
        out2 = llm.chat_json(
            MODEL, _SYS_PASS1, rendered + note, max_tokens=2200, temperature=0.0,
            timeout=max(10.0, min(TIMEOUT, deadline - time.time())), retries=0,
        )
        cards2 = out2.get("cards") if isinstance(out2, dict) else None
        if isinstance(cards2, list):
            for c in cards2:
                if not isinstance(c, dict):
                    continue
                c, probs = _validate(c, source)
                if not probs:
                    ok.append(c)

    # 每次最多 3 张(spec 3.4)
    return ok[:3] if ok else None


def build_profile(card_texts: list[str]) -> dict | None:
    """域级 profile 卡:全部卡文本 -> {repo, conventions, test}。"""
    if not ENABLED or not llm.available() or not card_texts:
        return None
    joined = "\n\n".join(t[:1200] for t in card_texts[:20])
    out = llm.chat_json(
        MODEL, _SYS_PROFILE, f"Experience cards from this repository:\n\n{joined}",
        max_tokens=600, temperature=0.0, timeout=TIMEOUT, retries=1,
    )
    if not isinstance(out, dict) or not out.get("repo"):
        return None
    return out


def format_profile(p: dict, when: str) -> str:
    lines = [f"[REPO] {p.get('repo', '')}"]
    if p.get("conventions"):
        lines.append(f"[CONVENTIONS] {p['conventions']}")
    if p.get("test"):
        lines.append(f"[TEST] {p['test']}")
    lines.append(f"[WHEN] {when}")
    return "\n".join(lines)
