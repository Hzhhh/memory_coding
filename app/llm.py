# -*- coding: utf-8 -*-
"""OpenRouter 统一 LLM 客户端:蒸馏(distill)/判别(judge)/离线仿真共用。

设计原则(与 rerank.py 一致):
- 零新增依赖:stdlib urllib;
- 永不抛出:任何失败返回 None,调用方各自降级;
- 有界重试 + 请求级超时。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

_URL = os.environ.get(
    "LLM_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/") + "/chat/completions"


def _load_key() -> str:
    """环境变量优先;否则读仓库根 .env(本地开发用,已 gitignore)。"""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


API_KEY = _load_key()


def available() -> bool:
    return bool(API_KEY)


def chat(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    timeout: float = 90.0,
    retries: int = 2,
) -> str | None:
    """单轮对话;返回 content 文本,失败返回 None。"""
    if not API_KEY:
        return None
    body = json.dumps(
        {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                _URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode())
            return out["choices"][0]["message"]["content"]
        except Exception as e:  # 网络/超时/限流/5xx 一律降级路径处理
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                print(f"[llm] {model} failed after {retries + 1} tries: {e}")
    return None


def chat_json(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    timeout: float = 90.0,
    retries: int = 2,
) -> dict | list | None:
    """要求 JSON 输出并解析;解析失败返回 None(等价于调用失败)。"""
    text = chat(model, system, user, max_tokens, temperature, timeout, retries)
    if text is None:
        return None
    # 剥掉 markdown 代码栅栏后取首个平衡的 JSON 对象/数组
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
