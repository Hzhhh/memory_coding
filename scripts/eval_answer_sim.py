# -*- coding: utf-8 -*-
"""answer 仿真闭环评测(spec §6):Add->Search->answer 仿真->判分。

与官方评测同构:每 (任务,条件) 一个隔离 user 域,灌入经验(±噪声),检索,
固定模型基于记忆解题,硬指标 file-localization + 软指标 LLM 判分。

用法(配置由环境变量注入,两配置各跑一次做配对比较):
  # baseline(v0.4 全配置):
  RERANK=1 DISTILL=0 JUDGE=0 OPENROUTER_API_KEY=... python scripts/eval_answer_sim.py \
      --tasks 12 --db data/poc_base.db --out results/poc_base.jsonl
  # v1(蒸馏+判别打包):
  RERANK=0 DISTILL=1 JUDGE=1 OPENROUTER_API_KEY=... python scripts/eval_answer_sim.py \
      --tasks 12 --db data/poc_v1.db --out results/poc_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import distill, judge, llm  # noqa: E402
from app.store import CodingMemoryStore  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "SWEContextBench", "data")

ANSWER_SYS = (
    "You are a software engineer solving a task in a repository. You are given the "
    "task and memories retrieved from past engineering work in the same repository. "
    "Using the memories as evidence (they may contain the exact fix pattern, the files "
    "to change, or how similar work was done), decide:\n"
    "1) which files must be modified;\n"
    "2) the fix/implementation approach.\n"
    'Respond with ONLY JSON: {"files": ["path", ...], "approach": str, '
    '"patch": str}. patch = unified diff if you can, else a precise edit description.'
)

SOFT_SYS = (
    "Compare a proposed solution with the reference fix for the same task. The "
    "proposal was produced with the help of retrieved memories; judge whether it "
    "would accomplish the same fix.\n"
    'Respond with ONLY JSON: {"score": <0-10 integer>, "captures_fix": true|false}. '
    "10 = equivalent to the reference fix."
)

BUG_KW = ("fix", "bug", "crash", "error", "fail", "wrong", "incorrect",
          "exception", "regression", "broken", "raises")
FEAT_KW = ("add", "support", "implement", "new ", "allow", "enable",
           "introduce", "feature", "extend")


def qtype_of(ps: str) -> str:
    head = (ps.split("\n", 1)[0] or "").lower()
    if any(k in head for k in FEAT_KW) and not any(k in head for k in BUG_KW):
        return "feature"
    return "bug"  # 数据里 bug 占 2/3,默认归 bug


def gold_files_of(patch: str) -> set[str]:
    files = set(re.findall(r"\+\+\+ b/(\S+)", patch or ""))
    files |= {m for m in re.findall(r"diff --git a/(\S+) b/\S+", patch or "")}
    return {f for f in files if f}


def norm(f: str) -> str:
    f = f.strip().strip("`\"'")
    for pre in ("a/", "b/"):
        if f.startswith(pre):
            f = f[len(pre):]
    return f


def build_messages(ps: str, patch: str) -> list[dict]:
    return [
        {"role": "user", "content": (ps or "")[:4000]},
        {"role": "assistant", "content": "[fix patch]\n" + (patch or "")[:4000]},
    ]


def load_tasks(n: int, seed: int) -> list[dict]:
    exp = pd.read_parquet(os.path.join(DATA, "SWEContextBench_Lite_Experience.parquet"))
    rel = pd.read_parquet(os.path.join(DATA, "SWEContextBench_Related_Lite.parquet"))
    ship = pd.read_parquet(os.path.join(DATA, "SWEContextBench_Relationship.parquet"))
    exp_by_id = {r["instance_id"]: r for _, r in exp.iterrows()}
    links: dict[str, list[str]] = {}
    for _, r in ship.iterrows():
        links.setdefault(r["related_instance_id"], []).append(r["experience_instance_id"])

    cands = []
    for _, r in rel.iterrows():
        gold_ids = links.get(r["instance_id"], [])
        if not gold_ids or not all(g in exp_by_id for g in gold_ids):
            continue
        cands.append({
            "rid": r["instance_id"], "repo": r["repo"],
            "ps": r["problem_statement"], "patch": r["patch"],
            "gold_ids": gold_ids, "qtype": qtype_of(r["problem_statement"]),
        })
    rng = random.Random(seed)
    bugs = [c for c in cands if c["qtype"] == "bug"]
    feats = [c for c in cands if c["qtype"] == "feature"]
    rng.shuffle(bugs); rng.shuffle(feats)
    n_feat = max(1, round(n / 3))
    picked = bugs[: max(0, n - n_feat)] + feats[:n_feat]
    # 噪声库:同仓非金标经验(同仓近邻干扰≈L3) + 跨仓经验(≈L4)
    for c in picked:
        same = [e for i, e in exp_by_id.items()
                if e["repo"] == c["repo"] and i not in c["gold_ids"]]
        others = [e for i, e in exp_by_id.items() if e["repo"] != c["repo"]]
        rng.shuffle(same); rng.shuffle(others)
        c["noise"] = [build_messages(e["problem_statement"], e["patch"])
                      for e in same[:6] + others[:6]]
        c["gold"] = [build_messages(exp_by_id[g]["problem_statement"],
                                    exp_by_id[g]["patch"]) for g in c["gold_ids"]]
    return picked


def run_case(store: CodingMemoryStore, task: dict, cond: str, args) -> dict:
    uid = f"{task['rid']}#{cond}"
    msgs_list = task["gold"] + (task["noise"] if cond == "noisy" else [])
    for i, msgs in enumerate(msgs_list):
        store.add(uid, uid, msgs, request_id=f"r{i:03d}")
    items = store.search_recall(uid, task["ps"][:3000], top_k=100)
    if judge.ENABLED:
        items = judge.judge_and_pack(
            task["ps"][:3000], None, items,
            fetch_excerpts=lambda it: store.fetch_excerpts(uid, it),
        )
    mem_text = "\n\n".join(
        f"<memory id={it['id']}>\n{it['content']}" for it in items
    )[:60000]
    ans = llm.chat_json(args.answer_model, ANSWER_SYS,
                        f"Task:\n{task['ps'][:3000]}\n\nMemories:\n{mem_text}",
                        max_tokens=2500, timeout=120.0, retries=1)
    pred_files = [norm(f) for f in (ans or {}).get("files", []) if isinstance(f, str)]
    gold_files = {norm(f) for f in gold_files_of(task["patch"])}
    inter = set(pred_files) & gold_files
    union = set(pred_files) | gold_files
    soft = llm.chat_json(
        args.soft_model, SOFT_SYS,
        f"Task:\n{task['ps'][:2000]}\n\nReference fix patch:\n{task['patch'][:4000]}"
        f"\n\nProposed solution:\n{json.dumps(ans or {}, ensure_ascii=False)[:4000]}",
        max_tokens=400, timeout=60.0, retries=1,
    )
    return {
        "rid": task["rid"], "repo": task["repo"], "qtype": task["qtype"], "cond": cond,
        "localized": bool(inter), "jaccard": round(len(inter) / len(union), 4) if union else 0.0,
        "pred_files": pred_files, "gold_files": sorted(gold_files),
        "soft_score": (soft or {}).get("score"),
        "captures_fix": (soft or {}).get("captures_fix"),
        "n_mem": len(items), "n_card": sum(1 for it in items if it.get("kind") == "card"),
        "has_profile": any(it.get("kind") == "profile" for it in items),
        "mem_chars": len(mem_text), "answer_ok": ans is not None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--db", default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--answer-model", default=os.environ.get("ANSWER_MODEL", "openai/gpt-4o"))
    ap.add_argument("--soft-model", default=os.environ.get("SOFT_MODEL", "openai/gpt-4o-mini"))
    args = ap.parse_args()

    if not llm.available():
        sys.exit("OPENROUTER_API_KEY missing -- answer simulation needs it")

    tasks = load_tasks(args.tasks, args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    db = args.db or re.sub(r"\.jsonl?$", "", args.out) + ".db"
    os.makedirs(os.path.dirname(db) or ".", exist_ok=True)
    store = CodingMemoryStore(db)

    print(f"[config] DISTILL={distill.ENABLED} JUDGE={judge.ENABLED} "
          f"answer={args.answer_model} tasks={len(tasks)} db={db}")

    cases = [(t, c) for t in tasks for c in ("relevant", "noisy")]
    done = 0
    with open(args.out, "w", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_case, store, t, c, args): (t, c) for t, c in cases}
        for fut in as_completed(futs):
            t, c = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"rid": t["rid"], "cond": c, "error": str(e)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            loc = row.get("localized")
            print(f"[{done}/{len(cases)}] {t['rid'][:40]} {c} "
                  f"localized={loc} soft={row.get('soft_score')} n={row.get('n_mem')}")

    # 汇总
    rows = [json.loads(l) for l in open(args.out, encoding="utf-8")]
    print("\n===== summary =====")
    for key in [("bug", "relevant"), ("bug", "noisy"), ("feature", "relevant"),
                ("feature", "noisy"), ("ALL", "relevant"), ("ALL", "noisy")]:
        qt, cd = key
        sel = [r for r in rows if r.get("cond") == cd
               and (qt == "ALL" or r.get("qtype") == qt) and "error" not in r]
        if not sel:
            continue
        loc = sum(1 for r in sel if r["localized"]) / len(sel)
        soft = [r["soft_score"] for r in sel if isinstance(r.get("soft_score"), (int, float))]
        print(f"{qt:8s} {cd:8s} n={len(sel):3d} localized={loc:.3f} "
              f"soft={sum(soft) / len(soft) if soft else float('nan'):.1f} "
              f"cap_fix={sum(1 for r in sel if r.get('captures_fix')) / len(sel):.3f}")


if __name__ == "__main__":
    main()
