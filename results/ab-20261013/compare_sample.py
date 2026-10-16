# -*- coding: utf-8 -*-
"""同查询集指标对比:从三轮 dump jsonl 过滤 ds 抽样查询集,计算 HitRate/Recall/MRR。"""
import json

KS = (5, 10, 25, 50, 100)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def metrics(rows):
    n = len(rows)
    found = [r["hit_rank"] for r in rows if r["hit_rank"] is not None]
    out = {"n": n, "Hit@100": len(found) / n}
    for k in KS:
        out[f"R@{k}"] = sum(1 for h in found if h < k) / n
    out["MRR"] = sum(1.0 / (h + 1) for h in found) / n
    return out


def main():
    base = load("results/ab-20261013/baseline.jsonl")
    gpt = load("results/ab-20261013/gpt.jsonl")
    ds = load("results/ab-20261013/ds30.jsonl")
    ids = {r["related_id"] for r in ds}
    groups = {
        "baseline": [r for r in base if r["related_id"] in ids],
        "gpt-4o-mini": [r for r in gpt if r["related_id"] in ids],
        "ds-v4.1-flash": ds,
    }
    m = {g: metrics(rows) for g, rows in groups.items()}
    keys = ["n", "Hit@100"] + [f"R@{k}" for k in KS] + ["MRR"]
    print(f"{'metric':<12}" + "".join(f"{g:>15}" for g in groups))
    for k in keys:
        row = f"{k:<12}"
        for g in groups:
            v = m[g][k]
            row += f"{v:>15.4f}" if isinstance(v, float) else f"{v:>15}"
        print(row)

    # 排序变化 case:相对 baseline 位次变化的查询(= rerank 真正生效的证据)
    bmap = {r["related_id"]: r["hit_rank"] for r in groups["baseline"]}
    for name in ("gpt-4o-mini", "ds-v4.1-flash"):
        changed = [
            (r["related_id"], bmap[r["related_id"]], r["hit_rank"])
            for r in groups[name]
            if r["hit_rank"] != bmap[r["related_id"]]
        ]
        better = sum(1 for _, b, d in changed if d is not None and (b is None or d < b))
        print(f"\n{name}: {len(changed)}/{len(ds)} 查询位次变化 (改善 {better} / 恶化 {len(changed)-better})")
        for rid, b, d in changed[:10]:
            print(f"  {rid[:58]:<58} {b} -> {d}")


if __name__ == "__main__":
    main()
