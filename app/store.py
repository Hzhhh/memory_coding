# -*- coding: utf-8 -*-
"""记忆存储:SQLite 持久化 + 每用户 BM25 索引 + 邻接扩展检索。

契约要点:
- Add 同步落盘(WAL 提交后才算成功);
- user_id 严格隔离——所有表和索引都带 user_id 维度;
- Search 只返回记忆证据(content 原文),不做任何答案生成。
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import dense, distill, judge, rerank
from .chunker import chunk_message
from .tokenizer import tokenize

DB_PATH = os.environ.get("AML_DB_PATH", "/data/memory.db")
SEED_TOP_N = int(os.environ.get("SEED_TOP_N", "20"))     # 每路检索的种子数
NEIGHBOR_SPAN = int(os.environ.get("NEIGHBOR_SPAN", "1"))  # 邻接扩展半径
MAX_RETURN = int(os.environ.get("MAX_RETURN", "100"))
RRF_K = int(os.environ.get("RRF_K", "60"))               # RRF 融合常数
DENSE_FLOOR = float(os.environ.get("DENSE_FLOOR", "0.40"))  # 稠密路噪声门控地板
QUERY_TITLE_BOOST = int(os.environ.get("QUERY_TITLE_BOOST", "1"))  # 标题 token 重复次数
CARD_BOOST = float(os.environ.get("CARD_BOOST", "1.3"))  # v1:卡/概览卡 RRF 加权


class BM25:
    """Lucene 风格 BM25:idf = ln(1 + (N-df+.5)/(df+.5)) 恒正,

    不像 rank_bm25.BM25Okapi 在小语料/高df词上会出现负 idf 把整表分数打负。"""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tfs: list[Counter] = [Counter(c) for c in corpus]
        self.doc_len = [len(c) for c in corpus]
        self.avgdl = (sum(self.doc_len) / len(corpus)) if corpus else 0.0
        n = len(corpus)
        df: Counter = Counter()
        for tf in self.doc_tfs:
            df.update(tf.keys())
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        out = [0.0] * len(self.doc_tfs)
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.doc_tfs):
                f = tf.get(term, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / (f + norm)
        return out


@dataclass
class _UserIndex:
    """单用户的内存态索引:BM25 + 向量矩阵(dirty 时重建)。"""
    chunks: list[dict] = field(default_factory=list)
    bm25: BM25 | None = None
    matrix: np.ndarray | None = None  # 与 chunks 对齐的 L2 归一化向量
    dirty: bool = True

    def rebuild(self) -> None:
        corpus = [c["tokens"] for c in self.chunks]
        self.bm25 = BM25(corpus) if corpus else None
        vecs = [c.get("vec") for c in self.chunks]
        if vecs and all(v is not None for v in vecs):
            self.matrix = np.vstack(vecs)
        else:
            self.matrix = None  # 有缺向量 → 该用户退回纯 BM25
        self.dirty = False


class CodingMemoryStore:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self._idx: dict[str, _UserIndex] = {}
        self._init_db()

    # ---------- 持久化 ----------
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS chunks(
                       id TEXT PRIMARY KEY,
                       user_id TEXT NOT NULL,
                       session_id TEXT NOT NULL,
                       msg_idx INTEGER NOT NULL,
                       chunk_idx INTEGER NOT NULL,
                       role TEXT,
                       content TEXT NOT NULL,
                       created_at TEXT NOT NULL,
                       tokens_json TEXT NOT NULL,
                       vec BLOB)"""
            )
            # 旧库迁移:v0.1 无 vec 列
            cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
            if "vec" not in cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN vec BLOB")
            # v1 迁移:kind(chunk/card/profile) + meta_json
            if "kind" not in cols:
                conn.execute(
                    "ALTER TABLE chunks ADD COLUMN kind TEXT NOT NULL DEFAULT 'chunk'")
            if "meta_json" not in cols:
                conn.execute(
                    "ALTER TABLE chunks ADD COLUMN meta_json TEXT NOT NULL DEFAULT '{}'")
            # Add 幂等登记表:平台重试会带同一 request_id,重复请求直接视为成功。
            # 关键:request_id 只在 user_id 内唯一(平台对每个样本用独立 user_id、
            # 样本内 request_id 从头编号),主键必须复合,否则跨样本吞写。
            conn.execute(
                """CREATE TABLE IF NOT EXISTS adds(
                       user_id TEXT NOT NULL,
                       request_id TEXT NOT NULL,
                       created_at TEXT NOT NULL,
                       PRIMARY KEY(user_id, request_id))"""
            )
            # 迁移:旧版主键只有 request_id -> 重建为复合主键
            pk = [r[1] for r in conn.execute("PRAGMA table_info(adds)") if r[5]]
            if pk == ["request_id"]:
                conn.execute(
                    "CREATE TABLE adds_new(user_id TEXT NOT NULL, request_id TEXT NOT NULL,"
                    " created_at TEXT NOT NULL, PRIMARY KEY(user_id, request_id))"
                )
                conn.execute("INSERT OR IGNORE INTO adds_new SELECT user_id, request_id, created_at FROM adds")
                conn.execute("DROP TABLE adds")
                conn.execute("ALTER TABLE adds_new RENAME TO adds")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON chunks(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_sess ON chunks(user_id, session_id)")

    def stats(self) -> dict:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT user_id), COUNT(DISTINCT session_id) FROM chunks"
            ).fetchone()
            kinds = dict(conn.execute(
                "SELECT kind, COUNT(*) FROM chunks GROUP BY kind").fetchall())
        return {"chunks": row[0], "users": row[1], "sessions": row[2],
                "by_kind": kinds}

    # ---------- Add ----------
    def add(self, user_id: str, session_id: str, messages: list[dict], request_id: str | None = None) -> int:
        """同步写入并提交;返回写入块数。失败抛异常 -> 上层 5xx,不会假成功。

        幂等:平台对 Add 的重试保持同一 request_id(官方错误处理规范),
        已登记过的 request_id 直接返回成功,不重复写入。
        """
        if request_id is None:
            request_id = uuid.uuid4().hex
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            with self._conn() as conn:
                try:
                    conn.execute(
                        "INSERT INTO adds(request_id, user_id, created_at) VALUES(?,?,?)",
                        (request_id, user_id, now),
                    )
                except sqlite3.IntegrityError:
                    return 0  # 重试请求:逻辑上已成功,幂等返回
        n = self._write_chunks(user_id, session_id, messages, request_id)
        # v1:Add 内同步蒸馏(契约:处理完成才回 200;LLM 调用全程在锁外,
        # 不阻塞并发 Add;蒸馏失败=只保留原块,行为退回 v0.4)
        if distill.ENABLED:
            cards = distill.distill_messages(messages)
            if cards:
                self._write_cards(user_id, session_id, request_id, cards)
            self._rebuild_profile(user_id, session_id)
        return n

    def _write_cards(self, user_id: str, session_id: str, request_id: str,
                     cards: list[dict]) -> int:
        """蒸馏卡落盘:与原块同表(kind='card'),卡文本进双索引。"""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        texts = [distill.format_card(c, c.get("when", now)) for c in cards]
        vecs = dense.encode(texts)
        rows: list[tuple] = []
        mem: list[dict] = []
        for k, text in enumerate(texts):
            cid = f"{user_id}:{request_id}:c{k}"
            toks = tokenize(text)
            vec = vecs[k] if vecs is not None else None
            meta = {"type": cards[k].get("type", "DEBUG"), "src_request": request_id,
                    "files": cards[k].get("files", []), "when": cards[k].get("when", now)}
            rows.append(
                (cid, user_id, session_id, -1, k, "card", text, now, json.dumps(toks),
                 vec.tobytes() if vec is not None else None, "card", json.dumps(meta))
            )
            mem.append({"id": cid, "session_id": session_id, "msg_idx": -1,
                        "chunk_idx": k, "role": "card", "content": text,
                        "created_at": now, "tokens": toks, "vec": vec,
                        "kind": "card", "meta": meta})
        with self._lock:
            with self._conn() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO chunks(id,user_id,session_id,msg_idx,chunk_idx,"
                    "role,content,created_at,tokens_json,vec,kind,meta_json)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    rows,
                )
            existed = user_id in self._idx
            self._get_index(user_id)
            if existed:
                self._idx[user_id].chunks.extend(mem)
            self._idx[user_id].dirty = True
        return len(rows)

    def _rebuild_profile(self, user_id: str, session_id: str) -> None:
        """域级 repo 概览卡:由该域全部经验卡合成,幂等重建;失败保留旧卡。"""
        with self._lock:
            idx = self._get_index(user_id)
            card_texts = [c["content"] for c in idx.chunks if c.get("kind") == "card"]
        if not card_texts:
            return
        p = distill.build_profile(card_texts)  # LLM 调用,锁外
        if not p:
            return
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        text = distill.format_profile(p, now)
        toks = tokenize(text)
        vec = dense.encode_one(text)
        meta = {"type": "PROFILE"}
        row = (f"{user_id}:profile", user_id, session_id, -1, 0, "profile", text, now,
               json.dumps(toks), vec.tobytes() if vec is not None else None,
               "profile", json.dumps(meta))
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "DELETE FROM chunks WHERE user_id=? AND kind='profile'", (user_id,))
                conn.execute(
                    "INSERT INTO chunks(id,user_id,session_id,msg_idx,chunk_idx,role,"
                    "content,created_at,tokens_json,vec,kind,meta_json)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    row,
                )
            existed = user_id in self._idx
            if existed:
                idx = self._idx[user_id]
                idx.chunks = [c for c in idx.chunks if c.get("kind") != "profile"]
                idx.chunks.append({"id": row[0], "session_id": session_id,
                                   "msg_idx": -1, "chunk_idx": 0, "role": "profile",
                                   "content": text, "created_at": now, "tokens": toks,
                                   "vec": vec, "kind": "profile", "meta": meta})
                idx.dirty = True

    def fetch_excerpts(self, user_id: str, item: dict, k: int = 2,
                       chars: int = 900) -> list[dict]:
        """行动卡的原文摘录配对:按卡 meta.src_request 定位源块,取头 k 块。

        返回的是已存储记录本身(content 允许截断,属"筛选/截断"而非生成)。
        """
        meta = item.get("meta") or {}
        src = meta.get("src_request")
        if not src:
            return []
        prefix = f"{user_id}:{src}:"
        with self._lock:
            idx = self._get_index(user_id)
            srcs = [c for c in idx.chunks
                    if c.get("kind") == "chunk" and c["id"].startswith(prefix)][:k]
        out = []
        for c in srcs:
            content = c["content"][:chars]
            out.append({"id": c["id"], "content": content,
                        "score": round(float(item.get("score") or 0) * 0.5, 4),
                        "created_at": c["created_at"], "kind": "chunk", "meta": {}})
        return out

    def _write_chunks(self, user_id: str, session_id: str, messages: list[dict], request_id: str) -> int:
        rows: list[tuple] = []
        mem_chunks: list[dict] = []
        pending: list[tuple] = []
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for mi, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            ts = msg.get("timestamp")
            created = (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)) if ts else now
            )
            for ci, piece in enumerate(chunk_message(content)):
                toks = tokenize(piece)
                pending.append((mi, ci, role, piece, created, toks))

        # 本请求所有块一次性批量编码(降级时 vecs=None → 纯 BM25)
        pieces = [p[3] for p in pending]
        vecs = dense.encode(pieces) if pieces else None

        for k, (mi, ci, role, piece, created, toks) in enumerate(pending):
            # 确定性 id:user + request_id 域内唯一(跨用户同 request_id 不冲突),
            # 与幂等登记表的双保险保持同一作用域
            cid = f"{user_id}:{request_id}:{mi}:{ci}"
            vec = vecs[k] if vecs is not None else None
            rows.append(
                (cid, user_id, session_id, mi, ci, role, piece, created,
                 json.dumps(toks),
                 vec.tobytes() if vec is not None else None, "chunk", "{}")
            )
            mem_chunks.append(
                {"id": cid, "session_id": session_id, "msg_idx": mi,
                 "chunk_idx": ci, "role": role, "content": piece,
                 "created_at": created, "tokens": toks, "vec": vec,
                 "kind": "chunk", "meta": {}}
            )
        with self._lock:
            with self._conn() as conn:
                conn.executemany(
                    "INSERT INTO chunks(id,user_id,session_id,msg_idx,chunk_idx,role,content,created_at,tokens_json,vec,kind,meta_json)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    rows,
                )
            # 增量维护内存索引:索引未装载时 _get_index 会连本次新行一起从磁盘读出,
            # 已装载时才需要手动追加,否则会双重计入
            existed = user_id in self._idx
            self._get_index(user_id)
            if existed:
                self._idx[user_id].chunks.extend(mem_chunks)
            self._idx[user_id].dirty = True
        return len(rows)

    # ---------- Search ----------
    def search_recall(self, user_id: str, query: str, top_k: int = 100) -> list[dict]:
        """混合召回(v0.4 管线)+ v1 卡加权;返回含 kind/meta 的富条目。

        judge 开启时跳过 rerank(判别器取代其职能,省一次 LLM 延迟)。
        """
        with self._lock:
            idx = self._get_index(user_id)
            if idx.bm25 is None or not idx.chunks:
                return []

            # 查询构造:标题行(第一行)承载最强信号,BM25 查询里加重一遍
            qtoks = tokenize(query)
            if QUERY_TITLE_BOOST > 0 and "\n" in query:
                title = query.split("\n", 1)[0]
                qtoks = qtoks + tokenize(title) * QUERY_TITLE_BOOST

            # 路径 1:BM25 词面命中
            bm_scores = idx.bm25.get_scores(qtoks)
            bm_order = sorted(range(len(idx.chunks)), key=lambda i: bm_scores[i], reverse=True)
            bm_seeds = [i for i in bm_order[:SEED_TOP_N] if bm_scores[i] > 0]

            # 路径 2:稠密语义近邻 + 噪声门控:
            # max sim 低于地板 → 语义上无真匹配,稠密路不出种子(防噪声灌入 Answer)
            dn_seeds: list[int] = []
            if idx.matrix is not None:
                qv = dense.encode_one(query)
                if qv is not None:
                    sims = idx.matrix @ qv
                    if float(sims.max(initial=0.0)) >= DENSE_FLOOR:
                        dn_seeds = [int(i) for i in np.argsort(-sims)[:SEED_TOP_N]]

            # 双路全空 = 纯噪声查询 → 返回空,不浪费 Answer 上下文
            if not bm_seeds and not dn_seeds:
                return []

            # RRF 融合:rank-based,对两路分数量纲不敏感
            fused: dict[int, float] = {}
            for rank, i in enumerate(bm_seeds):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            for rank, i in enumerate(dn_seeds):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)

            # v1:经验卡/概览卡乘 boost——同排名下卡优先于原块
            for i in list(fused):
                if idx.chunks[i].get("kind") in ("card", "profile"):
                    fused[i] *= CARD_BOOST

            # 邻接扩展:种子块带上前后邻居,patch 上下文不被切走;
            # 邻居按 0.99 降权,保证种子永远排在它拉进来的邻居之前。
            # v1:卡/profile 独立成条,不参与邻接(其 msg_idx=-1 无邻接语义)
            picked: dict[int, float] = {}
            for rank, (i, s) in enumerate(sorted(fused.items(), key=lambda kv: kv[1], reverse=True)):
                base_is_chunk = idx.chunks[i].get("kind") == "chunk"
                for j in range(i - NEIGHBOR_SPAN, i + NEIGHBOR_SPAN + 1):
                    if not 0 <= j < len(idx.chunks):
                        continue
                    if j != i and (not base_is_chunk or idx.chunks[j].get("kind") != "chunk"):
                        continue
                    cand = s if j == i else s * 0.99 - 1e-6 * (rank + 1)
                    picked[j] = max(picked.get(j, 0.0), cand)

            ranked = sorted(picked.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            items = []
            for j, s in ranked:
                c = idx.chunks[j]
                items.append(
                    {
                        "id": c["id"],
                        "content": c["content"],
                        "score": round(float(s), 4),
                        "created_at": c["created_at"],
                        "kind": c.get("kind", "chunk"),
                        "meta": c.get("meta", {}),
                    }
                )

        # gpt-4o-mini 精排(锁外 LLM):失败/关闭时保持 RRF 顺序,永不因重排报错
        if not judge.ENABLED and rerank.ENABLED and len(items) > 1:
            head = items[: rerank.TOP_N]
            order = rerank.rerank(
                query,
                [{"idx": n, "content": it["content"]} for n, it in enumerate(head)],
            )
            if order is not None:
                rest = [n for n in range(len(head)) if n not in set(order)]
                items = [head[n] for n in order + rest] + items[len(head):]
        return items

    def search(self, user_id: str, query: str, top_k: int = 100) -> list[dict]:
        """契约兼容视图:老调用方(eval 脚本)只拿 id/content/score/created_at。"""
        return [
            {k: it[k] for k in ("id", "content", "score", "created_at")}
            for it in self.search_recall(user_id, query, top_k)
        ]

    def _get_index(self, user_id: str) -> _UserIndex:
        idx = self._idx.get(user_id)
        if idx is None:
            idx = _UserIndex()
            with self._conn() as conn:
                for row in conn.execute(
                    "SELECT id,session_id,msg_idx,chunk_idx,role,content,created_at,tokens_json,kind,meta_json,vec"
                    " FROM chunks WHERE user_id=? ORDER BY session_id,msg_idx,chunk_idx",
                    (user_id,),
                ):
                    vec = (
                        np.frombuffer(row[10], dtype=np.float32)
                        if row[10] is not None
                        else None
                    )
                    try:
                        meta = json.loads(row[9]) if row[9] else {}
                    except Exception:
                        meta = {}
                    idx.chunks.append(
                        {
                            "id": row[0], "session_id": row[1], "msg_idx": row[2],
                            "chunk_idx": row[3], "role": row[4], "content": row[5],
                            "created_at": row[6], "tokens": json.loads(row[7]),
                            "kind": row[8] or "chunk", "meta": meta,
                            "vec": vec,
                        }
                    )
            # 旧数据补向量:仅当存在缺失且模型可用时批量编码(自愈迁移)
            missing = [i for i, c in enumerate(idx.chunks) if c.get("vec") is None]
            if missing and dense.available():
                vecs = dense.encode([idx.chunks[i]["content"] for i in missing])
                if vecs is not None:
                    for k, i in enumerate(missing):
                        idx.chunks[i]["vec"] = vecs[k]
                    with self._conn() as conn:
                        conn.executemany(
                            "UPDATE chunks SET vec=? WHERE id=?",
                            [
                                (idx.chunks[i]["vec"].tobytes(), idx.chunks[i]["id"])
                                for i in missing
                            ],
                        )
            self._idx[user_id] = idx
        if idx.dirty:
            idx.rebuild()
        return idx
