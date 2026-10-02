#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import URL
from transformers import AutoModelForCausalLM, AutoTokenizer

from xindex_batch.mag_hash_index import MAGHashIndex
from xindex_batch.historical_temporal import (
    DEFAULT_DISTANCE_RESOURCE_DIR,
    HistoricalQCache,
    TemporalDistanceLoader,
    compute_historical_temporal_xindex,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIST_CSV = str(REPO_ROOT / "resources" / "100w_MAG_from_centroid.csv")
DEFAULT_MAG_PARQUET = str(REPO_ROOT / "resources" / "MAG-DB.parquet")
DEFAULT_HISTORICAL_PROMPT_DIR = str(REPO_ROOT / "resources" / "prompt_2011-2021")
HISTORICAL_VARIANTS = {"historical_distance", "historical_distance_compact"}
HISTORICAL_XINDEX_PROFILES = {"historical_acemap_temporal", "static_2021"}

SYSTEM_PROMPT = """You are an expert academic assistant specialized in identifying research topics.

Given the title and abstract of a paper:
1. Identify its main and intersecting academic disciplines.
2. Select research keywords that best represent the paper’s overall focus and interdisciplinary context.
3. Rank the keywords by overall relevance (most central first).

Strict output rules:
- Output only the keywords, one per line.
- Do not include any headings, labels, discipline names, numbering, punctuation, or explanations.
- Never output section titles such as “Research keywords”, “Disciplines”, “Intersecting fields”, etc.
- Each line must contain exactly one keyword or key phrase.
- Stop when no more clearly relevant keywords remain.
"""

MAG_ROOT_LABELS = [
    "Physics",
    "Computer Science",
    "Engineering",
    "Chemistry",
    "Biology",
    "Mathematics",
    "Sociology",
    "Geology",
    "Economics",
    "Psychology",
    "Materials Science",
    "Philosophy",
    "Business",
    "History",
    "Geography",
    "Environmental Science",
    "Art",
    "Political Science",
    "Medicine",
]

MAG_ROOT_PROMPT_LABELS = [
    "Physics 物理学",
    "Computer Science 计算机科学",
    "Engineering 工程学",
    "Chemistry 化学",
    "Biology 生物学",
    "Mathematics 数学",
    "Sociology 社会学",
    "Geology 地质学",
    "Economics 经济学",
    "Psychology 心理学",
    "Materials Science 材料科学",
    "Philosophy 哲学",
    "Business 商学 / 工商管理",
    "History 历史学",
    "Geography 地理学",
    "Environmental Science 环境科学",
    "Art 艺术学",
    "Political Science 政治学",
    "Medicine 医学",
]

GATED_SYSTEM_PROMPT = """You are an expert academic assistant judging whether a paper is interdisciplinary.

Decide whether the paper is clearly non-interdisciplinary and return the appropriate decision.
Regardless of the decision, provide the 10 most relevant research keywords.

For each keyword, assign exactly one English MAG level-0 root from this allowed list.
The Chinese text is only a translation aid; the JSON "root" value must use the English name exactly:
{roots}

Strict output rules:
- Output valid JSON only.
- Do not include markdown, explanations outside JSON, or extra keys.
- Always provide relevant keywords, including when decision is "non_cross".
- The JSON schema is:
  {{
    "decision": "non_cross" or "candidate_cross",
    "confidence": number between 0 and 1,
    "reason": "short reason",
    "keywords": [
      {{"keyword": "keyword phrase", "root": "one allowed English MAG root"}}
    ]
  }}
- If decision is "non_cross", still output up to 10 relevant keywords.
- If decision is "candidate_cross", output at most 10 keywords ranked by relevance.
"""


FIELD_DIC = {
    "Sociology": ["Social Sciences"],
    "Philosophy": ["Social Sciences"],
    "History": ["Arts and Humanities", "Social Sciences"],
    "Medicine": ["Medicine", "Health Professions", "Nursing",
                 "Pharmacology, Toxicology and Pharmaceutics", "Dentistry", "Veterinary"],
    "Engineering": ["Engineering", "Energy"],
    "Art": ["Arts and Humanities"],
    "Computer Science": ["Computer Science"],
    "Computer science": ["Computer Science"],
    "Biology": ["Biochemistry, Genetics and Molecular Biology",
                "Agricultural and Biological Sciences", "Neuroscience", "Immunology and Microbiology"],
    "Environmental Science": ["Environmental Science"],
    "Environmental science": ["Environmental Science"],
    "Geography": ["Environmental Science", "Earth and Planetary Sciences"],
    "Materials Science": ["Materials Science"],
    "Materials science": ["Materials Science"],
    "Physics": ["Physics and Astronomy"],
    "Business": ["Business, Management and Accounting"],
    "Economics": ["Economics, Econometrics and Finance"],
    "Psychology": ["Psychology", "Neuroscience"],
    "Chemistry": ["Chemistry", "Chemical Engineering"],
    "Geology": ["Earth and Planetary Sciences"],
    "Mathematics": ["Mathematics"],
    "Political Science": ["Decision Sciences"],
    "Political science": ["Decision Sciences"],
}

LEXICAL_STOPWORDS = {"of", "and", "the", "to", "in", "for", "on", "with"}


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    decoder = json.JSONDecoder(strict=False)
    buffer = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() and not buffer:
                continue
            buffer += line
            try:
                obj, end = decoder.raw_decode(buffer.lstrip())
                rest = buffer.lstrip()[end:].strip()
                if rest:
                    continue
            except json.JSONDecodeError:
                continue
            rows.append(obj)
            buffer = ""
    return rows


def read_jsonl_map(path: Path, key: str = "doc_id") -> Dict[str, dict]:
    out = {}
    for row in read_jsonl(path):
        val = row.get(key)
        if val:
            out[str(val)] = row
    return out


def write_progress(path: Path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kwargs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seed_stage_file(target_path: Path, seed_results_dir: str, shard_name: str, filename: str):
    if not seed_results_dir or target_path.exists():
        return
    seed_path = Path(seed_results_dir) / shard_name / filename
    if not seed_path.exists() or not seed_path.is_file():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seed_path, target_path)
    print(f"[SEED] copied {filename} from {seed_path} -> {target_path}")


def extract_keywords(answer_text: str, max_k: int = 15) -> List[str]:
    text_in = (answer_text or "").strip().replace("\r\n", "\n")
    candidates = []
    if ("\n" not in text_in) and ("," in text_in):
        candidates = [p.strip() for p in text_in.split(",")]
    else:
        for line in text_in.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*[\-\*\u2022]\s*", "", line)
            line = re.sub(r"^\s*\d+\s*[\.\)\-:]\s*", "", line)
            if "," in line:
                candidates += [p.strip() for p in line.split(",") if p.strip()]
            else:
                candidates.append(line)
    cleaned = [re.sub(r"^[\'\"\s]+|[\'\"\s]+$", "", c) for c in candidates if c]
    seen, out = set(), []
    for kw in cleaned:
        k = kw.lower()
        if k not in seen:
            seen.add(k)
            out.append(kw)
        if len(out) >= max_k:
            break
    return out


def canonical_root_name(root: str) -> str:
    text = normalize_text(root)
    aliases = {normalize_text(name): name for name in MAG_ROOT_LABELS}
    aliases.update({
        "computer science": "Computer Science",
        "computer sciences": "Computer Science",
        "cs": "Computer Science",
        "biology": "Biology",
        "biological sciences": "Biology",
        "business management and accounting": "Business",
        "earth and planetary sciences": "Geology",
        "environmental science": "Environmental Science",
        "environmental sciences": "Environmental Science",
        "arts and humanities": "Art",
        "political science": "Political Science",
        "political sciences": "Political Science",
        "material science": "Materials Science",
        "materials science": "Materials Science",
    })
    return aliases.get(text, "")


def extract_json_object(text_in: str) -> Optional[dict]:
    text_in = (text_in or "").strip()
    if not text_in:
        return None
    try:
        obj = json.loads(text_in)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text_in, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def parse_gated_keyword_response(answer_text: str, max_k: int = 10) -> dict:
    obj = extract_json_object(answer_text)
    if not obj:
        # Fallback keeps old line-based behavior alive if the model fails JSON.
        kws = extract_keywords(answer_text, max_k=max_k)
        return {
            "decision": "candidate_cross",
            "confidence": 0.0,
            "reason": "json_parse_failed",
            "keywords": [{"keyword": kw, "root": ""} for kw in kws],
        }
    decision = str(obj.get("decision") or "candidate_cross").strip().lower()
    if decision not in {"non_cross", "candidate_cross"}:
        decision = "candidate_cross"
    try:
        confidence = float(obj.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    items = []
    for item in obj.get("keywords") or []:
        if not isinstance(item, dict):
            continue
        keyword = str(item.get("keyword") or "").strip()
        root = canonical_root_name(str(item.get("root") or "").strip())
        if keyword:
            items.append({"keyword": keyword, "root": root})
        if len(items) >= max_k:
            break
    return {
        "decision": decision,
        "confidence": confidence,
        "reason": str(obj.get("reason") or "").strip(),
        "keywords": items,
    }


def keyword_terms_and_roots(keyword_row: dict) -> Tuple[List[str], Dict[str, str]]:
    items = keyword_row.get("keyword_roots") or []
    terms = []
    roots = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            keyword = str(item.get("keyword") or "").strip()
            root = canonical_root_name(str(item.get("root") or "").strip())
            if keyword:
                terms.append(keyword)
                if root:
                    roots[norm_kw_full(keyword)] = root
    if not terms:
        terms = [str(kw).strip() for kw in keyword_row.get("keywords", []) if str(kw).strip()]
    root_map = keyword_row.get("keyword_root_map") or {}
    if isinstance(root_map, dict):
        for keyword, root in root_map.items():
            keyword = str(keyword or "").strip()
            root = canonical_root_name(str(root or "").strip())
            if keyword and root:
                roots[norm_kw_full(keyword)] = root
    return terms, roots


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t\r\n:：;,.!?-–—*()[]{}<>\"'`")


def norm_kw(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


def norm_kw_full(s: str) -> str:
    if s is None:
        return ""
    text = unicodedata.normalize("NFKC", str(s))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("_", " ").replace("-", " ")
    return text


def normalize_lexical_text(s: str) -> str:
    if s is None:
        return ""
    text = unicodedata.normalize("NFKC", str(s)).lower().strip()
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lexical_content_tokens(s: str) -> List[str]:
    text = normalize_lexical_text(s)
    if not text:
        return []
    return [tok for tok in text.split() if tok and tok not in LEXICAL_STOPWORDS]


def lexical_fidelity(source_text: str, candidate_text: str) -> float:
    return lexical_match_metrics(source_text, candidate_text)["lexical_fidelity"]


def lexical_match_metrics(source_text: str, candidate_text: str) -> dict:
    src_tokens = lexical_content_tokens(source_text)
    cand_tokens = lexical_content_tokens(candidate_text)
    if not src_tokens or not cand_tokens:
        return {
            "lexical_fidelity": 0.0,
            "lexical_precision": 0.0,
            "lexical_recall": 0.0,
            "shared_token_count": 0,
            "source_token_count": len(src_tokens),
            "candidate_token_count": len(cand_tokens),
            "source_normalized": " ".join(src_tokens),
            "candidate_normalized": " ".join(cand_tokens),
        }
    src_counts = defaultdict(int)
    cand_counts = defaultdict(int)
    for tok in src_tokens:
        src_counts[tok] += 1
    for tok in cand_tokens:
        cand_counts[tok] += 1
    shared = 0
    for tok, cnt in src_counts.items():
        shared += min(cnt, cand_counts.get(tok, 0))
    if shared <= 0:
        return {
            "lexical_fidelity": 0.0,
            "lexical_precision": 0.0,
            "lexical_recall": 0.0,
            "shared_token_count": 0,
            "source_token_count": len(src_tokens),
            "candidate_token_count": len(cand_tokens),
            "source_normalized": " ".join(src_tokens),
            "candidate_normalized": " ".join(cand_tokens),
        }
    precision = shared / max(len(cand_tokens), 1)
    recall = shared / max(len(src_tokens), 1)
    fidelity = 0.0
    if precision + recall <= 0:
        fidelity = 0.0
    else:
        fidelity = float(2.0 * precision * recall / (precision + recall))
    return {
        "lexical_fidelity": fidelity,
        "lexical_precision": float(precision),
        "lexical_recall": float(recall),
        "shared_token_count": int(shared),
        "source_token_count": len(src_tokens),
        "candidate_token_count": len(cand_tokens),
        "source_normalized": " ".join(src_tokens),
        "candidate_normalized": " ".join(cand_tokens),
    }


def is_strong_lexical_candidate(metrics: dict, threshold: float) -> bool:
    if float(metrics.get("lexical_fidelity") or 0.0) < threshold:
        return False
    source_count = int(metrics.get("source_token_count") or 0)
    shared_count = int(metrics.get("shared_token_count") or 0)
    if source_count >= 2:
        return shared_count >= 2
    if source_count == 1:
        return (
            int(metrics.get("candidate_token_count") or 0) == 1
            and str(metrics.get("source_normalized") or "") == str(metrics.get("candidate_normalized") or "")
        )
    return False


def l2n_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-10
    return a / n


def embed_texts(texts: List[str], model, batch_size: int, device: str) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        device=device,
        show_progress_bar=False,
    ).astype("float32", copy=False)
    return l2n_rows(embs)


def add_prefix_query(x: str, mode: str) -> str:
    return f"query: {x}" if mode == "e5" else x


def add_prefix_passage(x: str, mode: str) -> str:
    return f"passage: {x}" if mode == "e5" else x


def topk_ids(scores: np.ndarray, ids: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        return np.array([], dtype=ids.dtype), np.array([], dtype=scores.dtype)
    if len(scores) <= k:
        idx = np.argsort(scores)[::-1]
        return ids[idx], scores[idx]
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return ids[idx], scores[idx]


class MAGCache:
    def __init__(self, engine, table_field: str, table_rel: str):
        self.engine = engine
        self.table_field = table_field
        self.table_rel = table_rel
        self.cache_level_nodes: Dict[int, pd.DataFrame] = {}
        self.cache_children: Dict[Tuple[int, ...], pd.DataFrame] = {}
        self.embed_cache: Dict[int, np.ndarray] = {}
        self.name_cache: Dict[int, str] = {}
        self.level_cache: Dict[int, Optional[int]] = {}
        self.parents_cache: Dict[int, set] = {}

    def level_nodes(self, level: int) -> pd.DataFrame:
        if level in self.cache_level_nodes:
            return self.cache_level_nodes[level]
        sql = text(f"""
            SELECT field_id, name, level
            FROM {self.table_field}
            WHERE name IS NOT NULL AND TRIM(name) <> '' AND level = :lvl
            ORDER BY field_id
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"lvl": level})
        df["field_id"] = df["field_id"].astype("int64")
        self.cache_level_nodes[level] = df
        self._remember_nodes(df)
        return df

    def children_of(self, parent_ids: Iterable[int]) -> pd.DataFrame:
        key = tuple(sorted(set(map(int, parent_ids))))
        if not key:
            return pd.DataFrame(columns=["field_id", "name", "level"])
        if key in self.cache_children:
            return self.cache_children[key]
        chunks = []
        for i in range(0, len(key), 1000):
            part = key[i:i + 1000]
            sql = text(f"""
                SELECT f.field_id, f.name, f.level
                FROM {self.table_field} f
                JOIN {self.table_rel} r ON f.field_id = r.field_id
                WHERE r.parent_id IN :pids
                  AND f.name IS NOT NULL AND TRIM(f.name) <> ''
            """).bindparams(bindparam("pids", expanding=True))
            with self.engine.connect() as conn:
                chunks.append(pd.read_sql(sql, conn, params={"pids": tuple(part)}))
        out = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["field_id"])
        if not out.empty:
            out["field_id"] = out["field_id"].astype("int64")
        self.cache_children[key] = out
        self._remember_nodes(out)
        return out

    def ensure_embeds(self, ids: List[int], model, batch_size: int, device: str, prefix_mode: str):
        need = [int(i) for i in ids if int(i) not in self.embed_cache]
        if not need:
            return
        self.ensure_nodes(need)
        names = [add_prefix_passage(normalize_text(self.name_cache[i]), prefix_mode) for i in need]
        embs = embed_texts(names, model, batch_size=batch_size, device=device)
        for fid, vec in zip(need, embs):
            self.embed_cache[fid] = vec

    def ensure_nodes(self, ids: Iterable[int]):
        ids = [int(x) for x in ids]
        need = [i for i in ids if i not in self.name_cache or i not in self.level_cache]
        if need:
            sql = text(f"""
                SELECT field_id, name, level
                FROM {self.table_field}
                WHERE field_id IN :ids
            """).bindparams(bindparam("ids", expanding=True))
            with self.engine.connect() as conn:
                df = pd.read_sql(sql, conn, params={"ids": tuple(set(need))})
            self._remember_nodes(df)
            for i in need:
                self.name_cache.setdefault(i, "")
                self.level_cache.setdefault(i, None)
        self.ensure_parents(ids)

    def ensure_parents(self, ids: Iterable[int]):
        need = [int(i) for i in ids if int(i) not in self.parents_cache]
        if not need:
            return
        sql = text(f"""
            SELECT field_id, parent_id
            FROM {self.table_rel}
            WHERE field_id IN :ids
        """).bindparams(bindparam("ids", expanding=True))
        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"ids": tuple(set(need))})
        tmp = {i: set() for i in need}
        for _, row in df.iterrows():
            tmp[int(row["field_id"])].add(int(row["parent_id"]))
        for i in need:
            self.parents_cache[i] = tmp.get(i, set())

    def _remember_nodes(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        for row in df.itertuples(index=False):
            fid = int(row.field_id)
            self.name_cache[fid] = str(row.name)
            self.level_cache[fid] = None if pd.isna(row.level) else int(row.level)

    def name(self, fid: int) -> str:
        self.ensure_nodes([fid])
        return self.name_cache.get(int(fid), "")

    def level(self, fid: int) -> Optional[int]:
        self.ensure_nodes([fid])
        return self.level_cache.get(int(fid))

    def parents(self, fid: int) -> set:
        self.ensure_nodes([fid])
        return self.parents_cache.get(int(fid), set())


class OfflineMAGCache:
    """MAGCache-compatible reader backed by local node JSONL files and relation parquet."""

    def __init__(self, cache_dir: str, nodes_dir: str):
        self.cache_dir = Path(cache_dir)
        self.nodes_dir = Path(nodes_dir)
        relation_path = self.cache_dir / "field_relation.parquet"
        if not relation_path.exists():
            raise FileNotFoundError(f"MAG offline relation parquet not found: {relation_path}")
        self.cache_level_nodes: Dict[int, pd.DataFrame] = {}
        self.cache_children: Dict[Tuple[int, ...], pd.DataFrame] = {}
        self.embed_cache: Dict[int, np.ndarray] = {}
        self.name_cache: Dict[int, str] = {}
        self.level_cache: Dict[int, Optional[int]] = {}
        self.parents_cache: Dict[int, set] = {}
        self.children_by_parent: Dict[int, List[int]] = defaultdict(list)
        self._load_nodes()
        self._load_relations(relation_path)

    def _load_nodes(self):
        rows = []
        node_paths = sorted(self.nodes_dir.glob("level_*_nodes.jsonl"))
        if not node_paths:
            raise FileNotFoundError(f"MAG offline node files not found under: {self.nodes_dir}")
        for path in node_paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    fid = int(row["field_id"])
                    name = str(row.get("name") or "")
                    level = int(row.get("level")) if row.get("level") is not None else None
                    rows.append({"field_id": fid, "name": name, "level": level})
                    self.name_cache[fid] = name
                    self.level_cache[fid] = level
        self.nodes_df = pd.DataFrame(rows)
        if self.nodes_df.empty:
            raise RuntimeError(f"MAG offline node files are empty under: {self.nodes_dir}")
        self.nodes_df["field_id"] = self.nodes_df["field_id"].astype("int64")
        self.node_index = self.nodes_df.set_index("field_id", drop=False)

    def _load_relations(self, relation_path: Path):
        rel = pd.read_parquet(relation_path, columns=["field_id", "parent_id"])
        rel["field_id"] = rel["field_id"].astype("int64")
        rel["parent_id"] = rel["parent_id"].astype("int64")
        parents_tmp = defaultdict(set)
        children_tmp = defaultdict(list)
        for fid, parent_id in rel[["field_id", "parent_id"]].itertuples(index=False, name=None):
            fid = int(fid)
            parent_id = int(parent_id)
            parents_tmp[fid].add(parent_id)
            children_tmp[parent_id].append(fid)
        self.parents_cache = dict(parents_tmp)
        self.children_by_parent = dict(children_tmp)
        del rel

    def level_nodes(self, level: int) -> pd.DataFrame:
        level = int(level)
        if level not in self.cache_level_nodes:
            df = self.nodes_df[self.nodes_df["level"] == level].copy()
            self.cache_level_nodes[level] = df
        return self.cache_level_nodes[level]

    def children_of(self, parent_ids: Iterable[int]) -> pd.DataFrame:
        key = tuple(sorted(set(map(int, parent_ids))))
        if not key:
            return pd.DataFrame(columns=["field_id", "name", "level"])
        if key in self.cache_children:
            return self.cache_children[key]
        child_ids = []
        seen = set()
        for parent_id in key:
            for child_id in self.children_by_parent.get(int(parent_id), []):
                if child_id not in seen and child_id in self.name_cache:
                    seen.add(child_id)
                    child_ids.append(child_id)
        if not child_ids:
            out = pd.DataFrame(columns=["field_id", "name", "level"])
        else:
            out = self.node_index.loc[child_ids, ["field_id", "name", "level"]].reset_index(drop=True)
        self.cache_children[key] = out
        return out

    def ensure_embeds(self, ids: List[int], model, batch_size: int, device: str, prefix_mode: str):
        need = [int(i) for i in ids if int(i) not in self.embed_cache]
        if not need:
            return
        names = [add_prefix_passage(normalize_text(self.name_cache.get(i, "")), prefix_mode) for i in need]
        embs = embed_texts(names, model, batch_size=batch_size, device=device)
        for fid, vec in zip(need, embs):
            self.embed_cache[fid] = vec

    def ensure_nodes(self, ids: Iterable[int]):
        for fid in ids:
            fid = int(fid)
            self.name_cache.setdefault(fid, "")
            self.level_cache.setdefault(fid, None)
            self.parents_cache.setdefault(fid, set())

    def ensure_parents(self, ids: Iterable[int]):
        self.ensure_nodes(ids)

    def name(self, fid: int) -> str:
        return self.name_cache.get(int(fid), "")

    def level(self, fid: int) -> Optional[int]:
        return self.level_cache.get(int(fid))

    def parents(self, fid: int) -> set:
        return self.parents_cache.get(int(fid), set())


def create_mysql_engine(args):
    url = URL.create(
        "mysql+pymysql",
        username=args.db_user,
        password=args.db_pass,
        host=args.db_host,
        port=args.db_port,
        database=args.db_name,
        query={"charset": "utf8mb4"},
    )
    return create_engine(url, pool_pre_ping=True)


def create_mag_cache(args):
    offline_dir = str(getattr(args, "mag_offline_cache_dir", "") or "").strip()
    if offline_dir:
        return OfflineMAGCache(offline_dir, args.mag_hash_index_dir), None
    engine = create_mysql_engine(args)
    return MAGCache(engine, args.table_field, args.table_rel), engine



def select_mag_candidate_from_pool(term: str, candidate_pool: List[dict], args) -> Tuple[Optional[dict], str]:
    if not candidate_pool:
        return None, "no_candidate"
    def numeric_value(row: dict, key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key) or default)
        except (TypeError, ValueError):
            return float(default)
    anchored_candidates = []
    strong_candidates = []
    for cand in candidate_pool:
        cand_name = str(cand.get("mag_name") or "")
        metrics = lexical_match_metrics(term, cand_name)
        cand.update(metrics)
        cand["is_strong_anchor"] = is_strong_lexical_candidate(metrics, args.lexical_fidelity_threshold)
        if cand["lexical_fidelity"] >= args.lexical_fidelity_threshold:
            anchored_candidates.append(cand)
        if cand["is_strong_anchor"]:
            strong_candidates.append(cand)
    if args.algo_profile in {"revised_v4", "revised_v5"} and strong_candidates:
        return max(strong_candidates, key=lambda row: (row["score"], row["lexical_fidelity"], -row["field_id"])), "lexical_rerank"
    if args.algo_profile not in {"revised_v4", "revised_v5"} and anchored_candidates:
        return max(anchored_candidates, key=lambda row: (row["score"], row["lexical_fidelity"], -row["field_id"])), "lexical_rerank"
    return max(
        candidate_pool,
        key=lambda row: (
            numeric_value(row, "score"),
            numeric_value(row, "cluster_score"),
            numeric_value(row, "lexical_fidelity"),
            -int(row["field_id"]),
        ),
    ), "embedding_fallback"


def no_mag_candidate_row(term: str, mode: str = "no_candidate") -> dict:
    return {
        "term": term,
        "mag_field_id": -1,
        "mag_name": "",
        "score": 0.0,
        "final_level": -1,
        "lexical_fidelity": 0.0,
        "lexical_precision": 0.0,
        "lexical_recall": 0.0,
        "shared_token_count": 0,
        "source_token_count": 0,
        "candidate_token_count": 0,
        "is_strong_anchor": False,
        "is_anchored": False,
        "match_selection_mode": mode,
    }


def chosen_mag_row(term: str, chosen: dict, selection_mode: str, args, extra: Optional[dict] = None) -> dict:
    if args.algo_profile in {"revised_v4", "revised_v5"}:
        is_anchor = bool(chosen.get("is_strong_anchor"))
    else:
        is_anchor = bool(float(chosen.get("lexical_fidelity") or 0.0) >= args.lexical_fidelity_threshold)
    row = {
        "term": term,
        "mag_field_id": int(chosen["field_id"]),
        "mag_name": str(chosen.get("mag_name") or ""),
        "score": float(chosen.get("score") or 0.0),
        "final_level": int(chosen.get("level") if chosen.get("level") is not None else -1),
        "lexical_fidelity": float(chosen.get("lexical_fidelity") or 0.0),
        "lexical_precision": float(chosen.get("lexical_precision") or 0.0),
        "lexical_recall": float(chosen.get("lexical_recall") or 0.0),
        "shared_token_count": int(chosen.get("shared_token_count") or 0),
        "source_token_count": int(chosen.get("source_token_count") or 0),
        "candidate_token_count": int(chosen.get("candidate_token_count") or 0),
        "is_strong_anchor": bool(chosen.get("is_strong_anchor") or False),
        "is_anchored": is_anchor,
        "match_selection_mode": selection_mode,
    }
    if extra:
        row.update(extra)
    return row


def level0_root_maps(mag: MAGCache, args) -> Tuple[Dict[str, int], Dict[int, str]]:
    lv0 = mag.level_nodes(args.start_level)
    name_to_id = {}
    id_to_name = {}
    for row in lv0.itertuples(index=False):
        fid = int(row.field_id)
        name = str(row.name or "").strip()
        if not name:
            continue
        id_to_name[fid] = name
        name_to_id[normalize_text(name)] = fid
        canonical = canonical_root_name(name)
        if canonical:
            name_to_id[normalize_text(canonical)] = fid
    return name_to_id, id_to_name


def desired_root_for_term(term: str, root_by_term: Optional[Dict[str, str]]) -> str:
    if not root_by_term:
        return ""
    return canonical_root_name(root_by_term.get(norm_kw_full(term), ""))


def path_to_root(mag: MAGCache, start_id: int, root_id: int, max_hops: int = 256) -> List[int]:
    start_id = int(start_id)
    root_id = int(root_id)
    stack = [(start_id, [start_id])]
    visited = set()
    steps = 0
    while stack:
        cur, path = stack.pop()
        if cur == root_id:
            return path
        if cur in visited:
            continue
        visited.add(cur)
        steps += 1
        if steps > max_hops * 1000:
            break
        for parent in sorted(mag.parents(cur)):
            if parent not in path:
                stack.append((parent, path + [parent]))
    return []



def _hash_level_candidates(mag_hash_index: MAGHashIndex, level: int, q: np.ndarray,
                           top_clusters: int, top_nodes: int) -> List[dict]:
    data = mag_hash_index.level_data[int(level)]
    centroids = data["centroids"]
    if centroids.size == 0:
        return []
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    q = q / (float(np.linalg.norm(q)) + 1e-10)
    scores = centroids @ q
    k = max(1, min(int(top_clusters), len(scores)))
    if len(scores) <= k:
        cidx = np.argsort(scores)[::-1]
    else:
        cidx = np.argpartition(scores, -k)[-k:]
        cidx = cidx[np.argsort(scores[cidx])[::-1]]
    node_indices = []
    cluster_score_by_code = {}
    for idx in cidx.tolist():
        code = int(data["unique_codes"][idx])
        cluster_score_by_code[code] = float(scores[idx])
        sl = data["code_to_slice"].get(code)
        if sl is not None:
            node_indices.extend(range(sl.start, sl.stop))
    if not node_indices:
        return []
    node_indices = np.asarray(sorted(set(node_indices)), dtype=np.int64)
    node_scores = data["embeddings"][node_indices] @ q
    max_nodes = max(1, int(top_nodes))
    if len(node_scores) > max_nodes:
        keep = np.argpartition(node_scores, -max_nodes)[-max_nodes:]
        keep = keep[np.argsort(node_scores[keep])[::-1]]
    else:
        keep = np.argsort(node_scores)[::-1]
    candidates = []
    for local in keep.tolist():
        idx = int(node_indices[local])
        code = int(data["codes"][idx])
        candidates.append({
            "field_id": int(data["ids"][idx]),
            "mag_name": str(data["names"][idx]),
            "level": int(level),
            "score": float(node_scores[local]),
            "hash_code": code,
            "cluster_score": float(cluster_score_by_code.get(code, 0.0)),
        })
    return candidates


def match_keywords_to_mag_hash_topdown(
    terms: List[str],
    args,
    model,
    mag_hash_index: MAGHashIndex,
    mag: MAGCache,
    root_by_term: Optional[Dict[str, str]] = None,
):
    """Top-down single-path MAG grounding.

    Each keyword first selects exactly one level-0 root using the hash index.
    Then matching is constrained to descendants of that root, one best child per
    level. The returned row carries a single leaf-to-root path so MAG concepts do
    not expand to multiple roots through multi-parent backtracking.
    """
    if not terms:
        return []
    terms_norm = [add_prefix_query(normalize_text(t), args.prefix_mode) for t in terms]
    e_terms = embed_texts(terms_norm, model, batch_size=args.batch_enc, device=args.device)
    if bool(getattr(args, "mag_hash_fast_mode", False)):
        top_clusters = min(int(args.mag_hash_top_clusters_per_level), 4)
        top_nodes = min(int(args.mag_hash_max_candidate_nodes), 128)
    else:
        top_clusters = int(args.mag_hash_top_clusters_per_level)
        top_nodes = int(args.mag_hash_max_candidate_nodes)
    final_rows = []
    root_name_to_id, root_id_to_name = level0_root_maps(mag, args)
    for term, eterm in zip(terms, e_terms):
        desired_root = desired_root_for_term(term, root_by_term)
        desired_root_id = root_name_to_id.get(normalize_text(desired_root)) if desired_root else None
        if desired_root_id is not None:
            root = {
                "field_id": int(desired_root_id),
                "mag_name": root_id_to_name.get(int(desired_root_id), desired_root),
                "level": int(args.start_level),
                "score": 1.0,
            }
            root_mode = "llm_root"
        else:
            root_candidates = _hash_level_candidates(mag_hash_index, int(args.start_level), eterm, top_clusters, top_nodes)
            root, root_mode = select_mag_candidate_from_pool(term, root_candidates, args)
            if root is None:
                final_rows.append(no_mag_candidate_row(term, "hash_topdown_no_root"))
                continue
        root_id = int(root["field_id"])
        path_root_to_leaf = [root_id]
        current_parent = root_id
        visited = [dict(root)]
        for level in range(int(args.start_level) + 1, int(args.max_level) + 1):
            cand_df = mag.children_of([current_parent])
            if cand_df.empty:
                break
            cand_df = cand_df[cand_df["level"] == level].drop_duplicates(subset=["field_id"])
            if cand_df.empty:
                break
            cand_ids = cand_df["field_id"].astype("int64").to_numpy()
            mag.ensure_embeds(cand_ids.tolist(), model, args.batch_enc, args.device, args.prefix_mode)
            e_cand = np.stack([mag.embed_cache[int(i)] for i in cand_ids], axis=0)
            scores = (e_cand @ eterm.reshape(-1, 1)).squeeze(1)
            best_idx = int(np.argmax(scores))
            best_id = int(cand_ids[best_idx])
            best = {
                "field_id": best_id,
                "mag_name": mag.name(best_id),
                "level": int(level),
                "score": float(scores[best_idx]),
                "hash_code": None,
                "cluster_score": None,
            }
            visited.append(best)
            path_root_to_leaf.append(best_id)
            current_parent = best_id
        chosen, selection_mode = select_mag_candidate_from_pool(term, visited, args)
        if chosen is None:
            final_rows.append(no_mag_candidate_row(term, "hash_topdown_no_candidate"))
            continue
        chosen_id = int(chosen["field_id"])
        if chosen_id in path_root_to_leaf:
            chosen_pos = path_root_to_leaf.index(chosen_id)
            path_root_to_chosen = path_root_to_leaf[:chosen_pos + 1]
        else:
            path_root_to_chosen = path_root_to_leaf
        extra = {
            "hash_topdown_root_id": root_id,
            "hash_topdown_root_name": str(root.get("mag_name") or ""),
            "hash_topdown_root_mode": root_mode,
            "hash_topdown_path_ids": list(reversed(path_root_to_chosen)),
            "hash_topdown_path_root_to_leaf": path_root_to_chosen,
            "hash_topdown_visited_count": len(visited),
            "hash_fast_mode": bool(getattr(args, "mag_hash_fast_mode", False)),
            "llm_root_id": int(root_id) if desired_root_id is not None else None,
            "llm_root_name": root_id_to_name.get(int(root_id), str(root.get("mag_name") or "")) if desired_root_id is not None else "",
        }
        final_rows.append(chosen_mag_row(term, chosen, "hash_topdown_" + selection_mode, args, extra=extra))
    return final_rows

def match_keywords_to_mag_hash(
    terms: List[str],
    args,
    model,
    mag_hash_index: MAGHashIndex,
    mag: Optional[MAGCache] = None,
    root_by_term: Optional[Dict[str, str]] = None,
):
    if not terms:
        return []
    terms_norm = [add_prefix_query(normalize_text(t), args.prefix_mode) for t in terms]
    e_terms = embed_texts(terms_norm, model, batch_size=args.batch_enc, device=args.device)
    final_rows = []
    if bool(getattr(args, "mag_hash_fast_mode", False)):
        top_clusters_per_level = min(int(args.mag_hash_top_clusters_per_level), 4)
        target_level_top_clusters = min(int(args.mag_hash_target_level_top_clusters), 6)
        max_candidate_nodes = min(int(args.mag_hash_max_candidate_nodes), 128)
    else:
        top_clusters_per_level = int(args.mag_hash_top_clusters_per_level)
        target_level_top_clusters = int(args.mag_hash_target_level_top_clusters)
        max_candidate_nodes = int(args.mag_hash_max_candidate_nodes)
    if hasattr(mag_hash_index, "query_batch"):
        queries = mag_hash_index.query_batch(
            e_terms,
            top_clusters_per_level=top_clusters_per_level,
            target_level_top_clusters=target_level_top_clusters,
            max_candidate_nodes=max_candidate_nodes,
        )
    else:
        queries = [
            mag_hash_index.query(
                eterm,
                top_clusters_per_level=top_clusters_per_level,
                target_level_top_clusters=target_level_top_clusters,
                max_candidate_nodes=max_candidate_nodes,
            )
            for eterm in e_terms
        ]
    root_name_to_id, root_id_to_name = level0_root_maps(mag, args) if mag is not None else ({}, {})
    for term, query in zip(terms, queries):
        candidates = query.get("candidates") or []
        desired_root = desired_root_for_term(term, root_by_term)
        desired_root_id = root_name_to_id.get(normalize_text(desired_root)) if desired_root else None
        if desired_root_id is not None and mag is not None:
            rooted_candidates = []
            for cand in candidates:
                path = path_to_root(mag, int(cand.get("field_id", -1)), int(desired_root_id))
                if path:
                    cand = dict(cand)
                    cand["topdown_path_ids"] = path
                    rooted_candidates.append(cand)
            candidates = rooted_candidates
        chosen, selection_mode = select_mag_candidate_from_pool(term, candidates, args)
        if chosen is None:
            final_rows.append(no_mag_candidate_row(term, "hash_cluster_no_candidate"))
            continue
        chosen_path = []
        if desired_root_id is not None and mag is not None:
            chosen_path = chosen.get("topdown_path_ids") or path_to_root(mag, int(chosen["field_id"]), int(desired_root_id))
        extra = {
            "hash_target_level": query.get("target_level"),
            "hash_level_votes": query.get("level_votes"),
            "hash_code": chosen.get("hash_code"),
            "hash_cluster_score": chosen.get("cluster_score"),
            "hash_candidate_count": len(candidates),
            "hash_fast_mode": bool(getattr(args, "mag_hash_fast_mode", False)),
            "topdown_path_ids": chosen_path,
            "llm_root_id": int(desired_root_id) if desired_root_id is not None else None,
            "llm_root_name": root_id_to_name.get(int(desired_root_id), desired_root) if desired_root_id is not None else "",
        }
        final_rows.append(chosen_mag_row(term, chosen, "hash_cluster_" + selection_mode, args, extra=extra))
    return final_rows

def match_keywords_to_mag(
    terms: List[str],
    args,
    model,
    mag: MAGCache,
    e_lv0,
    lv0_ids,
    root_by_term: Optional[Dict[str, str]] = None,
):
    if not terms:
        return []
    terms_norm = [add_prefix_query(normalize_text(t), args.prefix_mode) for t in terms]
    e_terms = embed_texts(terms_norm, model, batch_size=args.batch_enc, device=args.device)
    final_rows = []

    root_name_to_id, root_id_to_name = level0_root_maps(mag, args)
    lv0_pos = {int(fid): pos for pos, fid in enumerate(lv0_ids.tolist())} if len(lv0_ids) else {}
    for term, eterm in zip(terms, e_terms):
        desired_root = desired_root_for_term(term, root_by_term)
        desired_root_id = root_name_to_id.get(normalize_text(desired_root)) if desired_root else None
        scores0 = (e_lv0 @ eterm.reshape(-1, 1)).squeeze(1)
        if desired_root_id is not None:
            root_pos = lv0_pos.get(int(desired_root_id))
            root_score = float(scores0[root_pos]) if root_pos is not None else 1.0
            top_ids = np.array([int(desired_root_id)], dtype=np.int64)
            top_scores = np.array([root_score], dtype=np.float32)
        else:
            k0 = max(1, min(args.topk_lv0, len(scores0)))
            top_ids, top_scores = topk_ids(scores0, lv0_ids, k0)
        cur_parents = top_ids.tolist()
        visited_best = [(args.start_level, int(fid), float(sc)) for fid, sc in zip(top_ids, top_scores)]

        for level in range(args.start_level + 1, args.max_level + 1):
            cand_df = mag.children_of(cur_parents)
            if cand_df.empty:
                break
            cand_df = cand_df[cand_df["level"] == level].drop_duplicates(subset=["field_id"])
            if cand_df.empty:
                break
            cand_ids = cand_df["field_id"].astype("int64").to_numpy()
            mag.ensure_embeds(cand_ids.tolist(), model, args.batch_enc, args.device, args.prefix_mode)
            e_cand = np.stack([mag.embed_cache[int(i)] for i in cand_ids], axis=0)
            scores = (e_cand @ eterm.reshape(-1, 1)).squeeze(1)
            keep = max(1, int(math.ceil(len(scores) * max(0.0, min(1.0, args.keep_percent)))))
            top_ids, top_scores = topk_ids(scores, cand_ids, keep)
            cur_parents = top_ids.tolist()
            visited_best.extend([(level, int(fid), float(sc)) for fid, sc in zip(top_ids, top_scores)])

        if visited_best:
            best_by_id = {}
            for level, fid, score in visited_best:
                prev = best_by_id.get(int(fid))
                if prev is None or float(score) > prev["score"]:
                    best_by_id[int(fid)] = {
                        "field_id": int(fid),
                        "level": int(level),
                        "score": float(score),
                    }
            candidate_pool = sorted(
                best_by_id.values(),
                key=lambda row: (-row["score"], row["field_id"]),
            )[: max(1, int(args.candidate_pool_size))]
            anchored_candidates = []
            strong_candidates = []
            for cand in candidate_pool:
                cand_name = mag.name(cand["field_id"])
                metrics = lexical_match_metrics(term, cand_name)
                cand["mag_name"] = cand_name
                cand.update(metrics)
                cand["is_strong_anchor"] = is_strong_lexical_candidate(metrics, args.lexical_fidelity_threshold)
                if cand["lexical_fidelity"] >= args.lexical_fidelity_threshold:
                    anchored_candidates.append(cand)
                if cand["is_strong_anchor"]:
                    strong_candidates.append(cand)
            if args.algo_profile in {"revised_v4", "revised_v5"} and strong_candidates:
                chosen_pool = strong_candidates
                selection_mode = "lexical_rerank"
            elif args.algo_profile not in {"revised_v4", "revised_v5"} and anchored_candidates:
                chosen_pool = anchored_candidates
                selection_mode = "lexical_rerank"
            else:
                chosen_pool = candidate_pool
                selection_mode = "embedding_fallback"
            chosen = max(
                chosen_pool,
                key=lambda row: (row["score"], row["lexical_fidelity"], -row["field_id"]),
            )
            if args.algo_profile in {"revised_v4", "revised_v5"}:
                is_anchor = bool(chosen["is_strong_anchor"])
            else:
                is_anchor = bool(chosen["lexical_fidelity"] >= args.lexical_fidelity_threshold)
            extra = {}
            if desired_root_id is not None:
                chosen_path = path_to_root(mag, int(chosen["field_id"]), int(desired_root_id))
                extra = {
                    "topdown_path_ids": chosen_path,
                    "llm_root_id": int(desired_root_id),
                    "llm_root_name": root_id_to_name.get(int(desired_root_id), desired_root),
                }
            row = {
                "term": term,
                "mag_field_id": chosen["field_id"],
                "mag_name": chosen["mag_name"],
                "score": chosen["score"],
                "final_level": chosen["level"],
                "lexical_fidelity": float(chosen["lexical_fidelity"]),
                "lexical_precision": float(chosen.get("lexical_precision") or 0.0),
                "lexical_recall": float(chosen.get("lexical_recall") or 0.0),
                "shared_token_count": int(chosen.get("shared_token_count") or 0),
                "source_token_count": int(chosen.get("source_token_count") or 0),
                "candidate_token_count": int(chosen.get("candidate_token_count") or 0),
                "is_strong_anchor": bool(chosen["is_strong_anchor"]),
                "is_anchored": is_anchor,
                "match_selection_mode": selection_mode,
            }
            row.update(extra)
            final_rows.append(row)
        else:
            final_rows.append({
                "term": term,
                "mag_field_id": -1,
                "mag_name": "",
                "score": 0.0,
                "final_level": -1,
                "lexical_fidelity": 0.0,
                "lexical_precision": 0.0,
                "lexical_recall": 0.0,
                "shared_token_count": 0,
                "source_token_count": 0,
                "candidate_token_count": 0,
                "is_strong_anchor": False,
                "is_anchored": False,
                "match_selection_mode": "no_candidate",
            })
    return final_rows


def all_paths_to_level0(mag: MAGCache, start_id: int, max_hops: int = 256):
    start_id = int(start_id)
    paths = []
    stack = [(start_id, [start_id])]
    steps = 0
    while stack:
        cur, path = stack.pop()
        steps += 1
        if steps > max_hops * 1000:
            paths.append(path)
            break
        level = mag.level(cur)
        parents = mag.parents(cur)
        if level == 0 or not parents or level is None:
            paths.append(path)
            continue
        branched = False
        for parent in sorted(parents):
            if parent in path:
                paths.append(path)
                continue
            branched = True
            stack.append((parent, path + [parent]))
        if not branched:
            paths.append(path)
    return paths


def build_mag_payload(doc_id: str, matches: List[dict], mag: MAGCache):
    paths_out = []
    seen_ids = set()
    ordered_ids = []
    for row in matches:
        fid = int(row.get("mag_field_id", -1))
        if fid < 0:
            continue
        explicit_path = row.get("hash_topdown_path_ids") or row.get("topdown_path_ids")
        if explicit_path:
            paths_iter = [[int(x) for x in explicit_path]]
        else:
            paths_iter = all_paths_to_level0(mag, fid)
        for path_ids in paths_iter:
            names = [mag.name(x) for x in path_ids]
            levels = [mag.level(x) for x in path_ids]
            paths_out.append({
                "keyword_field_id": fid,
                "path_ids": path_ids,
                "path_names": names,
                "path_levels": levels,
            })
            for x in path_ids:
                if x not in seen_ids:
                    seen_ids.add(x)
                    ordered_ids.append(x)

    mag.ensure_nodes(ordered_ids)
    nodes = [
        {"field_id": fid, "name": mag.name(fid), "level": mag.level(fid)}
        for fid in sorted(ordered_ids)
    ]
    prompt_keywords = [n["name"] for n in nodes if n["name"]]
    return {
        "doc_id": doc_id,
        "matches": matches,
        "paths": paths_out,
        "nodes": nodes,
        "prompt_keywords": prompt_keywords,
    }


def build_predecessor_style_prompt_keywords(paths: List[dict], mag: MAGCache) -> List[str]:
    seen_ids = set()
    unique_ids = []
    for path in paths or []:
        for fid in path.get("path_ids", []) or []:
            fid_int = int(fid)
            if fid_int not in seen_ids:
                seen_ids.add(fid_int)
                unique_ids.append(fid_int)
    if not unique_ids:
        return []
    mag.ensure_nodes(unique_ids)
    keywords = []
    for fid in sorted(unique_ids):
        name = str(mag.name(fid) or "").strip()
        if name:
            keywords.append(name)
    return keywords


def apply_profile_defaults(args):
    if args.algo_profile in {"revised_v2", "revised_v3", "revised_v4", "revised_v5"}:
        args.keyword_input = "title_abstract"
        args.max_keywords = 10
    return args


def read_text_prompt(doc: dict, keyword_input: str) -> str:
    title = doc.get("title", "")
    abstract = doc.get("abs", "")
    if keyword_input == "title_abstract":
        return f"Title: {title}\nAbstract: {abstract}"
    return abstract


def parse_int_list_csv(value: str) -> List[int]:
    out = []
    for part in re.split(r"[\s,]+", str(value or "").strip()):
        if not part:
            continue
        out.append(int(part))
    return out


def is_cuda_oom(exc: Exception) -> bool:
    oom_error = getattr(torch, "OutOfMemoryError", None)
    if oom_error is not None and isinstance(exc, oom_error):
        return True
    return "out of memory" in repr(exc).lower()


def clear_torch_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_keywords_for_doc(doc, model, tokenizer, args, max_input_tokens: Optional[int] = None):
    user_prompt = read_text_prompt(doc, args.keyword_input)
    if bool(getattr(args, "llm_cross_gate", True)):
        system_prompt = GATED_SYSTEM_PROMPT.format(roots="\n".join(f"- {root}" for root in MAG_ROOT_PROMPT_LABELS))
    else:
        system_prompt = SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        padding=True,
        tokenize=False,
        add_generation_prompt=True,
    )
    tok_kwargs = {"return_tensors": "pt"}
    if max_input_tokens and max_input_tokens > 0:
        tok_kwargs.update({"truncation": True, "max_length": max_input_tokens})
    enc = tokenizer(prompt_text, **tok_kwargs).to(model.device)
    out_ids = model.generate(
        **enc,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_p=0.7,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(out_ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    if bool(getattr(args, "llm_cross_gate", True)):
        parsed = parse_gated_keyword_response(answer, max_k=args.max_keywords)
        kws = [item["keyword"] for item in parsed["keywords"]]
        return kws, answer, parsed
    return extract_keywords(answer, max_k=args.max_keywords), answer, {
        "decision": "candidate_cross",
        "confidence": 0.0,
        "reason": "llm_cross_gate_disabled",
        "keywords": [],
    }


def generate_keywords_with_retry(doc, model, tokenizer, args):
    try:
        return generate_keywords_for_doc(doc, model, tokenizer, args)
    except Exception as exc:
        retry_limits = parse_int_list_csv(args.keyword_oom_retry_tokens)
        if not retry_limits or not is_cuda_oom(exc):
            raise
        last_exc = exc
        clear_torch_memory()
        for max_input_tokens in retry_limits:
            try:
                print(
                    f"[RETRY] doc_id={doc.get('doc_id')} phase=keywords "
                    f"reason=oom max_input_tokens={max_input_tokens}"
                )
                return generate_keywords_for_doc(
                    doc,
                    model,
                    tokenizer,
                    args,
                    max_input_tokens=max_input_tokens,
                )
            except Exception as inner_exc:
                last_exc = inner_exc
                if not is_cuda_oom(inner_exc):
                    raise
                clear_torch_memory()
        raise last_exc


def clean_for_prompt(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s.replace('"', r'\"')


def get_cutoff_year(doc: dict, cutoff_field: str = "year") -> Optional[int]:
    field_candidates = []
    for field in str(cutoff_field or "").split(","):
        field = field.strip()
        if field:
            field_candidates.append(field)
    for field in ["cutoff_year", "截止年份", "year"]:
        if field not in field_candidates:
            field_candidates.append(field)
    for field in field_candidates:
        value = doc.get(field)
        if value is None or str(value).strip() == "":
            continue
        try:
            return int(str(value).strip())
        except Exception:
            continue
    return None


def load_historical_prompt(year: int, prompt_dir: str, max_chars: int = 0) -> str:
    if not year or not prompt_dir:
        return ""
    path = Path(prompt_dir) / f"prompt_{int(year)}.txt"
    if not path.exists():
        return ""
    text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8")).strip()
    if max_chars and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def parse_historical_distance_pairs(text: str) -> List[Tuple[str, str, float]]:
    pairs = []
    for a, b, value in re.findall(r"\(([^；;()]+)[；;]([^；;()]+)[；;]([0-9.]+)\)", text or ""):
        try:
            pairs.append((a.strip(), b.strip(), float(value)))
        except Exception:
            continue
    return pairs


def normalize_subject_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def extract_level0_roots_from_mag_payload(mag_payload: Optional[dict]) -> List[str]:
    if not isinstance(mag_payload, dict):
        return []
    roots = set()
    for path in mag_payload.get("paths", []) or []:
        names = path.get("path_names", []) or []
        levels = path.get("path_levels", []) or []
        for name, level in zip(names, levels):
            try:
                is_root = int(level) == 0
            except Exception:
                is_root = False
            if is_root and str(name or "").strip():
                roots.add(str(name).strip())
    return sorted(roots)


def root_to_historical_subjects(root: str) -> List[str]:
    subjects = FIELD_DIC.get(root)
    if subjects:
        return list(subjects)
    # MAG root names sometimes differ only by case from the historical prompt fields.
    if str(root).strip().lower() == "computer science":
        return ["Computer Science"]
    return [str(root).strip()]


def build_compact_historical_context(
    year: int,
    prompt_dir: str,
    roots: List[str],
    max_chars: int = 0,
    max_pairs: int = 80,
) -> str:
    if not year or not prompt_dir or not roots:
        return ""
    raw = load_historical_prompt(year, prompt_dir, max_chars=0)
    pairs = parse_historical_distance_pairs(raw)
    if not pairs:
        return ""

    subject_to_roots = defaultdict(set)
    for root in roots:
        for subject in root_to_historical_subjects(root):
            if subject:
                subject_to_roots[normalize_subject_name(subject)].add(root)
    involved_subjects = set(subject_to_roots)
    if not involved_subjects:
        return ""

    selected = []
    for a, b, distance in pairs:
        a_norm = normalize_subject_name(a)
        b_norm = normalize_subject_name(b)
        a_hit = a_norm in involved_subjects
        b_hit = b_norm in involved_subjects
        if not (a_hit or b_hit):
            continue
        selected.append({
            "a": a,
            "b": b,
            "distance": float(distance),
            "hit_count": int(a_hit) + int(b_hit),
            "root_labels": sorted(subject_to_roots.get(a_norm, set()) | subject_to_roots.get(b_norm, set())),
        })

    if not selected:
        return ""
    selected.sort(key=lambda item: (-item["hit_count"], item["distance"], item["a"], item["b"]))
    if max_pairs and max_pairs > 0:
        selected = selected[:max_pairs]

    lines = [
        f"{year} compact historical discipline-distance context.",
        "Distances are Euclidean distances from the year-specific Node2Vec discipline space; smaller means closer.",
        "Only pairs touching the paper's MAG-matched level-0 disciplines are included.",
        "Matched MAG level-0 disciplines: " + ", ".join(roots) + ".",
    ]
    for item in selected:
        labels = ",".join(item["root_labels"]) if item["root_labels"] else "matched"
        lines.append(f"({item['a']}; {item['b']}; {item['distance']:.6f}; roots={labels})")
    text = " ".join(lines)
    if max_chars and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def build_attention_prompt(
    doc: dict,
    keywords: List[str],
    lower_keywords: bool = True,
    prompt_variant: str = "baseline",
    cutoff_year: Optional[int] = None,
    historical_context: str = "",
) -> str:
    kws = [str(k).strip() for k in keywords if str(k).strip()]
    if lower_keywords:
        kws = [k.lower() for k in kws]
    prompt = ""
    if prompt_variant == "historical_distance" and historical_context:
        prompt += (
            "Historical discipline-distance context for the paper cutoff year: "
            f'"{clean_for_prompt(historical_context)}". '
        )
    prompt += (
        f'Given a paper with title "{clean_for_prompt(doc.get("title", ""))}", '
        f'abstract "{clean_for_prompt(doc.get("abs", ""))}", '
    )
    if prompt_variant in {"cutoff_year", *HISTORICAL_VARIANTS} and cutoff_year is not None:
        prompt += (
            f"assess the paper using only knowledge available up to the end of {int(cutoff_year)}, "
        )
    if prompt_variant == "historical_distance_compact" and historical_context:
        prompt += (
            "use this compact historical discipline-distance context for the cutoff year: "
            f'"{clean_for_prompt(historical_context)}", '
        )
    prompt += f'key words: {", ".join(kws)}'
    return prompt


def build_attention_prompt_for_doc(doc: dict, keywords: List[str], args, mag_payload: Optional[dict] = None) -> Tuple[str, dict]:
    cutoff_year = get_cutoff_year(doc, args.cutoff_year_field)
    historical_context = ""
    historical_roots = []
    if args.prompt_variant in HISTORICAL_VARIANTS and cutoff_year is not None:
        historical_roots = extract_level0_roots_from_mag_payload(mag_payload)
    if args.prompt_variant == "historical_distance" and cutoff_year is not None:
        historical_context = load_historical_prompt(
            cutoff_year,
            args.historical_prompt_dir,
            max_chars=args.historical_prompt_max_chars,
        )
    elif args.prompt_variant == "historical_distance_compact" and cutoff_year is not None:
        historical_context = build_compact_historical_context(
            cutoff_year,
            args.historical_prompt_dir,
            historical_roots,
            max_chars=args.historical_compact_max_chars,
            max_pairs=args.historical_compact_max_pairs,
        )
    prompt = build_attention_prompt(
        doc,
        keywords,
        lower_keywords=True,
        prompt_variant=args.prompt_variant,
        cutoff_year=cutoff_year,
        historical_context=historical_context,
    )
    return prompt, {
        "prompt_variant": args.prompt_variant,
        "cutoff_year": cutoff_year,
        "historical_prompt_loaded": bool(historical_context),
        "historical_prompt_chars": len(historical_context),
        "historical_prompt_roots": historical_roots,
    }


def find_keywords_block(prompt: str):
    match = re.search(r"key\s*words\s*:\s*", prompt, flags=re.IGNORECASE)
    if not match:
        raise ValueError("prompt does not contain 'key words:'")
    return match.end(), prompt[match.end():]


def split_keywords_preserve(kw_text: str):
    items, buf, depth = [], [], 0
    for ch in kw_text:
        if ch == "," and depth == 0:
            item = "".join(buf).strip()
            if item:
                items.append(item)
            buf = []
            continue
        buf.append(ch)
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
    last = "".join(buf).strip()
    if last:
        items.append(last)
    return items


def encode_with_offsets(tokenizer, prompt: str, device, max_len: int):
    old_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
        )
    finally:
        tokenizer.truncation_side = old_truncation_side
    input_ids = enc["input_ids"].to(device)
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"][0].tolist()]
    return input_ids, offsets


def char_span_to_token_span(offsets, char_st, char_ed):
    tok_st, tok_ed = None, None
    for j, (s, e) in enumerate(offsets):
        if s == -1 and e == -1:
            continue
        if e <= char_st:
            continue
        if s >= char_ed:
            break
        if tok_st is None:
            tok_st = j
        tok_ed = j + 1
    return tok_st, tok_ed


def build_sink_mask(tokenizer, ids, sink_first_k: int, mask_special: bool):
    mask = np.zeros((len(ids),), dtype=bool)
    if mask_special:
        special = set(tokenizer.all_special_ids or [])
        for j, tid in enumerate(ids):
            if tid in special:
                mask[j] = True
    if sink_first_k > 0:
        mask[:min(sink_first_k, len(ids))] = True
    return mask


def token_ids_of(tokenizer, s: str):
    return tokenizer(s, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()


def subseq_search(hay, needle):
    if len(needle) == 0 or len(hay) < len(needle):
        return None
    first = needle[0]
    for i, x in enumerate(hay):
        if x == first and hay[i:i + len(needle)] == needle:
            return i, i + len(needle)
    return None


def fallback_locate_by_token_variants(tokenizer, ids, start_pos, kw_text: str):
    compressed = re.sub(r"\s+", " ", kw_text.strip())
    variants = [kw_text, " " + kw_text, compressed, " " + compressed]
    tail = ids[start_pos:]
    for variant in variants:
        cand = token_ids_of(tokenizer, variant)
        hit = subseq_search(tail, cand)
        if hit:
            return start_pos + hit[0], start_pos + hit[1]
    return None


def capture_layer_attention(model, input_ids, layer_idx: int):
    """Capture one layer's self-attention without asking every layer to return it."""
    captured = {}
    attn_mod = model.model.layers[layer_idx].self_attn
    original_forward = attn_mod.forward

    def wrapped_forward(*forward_args, **forward_kwargs):
        forward_kwargs["output_attentions"] = True
        output = original_forward(*forward_args, **forward_kwargs)
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            captured["attention"] = output[1].detach().to("cpu")
        return output

    attn_mod.forward = wrapped_forward
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, output_attentions=False, use_cache=False)
    finally:
        attn_mod.forward = original_forward
    if "attention" not in captured:
        raise RuntimeError(f"attention layer {layer_idx} was not captured")
    return captured["attention"]


def choose_attention_matrix(layer_attention, head: int = -1) -> np.ndarray:
    att = layer_attention[0].float()
    if head is not None and int(head) >= 0:
        h = min(int(head), int(att.shape[0]) - 1)
        mat = att[h].numpy()
    else:
        mat = att.mean(dim=0).numpy()
    return np.asarray(mat, dtype=np.float32)


def downsample_square(mat: np.ndarray, max_tokens: int) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    n = int(mat.shape[0])
    if max_tokens <= 0 or n <= max_tokens:
        return mat
    edges = np.linspace(0, n, int(max_tokens) + 1).astype(int)
    out = np.zeros((len(edges) - 1, len(edges) - 1), dtype=np.float32)
    for i in range(len(edges) - 1):
        r0, r1 = edges[i], max(edges[i + 1], edges[i] + 1)
        for j in range(len(edges) - 1):
            c0, c1 = edges[j], max(edges[j + 1], edges[j] + 1)
            block = mat[r0:r1, c0:c1]
            out[i, j] = float(block.mean()) if block.size else 0.0
    return out


def attention_heat_color(value: float):
    value = max(0.0, min(1.0, float(value)))
    stops = [
        (255, 255, 255),
        (221, 236, 246),
        (116, 169, 207),
        (43, 140, 190),
        (252, 141, 89),
        (215, 48, 39),
    ]
    scaled = value * (len(stops) - 1)
    idx = int(math.floor(scaled))
    if idx >= len(stops) - 1:
        return stops[-1]
    frac = scaled - idx
    a, b = stops[idx], stops[idx + 1]
    return tuple(int(round(a[k] * (1.0 - frac) + b[k] * frac)) for k in range(3))


def save_attention_png(path: Path, att: np.ndarray, max_tokens: int = 512):
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError(f"PIL is required to save attention PNG: {exc!r}") from exc
    mat = downsample_square(att, int(max_tokens))
    n = int(mat.shape[0])
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = mat[~upper]
    vmax = float(np.percentile(vals[vals > 0], 99.5)) if np.any(vals > 0) else 1.0
    vmax = max(vmax, 1e-12)
    img = Image.new("RGB", (n, n), (248, 248, 248))
    pix = img.load()
    for r in range(n):
        for c in range(r + 1):
            pix[c, r] = attention_heat_color(float(mat[r, c]) / vmax)
    scale = max(1, min(4, 900 // max(n, 1)))
    if scale > 1:
        img = img.resize((n * scale, n * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width - 1, img.height - 1), outline=(60, 60, 60))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def safe_doc_slug(doc_id: str, title: str = ""):
    basis = str(doc_id or title or "doc")
    digest = hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()[:10]
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(title or doc_id or "doc")).strip("_")[:72]
    return f"{clean or 'doc'}_{digest}"


def save_attention_map_artifacts(doc: dict, prompt: str, tokens: List[str], att: np.ndarray, prompt_meta: dict, args, out_dir: Path):
    base_dir = Path(args.attention_map_dir) if args.attention_map_dir else (Path(out_dir) / "attention_maps")
    variant = str(args.prompt_variant or "baseline")
    slug = safe_doc_slug(doc.get("doc_id"), doc.get("title"))
    doc_dir = base_dir / variant
    stem = f"{slug}_layer{int(args.layer)}"
    if int(getattr(args, "attention_map_head", -1)) >= 0:
        stem += f"_head{int(args.attention_map_head)}"
    meta_path = doc_dir / f"{stem}.json"
    png_path = doc_dir / f"{stem}.png"
    raw_path = doc_dir / f"{stem}.npz"
    save_attention_png(png_path, att, int(args.attention_map_png_max_tokens))
    if bool(getattr(args, "attention_map_save_raw", False)):
        doc_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_path, attention=att.astype(np.float16))
    payload = {
        "doc_id": doc.get("doc_id"),
        "title": doc.get("title"),
        "year": doc.get("year"),
        "prompt_variant": args.prompt_variant,
        "layer": int(args.layer),
        "head": int(getattr(args, "attention_map_head", -1)),
        "head_mode": "mean" if int(getattr(args, "attention_map_head", -1)) < 0 else "single_head",
        "token_count": int(len(tokens)),
        "tokens": tokens,
        "prompt_char_count": len(prompt),
        "prompt_meta": prompt_meta,
        "png_path": str(png_path),
        "raw_npz_path": str(raw_path) if bool(getattr(args, "attention_map_save_raw", False)) else "",
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"metadata_path": str(meta_path), "png_path": str(png_path), "raw_npz_path": payload["raw_npz_path"]}


def compute_attention_masses(prompt: str, model, tokenizer, args, return_map: bool = False):
    kw_char_start, kw_text = find_keywords_block(prompt)
    keywords = split_keywords_preserve(kw_text)
    if not keywords:
        return ({}, None) if return_map else {}
    device = next(model.parameters()).device
    input_ids, offsets = encode_with_offsets(tokenizer, prompt, device, max_len=args.max_len)
    ids = input_ids[0].tolist()
    token_count = len(ids)

    kw_start_tok = None
    for j, (s, e) in enumerate(offsets):
        if s == -1 and e == -1:
            continue
        if e > kw_char_start:
            kw_start_tok = j
            break
    if kw_start_tok is None:
        kw_start_tok = len(tokenizer(prompt[:kw_char_start], return_tensors="pt")["input_ids"][0])

    sink_mask = build_sink_mask(tokenizer, ids, args.sink_first_k, True)
    prefix_cols = np.where((np.arange(token_count) < kw_start_tok) & (~sink_mask))[0]
    total_layers = len(model.model.layers)
    layer_idx = (args.layer + total_layers) % total_layers
    captured = capture_layer_attention(model, input_ids, layer_idx)
    att = choose_attention_matrix(captured, int(getattr(args, "attention_map_head", -1)))
    del captured
    torch.cuda.empty_cache()
    att[:, sink_mask] = 0.0
    att_norm = att / (att.sum(axis=1, keepdims=True) + 1e-12)

    masses = {}
    max_offset_end = max((e for _s, e in offsets), default=0)
    for kw in keywords:
        pos = prompt[kw_char_start:].find(kw) if kw_char_start <= max_offset_end else -1
        if pos >= 0:
            char_st = kw_char_start + pos
            char_ed = char_st + len(kw)
            tok_st, tok_ed = char_span_to_token_span(offsets, char_st, char_ed)
        else:
            tok_st, tok_ed = fallback_locate_by_token_variants(tokenizer, ids, kw_start_tok, kw) or (None, None)
        if tok_st is None or tok_ed is None or tok_st >= tok_ed or len(prefix_cols) == 0:
            mass = 0.0
        else:
            rows = list(range(tok_st, tok_ed))
            mass = float(att_norm[rows, :][:, prefix_cols].sum(axis=1).mean())
        masses[norm_kw(kw)] = mass
    if return_map:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        return masses, {
            "attention": att_norm.astype(np.float32, copy=False),
            "tokens": tokens,
            "layer_idx": layer_idx,
            "token_count": int(token_count),
        }
    return masses


def aggregate_attention_masses(attn_masses: Dict[str, float], full_norm: bool) -> Dict[str, float]:
    if not attn_masses:
        return {}
    agg = defaultdict(float)
    normalizer = norm_kw_full if full_norm else norm_kw
    for key, value in attn_masses.items():
        normed = normalizer(key)
        if not normed:
            continue
        try:
            agg[normed] += float(value)
        except Exception:
            continue
    return dict(agg)


def load_subject_distances(dist_file: str) -> Dict[Tuple[str, str], float]:
    df = pd.read_csv(dist_file)
    cols = df.columns.tolist()
    col_a = next((c for c in cols if "mag_a" in c), "mag_a_pretty")
    col_b = next((c for c in cols if "mag_b" in c), "mag_b_pretty")
    col_d = next((c for c in cols if "distance" in c), "distance_norm_mean")
    dist = {}
    for _, row in df.iterrows():
        a = str(row[col_a]).strip()
        b = str(row[col_b]).strip()
        val = float(row[col_d])
        dist[(a, b)] = val
        dist[(b, a)] = val
    return dist


def query_parquet_for_keywords(parquet_path: str, year: int, keywords: List[str]) -> Dict[str, dict]:
    if not parquet_path or not os.path.exists(parquet_path) or not keywords:
        return {}
    kw_lower = sorted({norm_kw(k) for k in keywords if norm_kw(k)})
    if not kw_lower:
        return {}
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={max(1, os.cpu_count() or 1)};")
    kw_df = pd.DataFrame({"kw": kw_lower})
    con.register("kw", kw_df)
    query = f"""
    SELECT d.name, d.year, d.rel_final_root, kw.kw AS matched_keyword
    FROM read_parquet('{parquet_path}') AS d
    JOIN kw ON lower(trim(coalesce(d.name, ''))) = kw.kw
    WHERE d.year = {int(year)}
    """
    out = con.execute(query).df()
    con.close()
    if out.empty:
        return {}
    out["__name_norm__"] = out["name"].apply(norm_kw)
    out = out.drop_duplicates("__name_norm__", keep="first")
    return {row["__name_norm__"]: row.to_dict() for _, row in out.iterrows()}


def es_count_for_keyword_root(es_client, cache: dict, keyword: str, acemap_field: str,
                              publication_year: Optional[str], index: str = "acemap.works") -> int:
    key = (keyword, acemap_field, str(publication_year or ""), index)
    if key in cache:
        return cache[key]
    bool_query = {
        "must": [
            {"term": {"topics.field.display_name.keyword": acemap_field}},
            {
                "bool": {
                    "should": [
                        {"match_phrase": {"title": keyword}},
                        {"match_phrase": {"abstract": keyword}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        ],
        "filter": [],
    }
    if publication_year:
        bool_query["filter"].append({
            "range": {"publication_date": {"lte": f"{publication_year}-12-31"}}
        })
    try:
        resp = es_client.count(index=index, body={"query": {"bool": bool_query}})
        val = int(resp.get("count", 0))
    except Exception as exc:
        print(f"[ES ERROR] keyword={keyword} field={acemap_field} year={publication_year} err={exc}")
        val = 0
    cache[key] = val
    return val


def keyword_quantile(es_client, cache: dict, count_cache: dict, keyword: str, acemap_field: str, pub_year: int):
    key = (keyword, acemap_field, int(pub_year))
    if key in cache:
        return cache[key]
    total = es_count_for_keyword_root(es_client, count_cache, keyword, acemap_field, "2025")
    if total == 0:
        cache[key] = 0.0
        return 0.0
    count_pub = es_count_for_keyword_root(es_client, count_cache, keyword, acemap_field, str(int(pub_year) - 1))
    val = max(0.0, min(1.0, count_pub / total))
    cache[key] = val
    return val


def compute_vs(
    df: pd.DataFrame,
    dist_dict: Dict[Tuple[str, str], float],
    threshold: float,
    default_distance: float = 0.0,
):
    if df.empty or "attn_mass" not in df.columns:
        return {}, 0.0
    total = df["attn_mass"].sum()
    if total <= 0:
        return {}, 0.0
    tmp = df.copy()
    tmp["a_hat"] = tmp["attn_mass"] / float(total)
    weights = tmp.groupby("final_root", dropna=True)["a_hat"].sum()
    if weights.empty:
        return {}, 0.0
    anchor = weights.idxmax()
    adjusted = weights.copy()
    for subject, weight in weights.items():
        dis = dist_dict.get((anchor, subject), default_distance)
        if dis > threshold:
            adjusted[subject] = weight * (1.0 + dis - threshold)
        else:
            adjusted[subject] = weight * (1.0 - dis + threshold)
    total_adj = adjusted.sum()
    if total_adj <= 0:
        return {}, 0.0
    adjusted = adjusted / total_adj
    if len(adjusted) == 1:
        return adjusted.to_dict(), 0.0
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in adjusted)
    return adjusted.to_dict(), entropy / math.log(len(adjusted))


def root_distance_v5(dist_dict: Dict[Tuple[str, str], float], a: str, b: str) -> float:
    if a == b:
        return 0.0
    return float(dist_dict.get((a, b), 1.0))


def gate_roots_revised_v5(vs_input: pd.DataFrame, dist_dict: Dict[Tuple[str, str], float], args):
    strength_series = (
        vs_input.assign(root_strength_component=vs_input["a_hat"] * vs_input["lexical_fidelity"])
        .groupby("final_root", dropna=True)["root_strength_component"]
        .sum()
        .sort_values(ascending=False)
    )
    root_strength_by_root = {
        str(root): float(value)
        for root, value in strength_series.to_dict().items()
    }
    total_strength = float(strength_series.sum())
    if total_strength > 0:
        root_share_by_root = {
            str(root): float(value / total_strength)
            for root, value in strength_series.to_dict().items()
        }
    else:
        root_share_by_root = {str(root): 0.0 for root in strength_series.index.tolist()}

    strong_series = (
        vs_input[vs_input["is_strong_anchor"]]
        .groupby("final_root", dropna=True)
        .size()
    )
    strong_anchor_count_by_root = {
        str(root): int(value)
        for root, value in strong_series.to_dict().items()
    }
    root_counts = {
        str(root): int(value)
        for root, value in vs_input["final_root"].value_counts().to_dict().items()
    }
    roots = sorted(
        root_strength_by_root,
        key=lambda root: (
            root_share_by_root.get(root, 0.0),
            strong_anchor_count_by_root.get(root, 0),
            root_counts.get(root, 0),
            root,
        ),
        reverse=True,
    )
    if not roots:
        return {
            "root_strength_by_root": root_strength_by_root,
            "root_share_by_root": root_share_by_root,
            "strong_anchor_count_by_root": strong_anchor_count_by_root,
            "keep_roots_before_singleton": set(),
            "dropped_roots_by_distribution_gate": [],
            "root_gate_mode": "revised_v5_no_roots",
        }

    p1 = float(root_share_by_root.get(roots[0], 0.0))
    keep_roots = [roots[0]]
    for root in roots[1:]:
        strong_count = int(strong_anchor_count_by_root.get(root, 0))
        share = float(root_share_by_root.get(root, 0.0))
        ratio_to_top = float(share / p1) if p1 > 0 else 0.0
        if strong_count <= 0:
            continue
        if strong_count >= 2:
            has_evidence = (
                share >= float(args.strong2_min_share)
                and ratio_to_top >= float(args.strong2_min_ratio)
            )
        else:
            has_evidence = (
                share >= float(args.strong1_min_share)
                and ratio_to_top >= float(args.strong1_min_ratio)
            )
        if not has_evidence:
            continue
        min_distance = min(root_distance_v5(dist_dict, root, kept_root) for kept_root in keep_roots)
        if min_distance >= float(args.far_distance):
            keep_roots.append(root)
        elif (
            min_distance >= float(args.close_distance)
            and strong_count >= 2
            and share >= float(args.mid_min_share)
            and ratio_to_top >= float(args.mid_min_ratio)
        ):
            keep_roots.append(root)

    keep_set = set(str(root) for root in keep_roots)
    all_roots = set(str(root) for root in roots)
    return {
        "root_strength_by_root": root_strength_by_root,
        "root_share_by_root": root_share_by_root,
        "strong_anchor_count_by_root": strong_anchor_count_by_root,
        "keep_roots_before_singleton": keep_set,
        "dropped_roots_by_distribution_gate": sorted(all_roots - keep_set),
        "root_gate_mode": "revised_v5_distance_evidence",
    }


def root_rank_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["root", "attn_sum", "strong_count", "anchored_count", "max_fidelity", "row_count"])
    tmp = df.copy()
    tmp["final_root"] = tmp["final_root"].fillna("").astype(str).str.strip()
    tmp = tmp[tmp["final_root"] != ""].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["root", "attn_sum", "strong_count", "anchored_count", "max_fidelity", "row_count"])
    tmp["attn_mass"] = pd.to_numeric(tmp.get("attn_mass", 0.0), errors="coerce").fillna(0.0)
    tmp["lexical_fidelity"] = pd.to_numeric(tmp.get("lexical_fidelity", 0.0), errors="coerce").fillna(0.0)
    tmp["is_strong_anchor"] = tmp.get("is_strong_anchor", False)
    tmp["is_strong_anchor"] = tmp["is_strong_anchor"].fillna(False).astype(bool)
    tmp["is_anchored"] = tmp.get("is_anchored", False)
    tmp["is_anchored"] = tmp["is_anchored"].fillna(False).astype(bool)
    rows = []
    for root, group in tmp.groupby("final_root", dropna=True):
        rows.append({
            "root": str(root),
            "attn_sum": float(group["attn_mass"].sum()),
            "strong_count": int(group["is_strong_anchor"].sum()),
            "anchored_count": int(group["is_anchored"].sum()),
            "max_fidelity": float(group["lexical_fidelity"].max()),
            "row_count": int(len(group)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["attn_sum", "strong_count", "anchored_count", "max_fidelity", "row_count", "root"],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)


def rescue_cross_preserving_roots(
    detail_df: pd.DataFrame,
    keep_roots: set,
    initial_roots: set,
    max_roots: int,
    min_roots: int = 2,
) -> Tuple[set, List[str]]:
    """Preserve cross-root evidence found by initial MAG grounding.

    Quality/root gates remove noisy matches and cap over-broad MAG grounding,
    but should not collapse an initially multi-root paper into a single-root
    paper. If initial grounding has at least two roots, rescue the best extra
    root(s) by attention and anchor evidence. If initial grounding itself has
    fewer than two roots, the paper may remain non-cross.
    """
    if not initial_roots or len(initial_roots) < min_roots:
        return set(keep_roots), []
    effective_max = int(max_roots or 0)
    target = min(min_roots, effective_max) if effective_max > 0 else min_roots
    keep = {str(root) for root in keep_roots if str(root) in initial_roots}
    if len(keep) >= target:
        return keep, []
    candidates = root_rank_table(detail_df)
    rescued = []
    for _, row in candidates.iterrows():
        root = str(row["root"])
        if root not in initial_roots or root in keep:
            continue
        keep.add(root)
        rescued.append(root)
        if len(keep) >= target:
            break
    return keep, rescued


def compute_xindex(doc: dict, mag_payload: dict, attn_masses: Dict[str, float],
                   dist_dict, parquet_path: str, es_client, es_count_cache, quantile_cache, args):
    kwid_to_rootids = defaultdict(set)
    id2name, id2level = {}, {}
    for node in mag_payload.get("nodes", []):
        fid = int(node["field_id"])
        id2name[fid] = node.get("name", "")
        id2level[fid] = node.get("level")
    for path in mag_payload.get("paths", []):
        ids = [int(x) for x in path.get("path_ids", [])]
        if not ids:
            continue
        kwid_to_rootids[ids[0]].add(ids[-1])
        for fid, name, level in zip(ids, path.get("path_names", []), path.get("path_levels", [])):
            id2name.setdefault(int(fid), name)
            id2level.setdefault(int(fid), level)

    match_meta_by_fid = {}
    for match in mag_payload.get("matches", []):
        fid = int(match.get("mag_field_id", -1))
        if fid < 0:
            continue
        meta = match_meta_by_fid.setdefault(
            fid,
            {
                "source_terms": [],
                "lexical_fidelity": 0.0,
                "lexical_precision": 0.0,
                "lexical_recall": 0.0,
                "shared_token_count": 0,
                "source_token_count": 0,
                "candidate_token_count": 0,
                "is_strong_anchor": False,
                "is_anchored": False,
                "match_selection_mode": "",
                "hash_topdown_root_id": None,
                "hash_topdown_root_name": "",
                "llm_root_id": None,
                "llm_root_name": "",
            },
        )
        term = str(match.get("term") or "").strip()
        if term:
            meta["source_terms"].append(term)
        fidelity = float(match.get("lexical_fidelity") or 0.0)
        if fidelity > meta["lexical_fidelity"]:
            meta["lexical_fidelity"] = fidelity
            meta["lexical_precision"] = float(match.get("lexical_precision") or 0.0)
            meta["lexical_recall"] = float(match.get("lexical_recall") or 0.0)
            meta["shared_token_count"] = int(match.get("shared_token_count") or 0)
            meta["source_token_count"] = int(match.get("source_token_count") or 0)
            meta["candidate_token_count"] = int(match.get("candidate_token_count") or 0)
            meta["match_selection_mode"] = str(match.get("match_selection_mode") or "")
        if match.get("is_strong_anchor"):
            meta["is_strong_anchor"] = True
        if match.get("is_anchored"):
            meta["is_anchored"] = True
        if match.get("hash_topdown_root_id") is not None:
            meta["hash_topdown_root_id"] = int(match.get("hash_topdown_root_id"))
        if match.get("hash_topdown_root_name"):
            meta["hash_topdown_root_name"] = str(match.get("hash_topdown_root_name") or "").strip()
        if match.get("llm_root_id") is not None:
            meta["llm_root_id"] = int(match.get("llm_root_id"))
        if match.get("llm_root_name"):
            meta["llm_root_name"] = str(match.get("llm_root_name") or "").strip()

    kw_names = [(id2name.get(fid) or "").strip() for fid in kwid_to_rootids]
    parquet_lookup = query_parquet_for_keywords(parquet_path, int(doc["year"]), kw_names)
    use_full_attn_merge = args.algo_profile in {"revised_v2", "revised_v3", "revised_v4", "revised_v5"}
    attn_lookup = aggregate_attention_masses(attn_masses, full_norm=use_full_attn_merge)
    keyword_norm = norm_kw_full if use_full_attn_merge else norm_kw
    base_rows = []

    for kw_id, root_ids in kwid_to_rootids.items():
        kw_name = (id2name.get(kw_id) or "").strip()
        if not kw_name:
            continue
        level0_roots = []
        for rid in sorted(root_ids):
            if id2level.get(rid) == 0:
                rname = (id2name.get(rid) or "").strip()
                if rname:
                    level0_roots.append(rname)

        provisional_root = None
        best_count = -1
        best_af = None
        meta = match_meta_by_fid.get(int(kw_id), {})
        topdown_root = str(meta.get("hash_topdown_root_name") or meta.get("llm_root_name") or "").strip()
        if topdown_root:
            provisional_root = topdown_root
        else:
            hit = parquet_lookup.get(norm_kw(kw_name))
            if hit is not None:
                provisional_root = hit.get("rel_final_root") or None
            else:
                for root in level0_roots:
                    counts = [
                        es_count_for_keyword_root(es_client, es_count_cache, kw_name, af, str(doc["year"]))
                        for af in FIELD_DIC.get(root, [])
                    ]
                    max_count = max(counts) if counts else 0
                    if max_count > best_count:
                        best_count = max_count
                        provisional_root = root
                if provisional_root:
                    best_af_count = -1
                    for af in FIELD_DIC.get(provisional_root, []):
                        count = es_count_for_keyword_root(es_client, es_count_cache, kw_name, af, str(doc["year"]))
                        if count > best_af_count:
                            best_af_count = count
                            best_af = af

        if topdown_root and provisional_root:
            for root in level0_roots:
                if root == provisional_root:
                    counts = [
                        es_count_for_keyword_root(es_client, es_count_cache, kw_name, af, str(doc["year"]))
                        for af in FIELD_DIC.get(root, [])
                    ]
                    best_count = max(counts) if counts else 0
                    break

        query_af = best_af
        if not query_af and provisional_root and FIELD_DIC.get(provisional_root):
            query_af = FIELD_DIC[provisional_root][0]
        q = keyword_quantile(
            es_client,
            quantile_cache,
            es_count_cache,
            kw_name,
            query_af,
            int(doc["year"]),
        ) if query_af else 0.0
        is_strong_anchor = bool(meta.get("is_strong_anchor") or False)
        if args.algo_profile in {"revised_v4", "revised_v5"}:
            is_anchored = is_strong_anchor
        else:
            is_anchored = bool(
                meta.get("is_anchored")
                or float(meta.get("lexical_fidelity") or 0.0) >= args.lexical_fidelity_threshold
            )

        base_rows.append({
            "keyword_name": kw_name,
            "attn_mass": float(attn_lookup.get(keyword_norm(kw_name), 0.0)),
            "keyword_level": id2level.get(kw_id),
            "roots_level0_dict": json.dumps({kw_name: level0_roots}, ensure_ascii=False),
            "final_root": provisional_root,
            "provisional_root": provisional_root,
            "final_root_attn_mass": float(attn_lookup.get(keyword_norm(provisional_root), 0.0)) if provisional_root else np.nan,
            "final_count": best_count,
            "keyword_quantile": q,
            "published_year": int(doc["year"]),
            "candidate_roots": level0_roots,
            "candidate_root_count": len(level0_roots),
            "source_terms": list(dict.fromkeys(meta.get("source_terms", []))),
            "lexical_fidelity": float(meta.get("lexical_fidelity") or 0.0),
            "lexical_precision": float(meta.get("lexical_precision") or 0.0),
            "lexical_recall": float(meta.get("lexical_recall") or 0.0),
            "shared_token_count": int(meta.get("shared_token_count") or 0),
            "source_token_count": int(meta.get("source_token_count") or 0),
            "candidate_token_count": int(meta.get("candidate_token_count") or 0),
            "is_strong_anchor": is_strong_anchor,
            "is_anchored": is_anchored,
            "match_selection_mode": str(meta.get("match_selection_mode") or ""),
            "llm_root_name": str(meta.get("llm_root_name") or ""),
            "llm_root_id": meta.get("llm_root_id"),
        })

    detail_df = pd.DataFrame(base_rows)
    vt = 0.0
    vs_rows_before_filter = 0
    vs_rows_after_filter = 0
    singleton_root_count = 0
    distinct_final_root_before_filter = 0
    distinct_final_root_after_filter = 0
    dominant_roots = []
    anchored_root_counts = {}
    anchored_keyword_count = 0
    weak_keyword_count = 0
    root_gate_applied = False
    dropped_keyword_count = 0
    kept_keyword_count_after_gate = 0
    root_strength_by_root = {}
    root_share_by_root = {}
    dropped_roots_by_strength = []
    strong_anchor_count_by_root = {}
    root_gate_mode = "none"
    kept_roots_after_distribution_gate = []
    dropped_roots_by_distribution_gate = []
    dropped_roots_by_singleton_gate = []
    dropped_keywords_by_quality_gate = []
    dropped_roots_by_root_cap = []
    quality_gate_kept_keyword_count = 0
    quality_gate_dropped_keyword_count = 0
    root_cap_applied = False
    root_cap_max_roots = int(getattr(args, "xindex_max_roots_per_paper", 0) or 0)
    preserve_initial_cross_roots = bool(getattr(args, "xindex_preserve_initial_cross_roots", True))
    initial_candidate_roots = set()
    initial_provisional_roots = set()
    cross_preserve_rescued_roots = []
    cross_preserve_applied = False
    vs_rows_before_root_gate = 0
    vs_rows_after_distribution_gate = 0
    vs_rows_after_singleton_gate = 0

    if not detail_df.empty:
        detail_df["attn_mass"] = pd.to_numeric(detail_df["attn_mass"], errors="coerce").fillna(0.0)
        detail_df["lexical_fidelity"] = pd.to_numeric(detail_df["lexical_fidelity"], errors="coerce").fillna(0.0)
        detail_df["is_anchored"] = detail_df["is_anchored"].fillna(False).astype(bool)
        detail_df["is_strong_anchor"] = detail_df.get("is_strong_anchor", False)
        detail_df["is_strong_anchor"] = detail_df["is_strong_anchor"].fillna(False).astype(bool)
        if "dropped_by_root_gate" not in detail_df.columns:
            detail_df["dropped_by_root_gate"] = False
        else:
            detail_df["dropped_by_root_gate"] = detail_df["dropped_by_root_gate"].fillna(False).astype(bool)
        detail_df["dropped_by_quality_gate"] = False
        detail_df["dropped_by_root_cap"] = False
        min_fidelity = float(getattr(args, "xindex_min_lexical_fidelity", args.lexical_fidelity_threshold))
        allow_embedding_fallback = bool(getattr(args, "xindex_allow_embedding_fallback", False))
        final_root_series = detail_df["final_root"].fillna("").astype(str).str.strip()
        initial_provisional_roots = set(final_root_series[final_root_series.ne("")].astype(str).tolist())
        initial_candidate_roots = set()
        for roots in detail_df.get("candidate_roots", []):
            if isinstance(roots, list):
                initial_candidate_roots.update(str(root).strip() for root in roots if str(root).strip())
        initial_cross_roots = initial_provisional_roots if len(initial_provisional_roots) >= 2 else initial_candidate_roots
        mode_series = detail_df.get("match_selection_mode", "")
        mode_series = mode_series.fillna("").astype(str) if hasattr(mode_series, "fillna") else pd.Series([""] * len(detail_df), index=detail_df.index)
        quality_keep = (
            final_root_series.ne("")
            & (
                detail_df["is_strong_anchor"]
                | detail_df["is_anchored"]
                | (detail_df["lexical_fidelity"] >= min_fidelity)
            )
        )
        if not allow_embedding_fallback:
            quality_keep = quality_keep & ~mode_series.str.contains("embedding_fallback", na=False)
        detail_df.loc[~quality_keep, "dropped_by_quality_gate"] = True
        if root_cap_max_roots > 0 and quality_keep.any():
            root_weight = (
                detail_df.loc[quality_keep]
                .assign(__root=final_root_series.loc[quality_keep])
                .groupby("__root", dropna=True)["attn_mass"]
                .sum()
                .sort_values(ascending=False)
            )
            keep_roots_by_cap = set(root_weight.head(root_cap_max_roots).index.astype(str).tolist())
            root_cap_applied = len(root_weight) > root_cap_max_roots
            if root_cap_applied:
                cap_drop = quality_keep & ~final_root_series.isin(keep_roots_by_cap)
                detail_df.loc[cap_drop, "dropped_by_root_cap"] = True
                dropped_roots_by_root_cap = sorted(str(root) for root in set(root_weight.index.astype(str)) - keep_roots_by_cap)
        quality_or_cap_drop = detail_df["dropped_by_quality_gate"] | detail_df["dropped_by_root_cap"]
        if quality_or_cap_drop.any():
            dropped_keywords_by_quality_gate = detail_df.loc[quality_or_cap_drop, "keyword_name"].fillna("").astype(str).tolist()
            detail_df.loc[quality_or_cap_drop, "dropped_by_root_gate"] = True
        if preserve_initial_cross_roots:
            kept_roots_now = set(detail_df.loc[~detail_df["dropped_by_root_gate"], "final_root"].fillna("").astype(str).str.strip())
            kept_roots_now.discard("")
            rescued_keep_roots, rescued = rescue_cross_preserving_roots(
                detail_df,
                kept_roots_now,
                initial_cross_roots,
                root_cap_max_roots,
                min_roots=2,
            )
            if rescued:
                cross_preserve_applied = True
                cross_preserve_rescued_roots.extend(rescued)
                rescue_mask = final_root_series.isin(rescued_keep_roots)
                detail_df.loc[rescue_mask, "dropped_by_root_gate"] = False
                detail_df.loc[rescue_mask, "dropped_by_quality_gate"] = False
                detail_df.loc[rescue_mask, "dropped_by_root_cap"] = False
                dropped_roots_by_root_cap = sorted(set(dropped_roots_by_root_cap) - set(rescued_keep_roots))
        quality_or_cap_drop = detail_df["dropped_by_quality_gate"] | detail_df["dropped_by_root_cap"]
        quality_gate_kept_keyword_count = int((~quality_or_cap_drop).sum())
        quality_gate_dropped_keyword_count = int(quality_or_cap_drop.sum())
        anchored_rows = detail_df[
            detail_df["is_anchored"] &
            detail_df["provisional_root"].fillna("").astype(str).str.strip().ne("")
        ].copy()
        anchored_keyword_count = int(len(anchored_rows))
        weak_keyword_count = int((~detail_df["is_anchored"]).sum())
        if not anchored_rows.empty:
            anchored_root_counts_series = anchored_rows["provisional_root"].astype(str).str.strip().value_counts()
            anchored_root_counts = {str(k): int(v) for k, v in anchored_root_counts_series.to_dict().items()}
            dominant_roots = sorted(anchored_root_counts)

        # revised_v3 keeps the post-quality-gate keyword set for Vt and applies root-strength gating only on Vs.
        kept_df = detail_df[~detail_df.get("dropped_by_root_gate", False)].copy()
        dropped_keyword_count = int(detail_df.get("dropped_by_root_gate", False).sum()) if hasattr(detail_df.get("dropped_by_root_gate", False), "sum") else 0
        kept_keyword_count_after_gate = int(len(kept_df))

        detail_df["a_hat"] = 0.0
        if not kept_df.empty:
            total_attn = kept_df["attn_mass"].sum()
            if total_attn > 0:
                kept_df["a_hat"] = kept_df["attn_mass"] / total_attn
                kept_df["keyword_quantile"] = pd.to_numeric(kept_df["keyword_quantile"], errors="coerce").fillna(0.0)
                vt_attention_scale = float(getattr(args, "historical_vt_attention_scale", 1.0))
                vt = 1.0 - vt_attention_scale * float((kept_df["a_hat"] * kept_df["keyword_quantile"]).sum())
                detail_df.loc[kept_df.index, "a_hat"] = kept_df["a_hat"]
            else:
                kept_df["a_hat"] = 0.0
        detail_df["Vt"] = vt

        if args.algo_profile in {"revised_v2", "revised_v3", "revised_v4", "revised_v5"}:
            vs_input = detail_df.copy()
            if args.algo_profile in {"revised_v4", "revised_v5"}:
                vs_input["final_root"] = vs_input["final_root"].fillna("").astype(str).str.strip()
            else:
                vs_input["final_root"] = vs_input["final_root"].astype(str).str.strip()
            vs_input = vs_input[vs_input["final_root"] != ""].copy()
            if "dropped_by_root_gate" in vs_input.columns:
                vs_input = vs_input[~vs_input["dropped_by_root_gate"].fillna(False).astype(bool)].copy()

            if args.algo_profile == "revised_v3" and not vs_input.empty:
                root_gate_applied = True
                root_gate_mode = "absolute_strength"
                strength_series = (
                    vs_input.assign(root_strength_component=vs_input["a_hat"] * vs_input["lexical_fidelity"])
                    .groupby("final_root", dropna=True)["root_strength_component"]
                    .sum()
                )
                root_strength_by_root = {
                    str(root): float(value)
                    for root, value in strength_series.to_dict().items()
                }
                dropped_roots_by_strength = sorted(
                    [str(root) for root, value in root_strength_by_root.items() if float(value) < float(args.root_strength_threshold)]
                )
                if dropped_roots_by_strength:
                    vs_input = vs_input[~vs_input["final_root"].isin(dropped_roots_by_strength)].copy()

            if args.algo_profile == "revised_v4" and not vs_input.empty:
                root_gate_applied = True
                vs_rows_before_root_gate = int(len(vs_input))
                strength_series = (
                    vs_input.assign(root_strength_component=vs_input["a_hat"] * vs_input["lexical_fidelity"])
                    .groupby("final_root", dropna=True)["root_strength_component"]
                    .sum()
                    .sort_values(ascending=False)
                )
                root_strength_by_root = {
                    str(root): float(value)
                    for root, value in strength_series.to_dict().items()
                }
                total_strength = float(strength_series.sum())
                if total_strength > 0:
                    root_share_by_root = {
                        str(root): float(value / total_strength)
                        for root, value in strength_series.to_dict().items()
                    }
                else:
                    root_share_by_root = {str(root): 0.0 for root in strength_series.index.tolist()}

                strong_series = (
                    vs_input[vs_input["is_strong_anchor"]]
                    .groupby("final_root", dropna=True)
                    .size()
                )
                strong_anchor_count_by_root = {
                    str(root): int(value)
                    for root, value in strong_series.to_dict().items()
                }

                root_counts = vs_input["final_root"].value_counts().to_dict()
                root_count = len(root_strength_by_root)
                keep_roots = set()
                if root_count == 0:
                    root_gate_mode = "no_roots"
                elif not any(count > 0 for count in strong_anchor_count_by_root.values()):
                    root_gate_mode = "no_strong_anchor_keep_top_root"
                    top_root = max(
                        root_strength_by_root,
                        key=lambda root: (
                            root_strength_by_root.get(root, 0.0),
                            root_counts.get(root, 0),
                            str(root),
                        ),
                    )
                    keep_roots.add(str(top_root))
                elif total_strength <= 0:
                    root_gate_mode = "zero_strength_keep_strong_roots"
                    keep_roots = {
                        str(root)
                        for root, count in strong_anchor_count_by_root.items()
                        if int(count) > 0
                    }
                else:
                    root_gate_mode = "distribution_strength"
                    p1 = max(root_share_by_root.values()) if root_share_by_root else 0.0
                    min_share = max(float(args.root_share_min_abs), float(args.root_share_min_factor) / max(root_count, 1))
                    for root, share in root_share_by_root.items():
                        ratio_to_top = float(share / p1) if p1 > 0 else 0.0
                        has_strong_anchor = int(strong_anchor_count_by_root.get(root, 0)) > 0
                        keep = (
                            share >= min_share
                            and ratio_to_top >= float(args.root_share_ratio_to_top)
                        )
                        if keep and (has_strong_anchor or ratio_to_top >= float(args.weak_root_share_ratio_to_top)):
                            keep_roots.add(str(root))

                all_roots = set(str(root) for root in root_strength_by_root)
                if keep_roots:
                    dropped_roots_by_distribution_gate = sorted(all_roots - keep_roots)
                    kept_roots_after_distribution_gate = sorted(keep_roots)
                    if dropped_roots_by_distribution_gate:
                        vs_input = vs_input[vs_input["final_root"].isin(keep_roots)].copy()
                else:
                    dropped_roots_by_distribution_gate = sorted(all_roots)
                    kept_roots_after_distribution_gate = []
                    vs_input = vs_input.iloc[0:0].copy()
                vs_rows_after_distribution_gate = int(len(vs_input))

            if args.algo_profile == "revised_v5" and not vs_input.empty:
                root_gate_applied = True
                vs_rows_before_root_gate = int(len(vs_input))
                gate = gate_roots_revised_v5(vs_input, dist_dict, args)
                root_strength_by_root = gate["root_strength_by_root"]
                root_share_by_root = gate["root_share_by_root"]
                strong_anchor_count_by_root = gate["strong_anchor_count_by_root"]
                root_gate_mode = gate["root_gate_mode"]
                keep_roots = gate["keep_roots_before_singleton"]
                if preserve_initial_cross_roots:
                    keep_roots, rescued = rescue_cross_preserving_roots(
                        vs_input,
                        keep_roots,
                        initial_cross_roots,
                        root_cap_max_roots,
                        min_roots=2,
                    )
                    if rescued:
                        cross_preserve_applied = True
                        cross_preserve_rescued_roots.extend(rescued)
                        root_gate_mode = f"{root_gate_mode}+cross_preserve"
                all_gate_roots = set(str(root) for root in root_strength_by_root)
                dropped_roots_by_distribution_gate = sorted(all_gate_roots - set(keep_roots))
                kept_roots_after_distribution_gate = sorted(keep_roots)
                if keep_roots:
                    vs_input = vs_input[vs_input["final_root"].isin(keep_roots)].copy()
                else:
                    vs_input = vs_input.iloc[0:0].copy()
                vs_rows_after_distribution_gate = int(len(vs_input))

            vs_rows_before_filter = int(len(vs_input))
            distinct_final_root_before_filter = int(vs_input["final_root"].nunique()) if not vs_input.empty else 0
            root_counts = vs_input["final_root"].value_counts()
            singleton_roots = set(root_counts[root_counts == 1].index.tolist())
            if preserve_initial_cross_roots and len(set(vs_input["final_root"].astype(str))) >= 2 and len(initial_cross_roots) >= 2:
                singleton_roots = set()
            singleton_root_count = int(len(singleton_roots))
            dropped_roots_by_singleton_gate = sorted(str(root) for root in singleton_roots)
            if singleton_roots:
                vs_input = vs_input[~vs_input["final_root"].isin(singleton_roots)].copy()
            vs_rows_after_filter = int(len(vs_input))
            vs_rows_after_singleton_gate = vs_rows_after_filter
            distinct_final_root_after_filter = int(vs_input["final_root"].nunique()) if not vs_input.empty else 0
            if args.algo_profile == "revised_v5":
                vs_rows_before_filter = vs_rows_before_root_gate
                distinct_final_root_before_filter = len(root_strength_by_root)
                dominant_roots = sorted(vs_input["final_root"].unique().tolist()) if not vs_input.empty else []
            if args.algo_profile in {"revised_v4", "revised_v5"}:
                dropped_roots_all = set(dropped_roots_by_distribution_gate) | set(dropped_roots_by_singleton_gate)
                if dropped_roots_all:
                    detail_roots = detail_df["final_root"].fillna("").astype(str).str.strip()
                    detail_df.loc[detail_roots.isin(dropped_roots_all), "dropped_by_root_gate"] = True
        else:
            vs_input = detail_df.dropna().copy()
            if "dropped_by_root_gate" in vs_input.columns:
                vs_input = vs_input[~vs_input["dropped_by_root_gate"].fillna(False).astype(bool)].copy()
            if "final_root" in vs_input.columns:
                vs_input["final_root"] = vs_input["final_root"].astype(str).str.strip()
            vs_rows_before_filter = int(len(vs_input))
            vs_rows_after_filter = int(len(vs_input))
            vs_rows_before_root_gate = vs_rows_before_filter
            vs_rows_after_distribution_gate = vs_rows_before_filter
            vs_rows_after_singleton_gate = vs_rows_after_filter
            distinct_final_root_before_filter = int(vs_input["final_root"].nunique()) if not vs_input.empty else 0
            distinct_final_root_after_filter = distinct_final_root_before_filter
    else:
        vs_input = detail_df

    metric_df = vs_input
    _weights, vs = compute_vs(
        metric_df,
        dist_dict,
        args.threshold_subjects,
        default_distance=1.0 if args.algo_profile == "revised_v5" else 0.0,
    )
    xindex_zero_reason = ""
    if distinct_final_root_after_filter <= 1 or vs <= 0:
        if distinct_final_root_after_filter <= 0:
            xindex_zero_reason = "no_valid_root"
        elif distinct_final_root_after_filter == 1:
            xindex_zero_reason = "single_valid_root"
        elif vs <= 0:
            xindex_zero_reason = "zero_vs"
        cross_index = 0.0
    else:
        cross_index = vs * vt
    summary = {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "year": int(doc["year"]),
        "prompt_variant": args.prompt_variant,
        "cutoff_year": get_cutoff_year(doc, args.cutoff_year_field),
        "algo_profile": args.algo_profile,
        "status": "ok",
        "keyword_count": len(mag_payload.get("matches", [])),
        "matched_keyword_count": len(kwid_to_rootids),
        "detail_rows": int(len(detail_df)),
        "Vs": float(vs),
        "Vt": float(vt),
        "Vt_prime": float(vt),
        "cross_index": float(cross_index),
        "xindex_zero_reason": xindex_zero_reason,
        "vs_rows_before_filter": vs_rows_before_filter,
        "vs_rows_after_filter": vs_rows_after_filter,
        "singleton_root_count": singleton_root_count,
        "distinct_final_root_before_filter": distinct_final_root_before_filter,
        "distinct_final_root_after_filter": distinct_final_root_after_filter,
        "dominant_roots": dominant_roots,
        "anchored_root_counts": anchored_root_counts,
        "anchored_keyword_count": anchored_keyword_count,
        "weak_keyword_count": weak_keyword_count,
        "root_gate_applied": root_gate_applied,
        "dropped_keyword_count": dropped_keyword_count,
        "kept_keyword_count_after_gate": kept_keyword_count_after_gate,
        "root_strength_by_root": root_strength_by_root,
        "root_share_by_root": root_share_by_root,
        "dropped_roots_by_strength": dropped_roots_by_strength,
        "strong_anchor_count_by_root": strong_anchor_count_by_root,
        "root_gate_mode": root_gate_mode,
        "kept_roots_after_distribution_gate": kept_roots_after_distribution_gate,
        "dropped_roots_by_distribution_gate": dropped_roots_by_distribution_gate,
        "dropped_roots_by_singleton_gate": dropped_roots_by_singleton_gate,
        "dropped_keywords_by_quality_gate": dropped_keywords_by_quality_gate,
        "dropped_roots_by_root_cap": dropped_roots_by_root_cap,
        "quality_gate_kept_keyword_count": quality_gate_kept_keyword_count,
        "quality_gate_dropped_keyword_count": quality_gate_dropped_keyword_count,
        "root_cap_applied": root_cap_applied,
        "root_cap_max_roots": root_cap_max_roots,
        "xindex_preserve_initial_cross_roots": preserve_initial_cross_roots,
        "initial_provisional_root_count": int(len(initial_provisional_roots)),
        "initial_provisional_roots": sorted(initial_provisional_roots),
        "initial_candidate_root_count": int(len(initial_candidate_roots)),
        "initial_candidate_roots": sorted(initial_candidate_roots),
        "cross_preserve_applied": bool(cross_preserve_applied),
        "cross_preserve_rescued_roots": sorted(set(cross_preserve_rescued_roots)),
        "xindex_min_lexical_fidelity": float(getattr(args, "xindex_min_lexical_fidelity", args.lexical_fidelity_threshold)),
        "xindex_allow_embedding_fallback": bool(getattr(args, "xindex_allow_embedding_fallback", False)),
        "vs_rows_before_root_gate": vs_rows_before_root_gate,
        "vs_rows_after_distribution_gate": vs_rows_after_distribution_gate,
        "vs_rows_after_singleton_gate": vs_rows_after_singleton_gate,
    }
    if not detail_df.empty:
        detail_df["prompt_variant"] = args.prompt_variant
        detail_df["cutoff_year"] = get_cutoff_year(doc, args.cutoff_year_field)
    details = json.loads(detail_df.replace({np.nan: None}).to_json(orient="records", force_ascii=False))
    return summary, details


def load_llama_model(model_dir: str, tokenizer_fast: bool, eager: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=tokenizer_fast)
    kwargs = {"torch_dtype": torch.float16}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": "cuda:0"}
    else:
        kwargs["device_map"] = None
    if eager:
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs).eval()
    return model, tokenizer


def free_model(model=None, tokenizer=None):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description="Run one shard of large-batch paper X-index.")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--flush_every", type=int, default=20)
    ap.add_argument("--run_stage", choices=["all", "keywords", "mag", "xindex"], default="all")
    ap.add_argument("--algo_profile", choices=["legacy_batch", "revised_v2", "revised_v3", "revised_v4", "revised_v5"], default="legacy_batch")
    ap.add_argument("--keyword_input", choices=["abstract", "title_abstract"], default="abstract")
    ap.add_argument(
        "--prompt_variant",
        choices=["baseline", "cutoff_year", "historical_distance", "historical_distance_compact"],
        default="baseline",
    )
    ap.add_argument("--xindex_profile", choices=["existing", "original", "historical_acemap_temporal", "static_2021"], default="existing")
    ap.add_argument("--keyword_field_mode", choices=["existing", "historical_acemap"], default="existing")
    ap.add_argument("--distance_mode", choices=["existing", "temporal_node2vec", "static_2021"], default="existing")
    ap.add_argument("--distance_resource_dir", default=DEFAULT_DISTANCE_RESOURCE_DIR)
    ap.add_argument("--historical_q_alpha", type=float, default=1e-6)
    ap.add_argument("--historical_q_cache_dir", default="")
    ap.add_argument("--historical_q_global_cache_path", default=str(REPO_ROOT / "resources" / "historical_q_cache" / "global_historical_q.jsonl"))
    ap.add_argument("--keyword_salience_mode", choices=["uniform", "lexical", "existing"], default="existing")
    ap.add_argument("--historical_attention_lambda", type=float, default=1.0)
    ap.add_argument("--historical_temporal_penalty_gamma", type=float, default=0.0)
    ap.add_argument("--historical_vt_attention_scale", type=float, default=1.0)
    ap.add_argument("--subject_matrix_lambda", type=float, default=1.0)
    ap.add_argument("--historical_root_filter_mode", choices=["baseline_compatible", "revised_v5_strict"], default="baseline_compatible")
    ap.add_argument("--es_index", default="acemap.works")
    ap.add_argument("--result_suffix", default="")
    ap.add_argument("--cutoff_year_field", default="year")
    ap.add_argument("--historical_prompt_dir", default=DEFAULT_HISTORICAL_PROMPT_DIR)
    ap.add_argument("--historical_prompt_max_chars", type=int, default=6000)
    ap.add_argument("--historical_compact_max_chars", type=int, default=2400)
    ap.add_argument("--historical_compact_max_pairs", type=int, default=30)
    ap.add_argument("--max_keywords", type=int, default=15)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--keyword_oom_retry_tokens", default="")
    ap.add_argument("--llama_model", default="/data1/lixinhui-20418/llama3_model/llama3-8b-instruct")
    ap.add_argument("--sbert_model", default="/home/lixinhui/code/SBERT/all-MiniLM-L6-v2")
    ap.add_argument("--prefix_mode", choices=["SBERT", "e5"], default="SBERT")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--batch_enc", type=int, default=256)
    ap.add_argument("--topk_lv0", type=int, default=2)
    ap.add_argument("--keep_percent", type=float, default=0.2)
    ap.add_argument("--candidate_pool_size", type=int, default=20)
    ap.add_argument("--mag_match_mode", choices=["hierarchical", "hash_cluster", "hash_topdown"], default="hierarchical")
    ap.add_argument("--mag_hash_index_dir", default=str(REPO_ROOT / "resources" / "mag_hash_index"))
    ap.add_argument("--mag_offline_cache_dir", default="")
    ap.add_argument("--mag_hash_top_clusters_per_level", type=int, default=8)
    ap.add_argument("--mag_hash_target_level_top_clusters", type=int, default=12)
    ap.add_argument("--mag_hash_max_candidate_nodes", type=int, default=512)
    ap.add_argument("--mag_hash_fast_mode", action="store_true")
    ap.add_argument("--lexical_fidelity_threshold", type=float, default=0.60)
    ap.add_argument("--xindex_min_lexical_fidelity", type=float, default=0.60)
    ap.add_argument("--xindex_allow_embedding_fallback", action="store_true")
    ap.add_argument("--xindex_max_roots_per_paper", type=int, default=3)
    ap.add_argument("--xindex_preserve_initial_cross_roots", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed_keywords_results_dir", default="")
    ap.add_argument("--seed_mag_results_dir", default="")
    ap.add_argument("--llm_cross_gate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--llm_non_cross_conf_threshold", type=float, default=0.85,
        help="Deprecated; retained for CLI compatibility and no longer gates MAG matching.",
    )
    ap.add_argument("--dominant_root_min_count", type=int, default=2)
    ap.add_argument("--root_strength_threshold", type=float, default=0.12)
    ap.add_argument("--root_share_min_abs", type=float, default=0.03)
    ap.add_argument("--root_share_min_factor", type=float, default=0.33)
    ap.add_argument("--root_share_ratio_to_top", type=float, default=0.35)
    ap.add_argument("--weak_root_share_ratio_to_top", type=float, default=0.60)
    ap.add_argument("--close_distance", type=float, default=0.32)
    ap.add_argument("--far_distance", type=float, default=0.38)
    ap.add_argument("--strong2_min_share", type=float, default=0.12)
    ap.add_argument("--strong2_min_ratio", type=float, default=0.30)
    ap.add_argument("--strong1_min_share", type=float, default=0.14)
    ap.add_argument("--strong1_min_ratio", type=float, default=0.35)
    ap.add_argument("--mid_min_share", type=float, default=0.20)
    ap.add_argument("--mid_min_ratio", type=float, default=0.50)
    ap.add_argument("--start_level", type=int, default=0)
    ap.add_argument("--max_level", type=int, default=5)
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--sink_first_k", type=int, default=1)
    ap.add_argument("--save_attention_maps", action="store_true")
    ap.add_argument("--attention_map_dir", default="")
    ap.add_argument("--attention_map_head", type=int, default=-1, help="-1 means average all heads.")
    ap.add_argument("--attention_map_png_max_tokens", type=int, default=512)
    ap.add_argument("--attention_map_save_raw", action="store_true")
    ap.add_argument("--dist_csv", default=DEFAULT_DIST_CSV)
    ap.add_argument("--mag_parquet", default=DEFAULT_MAG_PARQUET)
    ap.add_argument("--threshold_subjects", type=float, default=0.321)
    ap.add_argument("--es_url", default="http://readonly:readonly@10.10.12.1:9201")
    ap.add_argument("--db_host", default="10.10.12.1")
    ap.add_argument("--db_port", type=int, default=3306)
    ap.add_argument("--db_user", default="readonly_ampaper")
    ap.add_argument("--db_pass", default="readonly@ampaper1")
    ap.add_argument("--db_name", default="am_paper")
    ap.add_argument("--table_field", default="am_field")
    ap.add_argument("--table_rel", default="am_field_relation")
    args = ap.parse_args()
    args = apply_profile_defaults(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; using CPU for SBERT.")
        args.device = "cpu"

    shard_path = Path(args.shard)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = read_jsonl(shard_path)

    keywords_path = out_dir / "keywords.jsonl"
    mag_path = out_dir / "mag_matches.jsonl"
    suffix = str(args.result_suffix or "").strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    xindex_path = out_dir / f"xindex_partial{suffix}.jsonl"
    detail_path = out_dir / f"xindex_detail{suffix}.jsonl"
    attention_path = out_dir / f"attention{suffix}.jsonl"
    error_path = out_dir / "errors.jsonl"
    progress_path = out_dir / "progress.json"

    llama_model = None
    gen_tok = None
    attn_tok = None

    seed_stage_file(keywords_path, args.seed_keywords_results_dir, shard_path.stem, "keywords.jsonl")
    keyword_map = read_jsonl_map(keywords_path)
    missing_keywords = [
        d for d in docs
        if d["doc_id"] not in keyword_map
        or keyword_map[d["doc_id"]].get("status") == "non_cross_high_conf"
    ]
    if missing_keywords and args.run_stage in {"all", "keywords", "mag"}:
        print(f"[PHASE1] keyword extraction pending={len(missing_keywords)}")
        llama_model, gen_tok = load_llama_model(args.llama_model, tokenizer_fast=False, eager=True)
        for i, doc in enumerate(missing_keywords, 1):
            try:
                kws, raw, parsed_gate = generate_keywords_with_retry(doc, llama_model, gen_tok, args)
                row = {
                    "doc_id": doc["doc_id"],
                    "keywords": kws,
                    "keyword_roots": parsed_gate.get("keywords", []),
                    "keyword_root_map": {
                        item["keyword"]: item.get("root", "")
                        for item in parsed_gate.get("keywords", [])
                        if item.get("keyword")
                    },
                    "raw_answer": raw,
                    "status": "ok",
                    "llm_cross_decision": parsed_gate.get("decision"),
                    "llm_cross_confidence": parsed_gate.get("confidence"),
                    "llm_cross_reason": parsed_gate.get("reason"),
                }
                append_jsonl(keywords_path, row)
                keyword_map[doc["doc_id"]] = row
            except Exception as exc:
                append_jsonl(error_path, {"doc_id": doc["doc_id"], "phase": "keywords", "error": repr(exc)})
            if i % args.flush_every == 0:
                write_progress(progress_path, phase="keywords", done=i, pending=len(missing_keywords))
    else:
        if missing_keywords:
            print(f"[PHASE1] keyword extraction skipped by run_stage={args.run_stage}; pending={len(missing_keywords)}")
        else:
            print("[PHASE1] keyword extraction already complete")

    if args.run_stage == "keywords":
        if llama_model is not None:
            free_model(llama_model, gen_tok)
            llama_model = None
            gen_tok = None
        write_progress(
            progress_path,
            phase="keywords_only",
            docs=len(docs),
            keywords=len(read_jsonl_map(keywords_path)),
            mag=len(read_jsonl_map(mag_path)),
            xindex=len(read_jsonl_map(xindex_path)),
        )
        print(f"[DONE] shard={shard_path} out={out_dir} stage=keywords")
        return

    if llama_model is not None:
        free_model(llama_model, gen_tok)
        llama_model = None
        gen_tok = None

    seed_stage_file(mag_path, args.seed_mag_results_dir, shard_path.stem, "mag_matches.jsonl")
    mag_map = read_jsonl_map(mag_path)
    missing_mag = [
        d for d in docs
        if d["doc_id"] not in mag_map
        and d["doc_id"] in keyword_map
    ]
    if missing_mag and args.run_stage in {"all", "mag"}:
        print(f"[PHASE2] MAG matching pending={len(missing_mag)}")
        mag, engine = create_mag_cache(args)
        sbert = SentenceTransformer(args.sbert_model, device=args.device).eval()
        mag_hash_index = None
        lv0_ids = np.array([], dtype=np.int64)
        e_lv0 = np.zeros((0, 1), dtype=np.float32)
        if args.mag_match_mode in {"hash_cluster", "hash_topdown"}:
            try:
                print(f"[PHASE2] loading MAG hash index: {args.mag_hash_index_dir}")
                mag_hash_index = MAGHashIndex(args.mag_hash_index_dir)
            except Exception as index_exc:
                print(f"[WARN] failed to load MAG hash index; fallback=hierarchical error={repr(index_exc)}")
                args.mag_match_mode = "hierarchical"
        if args.mag_match_mode not in {"hash_cluster", "hash_topdown"}:
            lv0 = mag.level_nodes(args.start_level)
            lv0_ids = lv0["field_id"].to_numpy(dtype=np.int64)
            mag.ensure_embeds(lv0_ids.tolist(), sbert, args.batch_enc, args.device, args.prefix_mode)
            e_lv0 = np.stack([mag.embed_cache[int(i)] for i in lv0_ids], axis=0)
        for i, doc in enumerate(missing_mag, 1):
            try:
                kws, root_by_term = keyword_terms_and_roots(keyword_map[doc["doc_id"]])
                if args.mag_match_mode in {"hash_cluster", "hash_topdown"}:
                    try:
                        if args.mag_match_mode == "hash_topdown":
                            matches = match_keywords_to_mag_hash_topdown(kws, args, sbert, mag_hash_index, mag, root_by_term)
                        else:
                            matches = match_keywords_to_mag_hash(kws, args, sbert, mag_hash_index, mag, root_by_term)
                    except Exception as hash_exc:
                        print(f"[WARN] {args.mag_match_mode} MAG matching failed; fallback=hierarchical doc_id={doc.get('doc_id')} error={repr(hash_exc)}")
                        if e_lv0.shape[0] == 0:
                            lv0 = mag.level_nodes(args.start_level)
                            lv0_ids = lv0["field_id"].to_numpy(dtype=np.int64)
                            mag.ensure_embeds(lv0_ids.tolist(), sbert, args.batch_enc, args.device, args.prefix_mode)
                            e_lv0 = np.stack([mag.embed_cache[int(i)] for i in lv0_ids], axis=0)
                        matches = match_keywords_to_mag(kws, args, sbert, mag, e_lv0, lv0_ids, root_by_term)
                else:
                    matches = match_keywords_to_mag(kws, args, sbert, mag, e_lv0, lv0_ids, root_by_term)
                payload = build_mag_payload(doc["doc_id"], matches, mag)
                payload["llm_keyword_roots"] = keyword_map[doc["doc_id"]].get("keyword_roots", [])
                payload["llm_cross_decision"] = keyword_map[doc["doc_id"]].get("llm_cross_decision")
                payload["llm_cross_confidence"] = keyword_map[doc["doc_id"]].get("llm_cross_confidence")
                payload["llm_cross_reason"] = keyword_map[doc["doc_id"]].get("llm_cross_reason")
                if args.algo_profile in {"revised_v2", "revised_v3", "revised_v4", "revised_v5"}:
                    payload["prompt_keywords"] = build_predecessor_style_prompt_keywords(payload.get("paths", []), mag)
                append_jsonl(mag_path, payload)
                mag_map[doc["doc_id"]] = payload
            except Exception as exc:
                append_jsonl(error_path, {"doc_id": doc["doc_id"], "phase": "mag", "error": repr(exc)})
            if i % args.flush_every == 0:
                write_progress(progress_path, phase="mag", done=i, pending=len(missing_mag))
        del sbert
        if engine is not None:
            engine.dispose()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        if missing_mag:
            print(f"[PHASE2] MAG matching skipped by run_stage={args.run_stage}; pending={len(missing_mag)}")
        else:
            print("[PHASE2] MAG matching already complete")

    if args.run_stage == "mag":
        write_progress(
            progress_path,
            phase="mag_only",
            docs=len(docs),
            keywords=len(read_jsonl_map(keywords_path)),
            mag=len(read_jsonl_map(mag_path)),
            xindex=len(read_jsonl_map(xindex_path)),
        )
        print(f"[DONE] shard={shard_path} out={out_dir} stage=mag")
        return

    xindex_map = {
        doc_id: row
        for doc_id, row in read_jsonl_map(xindex_path).items()
        if row.get("status") != "llm_high_conf_non_cross"
    }
    pending_xindex = [
        d for d in docs
        if d["doc_id"] not in xindex_map
        and d["doc_id"] in mag_map
    ]
    if pending_xindex and args.run_stage in {"all", "xindex"}:
        print(f"[PHASE3] attention + X-index pending={len(pending_xindex)}")
        dist_dict = load_subject_distances(args.dist_csv)
        es_client = Elasticsearch(args.es_url, request_timeout=60)
        es_count_cache, quantile_cache = {}, {}
        historical_distance_loader = None
        historical_q_cache = None
        if args.xindex_profile in HISTORICAL_XINDEX_PROFILES or args.keyword_field_mode == "historical_acemap" or args.distance_mode in {"temporal_node2vec", "static_2021"}:
            if args.keyword_field_mode == "existing":
                args.keyword_field_mode = "historical_acemap"
            if args.distance_mode == "existing":
                args.distance_mode = "temporal_node2vec"
            historical_distance_loader = TemporalDistanceLoader(args.distance_resource_dir)
            # Force validation before the long loop so field-order/resource issues fail loudly.
            historical_distance_loader.root_fields()
            if not args.historical_q_cache_dir:
                work_dir_for_cache = out_dir.parent.parent if out_dir.parent.name == "results" else out_dir
                args.historical_q_cache_dir = str(work_dir_for_cache / "cache" / "historical_acemap_q")
            task_id = os.environ.get("SLURM_ARRAY_TASK_ID") or shard_path.stem.replace("shard_", "")
            shard_cache_path = Path(args.historical_q_cache_dir) / f"shard_{task_id}.jsonl"
            global_cache_path = Path(args.historical_q_global_cache_path) if args.historical_q_global_cache_path else shard_cache_path
            historical_q_cache = HistoricalQCache(str(global_cache_path), read_paths=[str(shard_cache_path)])
        prompt_engine = None
        prompt_mag = None
        if llama_model is None:
            llama_model, attn_tok = load_llama_model(args.llama_model, tokenizer_fast=True, eager=True)
        else:
            attn_tok = AutoTokenizer.from_pretrained(args.llama_model, use_fast=True)
        for i, doc in enumerate(pending_xindex, 1):
            try:
                payload = mag_map[doc["doc_id"]]
                if args.algo_profile in {"revised_v2", "revised_v3", "revised_v4", "revised_v5"} and not payload.get("prompt_keywords"):
                    if prompt_mag is None:
                        prompt_mag, prompt_engine = create_mag_cache(args)
                    payload["prompt_keywords"] = build_predecessor_style_prompt_keywords(payload.get("paths", []), prompt_mag)
                prompt, prompt_meta = build_attention_prompt_for_doc(doc, payload.get("prompt_keywords", []), args, payload)
                if args.save_attention_maps:
                    attn, attn_map = compute_attention_masses(prompt, llama_model, attn_tok, args, return_map=True)
                else:
                    attn = compute_attention_masses(prompt, llama_model, attn_tok, args)
                    attn_map = None
                summary, details = compute_xindex(
                    doc, payload, attn, dist_dict, args.mag_parquet,
                    es_client, es_count_cache, quantile_cache, args,
                )
                keyword_row = keyword_map.get(doc["doc_id"], {})
                summary["llm_cross_decision"] = keyword_row.get("llm_cross_decision")
                summary["llm_cross_confidence"] = keyword_row.get("llm_cross_confidence")
                summary["llm_cross_reason"] = keyword_row.get("llm_cross_reason")
                summary["llm_keyword_roots"] = keyword_row.get("keyword_roots", [])
                if historical_distance_loader is not None:
                    historical_summary = compute_historical_temporal_xindex(
                        doc,
                        details,
                        payload.get("prompt_keywords", []),
                        es_client,
                        args,
                        cache=historical_q_cache,
                        distance_loader=historical_distance_loader,
                    )
                    summary.update(historical_summary)
                summary.update(prompt_meta)
                append_jsonl(xindex_path, summary)
                append_jsonl(detail_path, {"doc_id": doc["doc_id"], "details": details})
                attention_map_artifacts = {}
                if args.save_attention_maps and attn_map is not None:
                    attention_map_artifacts = save_attention_map_artifacts(
                        doc,
                        prompt,
                        attn_map.get("tokens", []),
                        attn_map["attention"],
                        prompt_meta,
                        args,
                        out_dir,
                    )
                append_jsonl(attention_path, {
                    "doc_id": doc["doc_id"],
                    "prompt_variant": args.prompt_variant,
                    "cutoff_year": prompt_meta.get("cutoff_year"),
                    "historical_prompt_loaded": prompt_meta.get("historical_prompt_loaded"),
                    "historical_prompt_chars": prompt_meta.get("historical_prompt_chars"),
                    "historical_prompt_roots": prompt_meta.get("historical_prompt_roots"),
                    "prompt_char_count": len(prompt),
                    "prompt_keywords": payload.get("prompt_keywords", []),
                    "attention_masses": attn,
                    "attention_map": attention_map_artifacts,
                })
                xindex_map[doc["doc_id"]] = summary
            except Exception as exc:
                append_jsonl(error_path, {"doc_id": doc["doc_id"], "phase": "xindex", "error": repr(exc)})
            if i % args.flush_every == 0:
                write_progress(progress_path, phase="xindex", done=i, pending=len(pending_xindex))
        if prompt_engine is not None:
            prompt_engine.dispose()
    else:
        if pending_xindex:
            print(f"[PHASE3] attention + X-index skipped by run_stage={args.run_stage}; pending={len(pending_xindex)}")
        else:
            print("[PHASE3] attention + X-index already complete")

    if llama_model is not None:
        free_model(llama_model, gen_tok)
    if attn_tok is not None and attn_tok is not gen_tok:
        del attn_tok

    write_progress(
        progress_path,
        phase="done",
        docs=len(docs),
        keywords=len(read_jsonl_map(keywords_path)),
        mag=len(read_jsonl_map(mag_path)),
        xindex=len(read_jsonl_map(xindex_path)),
    )
    print(f"[DONE] shard={shard_path} out={out_dir}")


if __name__ == "__main__":
    main()
