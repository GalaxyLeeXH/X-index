#!/usr/bin/env python3
"""Historically grounded X-index utilities.

This module intentionally avoids modern embedding or LLM calls for historical
keyword-field distributions.  Q_f^{<t} is estimated from Acemap/Elasticsearch
historical title/abstract counts with strict publication_date < paper_year.
"""

import fcntl
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_DISTANCE_RESOURCE_DIR = str(
    Path(__file__).resolve().parent.parent
    / "resources"
    / "node2vec_discipline_distances_2011-2021"
)

STATIC_DISTANCE_YEAR = 2021
ALPHA_DEFAULT = 1e-6
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "by", "for", "from", "in",
    "into", "is", "method", "methods", "model", "models", "of", "on", "or",
    "the", "to", "using", "via", "with",
}


def normalize_keyword(value: str) -> str:
    text = str(value or "").lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text


def clean_keyword_for_query(value: str) -> str:
    text = normalize_keyword(value)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    return text.strip()


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class TemporalDistanceLoader:
    def __init__(self, distance_dir: str = DEFAULT_DISTANCE_RESOURCE_DIR):
        self.distance_dir = Path(distance_dir)
        self._years: Optional[List[int]] = None
        self._root_fields: Optional[List[str]] = None
        self._matrices: Dict[int, np.ndarray] = {}

    def available_years(self) -> List[int]:
        if self._years is None:
            years = []
            for path in self.distance_dir.glob("node2vec_discipline_distances_*.json"):
                m = re.search(r"(\d{4})\.json$", path.name)
                if m:
                    years.append(int(m.group(1)))
            self._years = sorted(set(years))
            if not self._years:
                raise FileNotFoundError(f"no temporal distance json files under {self.distance_dir}")
        return list(self._years)

    def root_fields(self) -> List[str]:
        if self._root_fields is None:
            years = self.available_years()
            first = self._load_pairs(years[0])
            fields = sorted({p["d1"] for p in first} | {p["d2"] for p in first})
            if len(fields) != 26:
                raise ValueError(
                    f"cannot confirm 26 root fields from {years[0]} distance file; got {len(fields)}"
                )
            expected_pairs = len(fields) * (len(fields) - 1) // 2
            for year in years:
                pairs = self._load_pairs(year)
                year_fields = sorted({p["d1"] for p in pairs} | {p["d2"] for p in pairs})
                if year_fields != fields:
                    raise ValueError(f"distance field set differs in year {year}")
                if len(pairs) != expected_pairs:
                    raise ValueError(
                        f"distance file for {year} has {len(pairs)} pairs; expected {expected_pairs}"
                    )
            self._root_fields = fields
        return list(self._root_fields)

    def _load_pairs(self, year: int) -> List[dict]:
        path = self.distance_dir / f"node2vec_discipline_distances_{int(year)}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"distance file must be a list of pairs: {path}")
        out = []
        for row in data:
            d1 = str(row.get("d1") or "").strip()
            d2 = str(row.get("d2") or "").strip()
            if not d1 or not d2:
                continue
            out.append({"d1": d1, "d2": d2, "distance": float(row["distance"])})
        return out

    def choose_year(self, requested_year: int) -> Tuple[int, str]:
        years = self.available_years()
        requested = int(requested_year)
        if requested in years:
            return requested, "exact"
        if requested < years[0]:
            return years[0], "boundary_clipped"
        if requested > years[-1]:
            return years[-1], "boundary_clipped"
        before = [y for y in years if y < requested]
        if before:
            return before[-1], "nearest_available_before"
        after = [y for y in years if y > requested]
        return after[0], "nearest_available_after"

    def load(self, requested_year: int) -> Tuple[np.ndarray, dict]:
        used_year, reason = self.choose_year(requested_year)
        if used_year not in self._matrices:
            fields = self.root_fields()
            idx = {field: i for i, field in enumerate(fields)}
            matrix = np.zeros((len(fields), len(fields)), dtype=np.float64)
            for row in self._load_pairs(used_year):
                i = idx[row["d1"]]
                j = idx[row["d2"]]
                matrix[i, j] = matrix[j, i] = float(row["distance"])
            if matrix.shape != (26, 26):
                raise ValueError(f"temporal distance matrix shape is {matrix.shape}, expected (26, 26)")
            self._matrices[used_year] = matrix
        meta = {
            "distance_year_requested": int(requested_year),
            "distance_year_used": int(used_year),
            "distance_year_reason": reason,
            "distance_resource_dir": str(self.distance_dir),
            "distance_field_order": self.root_fields(),
        }
        return self._matrices[used_year].copy(), meta


class HistoricalQCache:
    def __init__(self, cache_path: str, read_paths: Optional[List[str]] = None):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.cache_path.with_suffix(self.cache_path.suffix + ".lock")
        self.rows = {}
        paths = []
        for item in read_paths or []:
            if item:
                paths.append(Path(item))
        paths.append(self.cache_path)
        seen = set()
        for path in paths:
            if str(path) in seen:
                continue
            seen.add(str(path))
            for row in iter_jsonl(path) or []:
                key = row.get("cache_key")
                if key:
                    self.rows[str(key)] = row.get("value", row)

    @staticmethod
    def make_key(keyword: str, paper_year: int, root_fields: Iterable[str]) -> str:
        version = "|".join(root_fields)
        return json.dumps(
            [normalize_keyword(keyword), int(paper_year), version],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def get(self, keyword: str, paper_year: int, root_fields: Iterable[str]):
        return self.rows.get(self.make_key(keyword, paper_year, root_fields))

    def set(self, keyword: str, paper_year: int, root_fields: Iterable[str], value: dict):
        key = self.make_key(keyword, paper_year, root_fields)
        if key in self.rows:
            return
        self.rows[key] = value
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                with self.cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"cache_key": key, "value": value}, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def set_many(self, rows: List[Tuple[str, int, dict]], root_fields: Iterable[str]):
        pending = []
        for keyword, paper_year, value in rows:
            key = self.make_key(keyword, int(paper_year), root_fields)
            if key in self.rows:
                continue
            self.rows[key] = value
            pending.append({"cache_key": key, "value": value})
        if not pending:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                with self.cache_path.open("a", encoding="utf-8") as f:
                    for row in pending:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def phrase_components(keyword: str) -> List[str]:
    base = clean_keyword_for_query(keyword)
    candidates = [base]
    dehyphen = base.replace("-", " ")
    candidates.append(dehyphen)
    tokens = [t for t in re.split(r"[^a-z0-9]+", dehyphen) if len(t) >= 2 and t not in STOPWORDS]
    for n in range(min(3, len(tokens)), 1, -1):
        for i in range(0, len(tokens) - n + 1):
            candidates.append(" ".join(tokens[i:i + n]))
    candidates.extend(tokens)
    out = []
    seen = set()
    for item in candidates:
        item = clean_keyword_for_query(item)
        if len(item) < 2 or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _count_query(keyword: Optional[str], root_field: str, paper_year: int) -> dict:
    must = [{"term": {"topics.field.display_name.keyword": root_field}}]
    if keyword:
        should = [
            {"match_phrase": {field: keyword}}
            for field in [
                "title",
                "abstract",
                "keywords",
                "keywords.display_name",
                "concepts.display_name",
                "concepts.normalized_name",
                "topics.display_name",
                "normalized_terms",
            ]
        ]
        must.append({"bool": {"should": should, "minimum_should_match": 1}})
    return {
        "query": {
            "bool": {
                "must": must,
                "filter": [{"range": {"publication_date": {"lt": f"{int(paper_year)}-01-01"}}}],
            }
        }
    }


def _safe_es_count(es_client, index: str, body: dict) -> int:
    if es_client is None:
        return 0
    try:
        resp = es_client.count(index=index, body=body)
        return int(resp.get("count", 0))
    except Exception:
        return 0




def _total_hits_value(total) -> int:
    if isinstance(total, dict):
        return int(total.get("value", 0) or 0)
    try:
        return int(total or 0)
    except Exception:
        return 0


def _safe_es_msearch_counts(es_client, index: str, requests: List[Tuple[str, str, dict]]) -> Dict[Tuple[str, str], int]:
    if es_client is None or not requests:
        return {(keyword, field): 0 for keyword, field, _body in requests}
    body = []
    for _keyword, _field, count_body in requests:
        body.append({"index": index})
        body.append({"size": 0, "track_total_hits": True, "query": count_body.get("query", {"match_none": {}})})
    try:
        resp = es_client.msearch(body=body, request_timeout=60)
        responses = resp.get("responses", [])
    except Exception:
        return {(keyword, field): _safe_es_count(es_client, index, body) for keyword, field, body in requests}
    out = {}
    for (keyword, field, _body), item in zip(requests, responses):
        if item.get("error"):
            out[(keyword, field)] = 0
        else:
            out[(keyword, field)] = _total_hits_value(item.get("hits", {}).get("total", 0))
    return out


def query_keywords_counts_batch(es_client, keywords: List[str], paper_year: int, root_fields: List[str],
                                index: str = "acemap.works") -> Dict[str, Dict[str, int]]:
    clean_keywords = [clean_keyword_for_query(keyword) for keyword in keywords]
    requests = []
    for original, clean in zip(keywords, clean_keywords):
        for field in root_fields:
            requests.append((original, field, _count_query(clean, field, paper_year)))
    counts = _safe_es_msearch_counts(es_client, index, requests)
    return {
        keyword: {field: int(counts.get((keyword, field), 0)) for field in root_fields}
        for keyword in keywords
    }

def counts_to_distribution(keyword: str, paper_year: int, root_fields: List[str], counts: Dict[str, int],
                           alpha: float, fallback_level: str, query_mode: str) -> dict:
    total = float(sum(float(counts.get(field, 0.0)) for field in root_fields))
    denom = total + len(root_fields) * float(alpha)
    probs = {
        field: (float(counts.get(field, 0.0)) + float(alpha)) / denom
        for field in root_fields
    }
    prob_sum = sum(probs.values())
    if abs(prob_sum - 1.0) > 1e-10:
        probs = {field: value / prob_sum for field, value in probs.items()}
    top_fields = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "keyword": keyword,
        "year_cutoff": int(paper_year),
        "field_probs": probs,
        "field_counts": {field: int(counts.get(field, 0)) for field in root_fields},
        "match_count": int(total),
        "fallback_level": fallback_level,
        "query_mode": query_mode,
        "top_fields": [[field, float(prob)] for field, prob in top_fields],
    }


def query_keyword_counts(es_client, keyword: str, paper_year: int, root_fields: List[str],
                         index: str = "acemap.works") -> Dict[str, int]:
    query_keyword = clean_keyword_for_query(keyword)
    return {
        field: _safe_es_count(es_client, index, _count_query(query_keyword, field, paper_year))
        for field in root_fields
    }


def get_historical_field_prior(es_client, paper_year: int, root_fields: List[str], alpha: float,
                               index: str = "acemap.works") -> dict:
    counts = {
        field: _safe_es_count(es_client, index, _count_query(None, field, paper_year))
        for field in root_fields
    }
    if sum(counts.values()) <= 0:
        return counts_to_distribution(
            "__uniform_prior__", paper_year, root_fields,
            {field: 0 for field in root_fields}, alpha, "uniform", "historical_prior_failed",
        )
    return counts_to_distribution(
        "__historical_field_prior__", paper_year, root_fields,
        counts, alpha, "historical_prior", "field_prior",
    )


def get_historical_acemap_field_distribution(keyword: str, paper_year: int, root_fields: List[str],
                                             es_client=None, alpha: float = ALPHA_DEFAULT,
                                             cache: Optional[HistoricalQCache] = None,
                                             index: str = "acemap.works") -> dict:
    if not paper_year:
        return counts_to_distribution(
            keyword, 0, root_fields, {field: 0 for field in root_fields},
            alpha, "uniform", "missing_year",
        )
    if cache is not None:
        cached = cache.get(keyword, int(paper_year), root_fields)
        if cached is not None:
            return cached

    counts = query_keyword_counts(es_client, keyword, int(paper_year), root_fields, index=index)
    if sum(counts.values()) > 0:
        out = counts_to_distribution(keyword, paper_year, root_fields, counts, alpha, "none", "exact")
        if cache is not None:
            cache.set(keyword, int(paper_year), root_fields, out)
        return out

    component_rows = []
    for component in phrase_components(keyword):
        if normalize_keyword(component) == normalize_keyword(keyword):
            continue
        comp_counts = query_keyword_counts(es_client, component, int(paper_year), root_fields, index=index)
        match_count = sum(comp_counts.values())
        if match_count > 0:
            component_rows.append((component, match_count, comp_counts))
    if component_rows:
        merged = {field: 0.0 for field in root_fields}
        total_weight = 0.0
        for _component, match_count, comp_counts in component_rows:
            total_weight += float(match_count)
            for field in root_fields:
                merged[field] += float(match_count) * float(comp_counts.get(field, 0.0))
        if total_weight > 0:
            merged = {field: int(round(value / total_weight)) for field, value in merged.items()}
        out = counts_to_distribution(keyword, paper_year, root_fields, merged, alpha, "phrase_components", "exact")
        out["component_keywords"] = [row[0] for row in component_rows]
        out["component_match_count"] = int(sum(row[1] for row in component_rows))
        if cache is not None:
            cache.set(keyword, int(paper_year), root_fields, out)
        return out

    out = get_historical_field_prior(es_client, int(paper_year), root_fields, alpha, index=index)
    out = dict(out)
    out["keyword"] = keyword
    if cache is not None:
        cache.set(keyword, int(paper_year), root_fields, out)
    return out




def get_historical_acemap_field_distributions_batch(keywords: List[str], paper_year: int, root_fields: List[str],
                                                    es_client=None, alpha: float = ALPHA_DEFAULT,
                                                    cache: Optional[HistoricalQCache] = None,
                                                    index: str = "acemap.works") -> List[dict]:
    if not keywords:
        return []
    if not paper_year:
        return [
            counts_to_distribution(keyword, 0, root_fields, {field: 0 for field in root_fields}, alpha, "uniform", "missing_year")
            for keyword in keywords
        ]
    paper_year = int(paper_year)
    out: List[Optional[dict]] = [None] * len(keywords)
    missing_positions = []
    for idx, keyword in enumerate(keywords):
        cached = cache.get(keyword, paper_year, root_fields) if cache is not None else None
        if cached is not None:
            out[idx] = cached
        else:
            missing_positions.append(idx)
    if not missing_positions:
        return [row for row in out if row is not None]

    missing_keywords = [keywords[idx] for idx in missing_positions]
    exact_counts = query_keywords_counts_batch(es_client, missing_keywords, paper_year, root_fields, index=index)
    still_missing = []
    cache_rows = []
    for idx, keyword in zip(missing_positions, missing_keywords):
        counts = exact_counts.get(keyword, {})
        if sum(counts.values()) > 0:
            row = counts_to_distribution(keyword, paper_year, root_fields, counts, alpha, "none", "exact_batch")
            out[idx] = row
            cache_rows.append((keyword, paper_year, row))
        else:
            still_missing.append(idx)

    component_map = {}
    component_to_owner = defaultdict(list)
    for idx in still_missing:
        keyword = keywords[idx]
        comps = [c for c in phrase_components(keyword) if normalize_keyword(c) != normalize_keyword(keyword)]
        component_map[idx] = comps
        for comp in comps:
            component_to_owner[comp].append(idx)
    if component_to_owner:
        component_counts = query_keywords_counts_batch(es_client, list(component_to_owner), paper_year, root_fields, index=index)
    else:
        component_counts = {}

    unresolved = []
    for idx in still_missing:
        keyword = keywords[idx]
        component_rows = []
        for comp in component_map.get(idx, []):
            comp_counts = component_counts.get(comp, {})
            match_count = sum(comp_counts.values())
            if match_count > 0:
                component_rows.append((comp, match_count, comp_counts))
        if component_rows:
            merged = {field: 0.0 for field in root_fields}
            total_weight = 0.0
            for _component, match_count, comp_counts in component_rows:
                total_weight += float(match_count)
                for field in root_fields:
                    merged[field] += float(match_count) * float(comp_counts.get(field, 0.0))
            if total_weight > 0:
                merged = {field: int(round(value / total_weight)) for field, value in merged.items()}
            row = counts_to_distribution(keyword, paper_year, root_fields, merged, alpha, "phrase_components", "component_batch")
            row["component_keywords"] = [item[0] for item in component_rows]
            row["component_match_count"] = int(sum(item[1] for item in component_rows))
            out[idx] = row
            cache_rows.append((keyword, paper_year, row))
        else:
            unresolved.append(idx)

    prior_row = None
    for idx in unresolved:
        if prior_row is None:
            prior_row = get_historical_field_prior(es_client, paper_year, root_fields, alpha, index=index)
        row = dict(prior_row)
        row["keyword"] = keywords[idx]
        out[idx] = row
        cache_rows.append((keywords[idx], paper_year, row))

    if cache is not None:
        cache.set_many(cache_rows, root_fields)
    return [row if row is not None else counts_to_distribution(keywords[i], paper_year, root_fields, {field: 0 for field in root_fields}, alpha, "uniform", "internal_missing") for i, row in enumerate(out)]

def distribution_to_vector(row: dict, root_fields: List[str]) -> np.ndarray:
    probs = row.get("field_probs") or {}
    vec = np.array([float(probs.get(field, 0.0)) for field in root_fields], dtype=np.float64)
    if len(vec) != 26:
        raise ValueError(f"historical Q vector length is {len(vec)}, expected 26")
    total = float(vec.sum())
    if total <= 0:
        vec[:] = 1.0 / len(vec)
    else:
        vec = vec / total
    if np.any(vec < -1e-12):
        raise ValueError("historical Q vector has negative values")
    if abs(float(vec.sum()) - 1.0) > 1e-6:
        raise ValueError(f"historical Q vector sum is {vec.sum()}, expected 1")
    return vec


def lexical_salience(keywords: List[str], doc: dict) -> List[float]:
    title = normalize_keyword(doc.get("title", ""))
    abs_text = normalize_keyword(doc.get("abs", ""))
    full = f"{title} {abs_text}"
    scores = []
    for keyword in keywords:
        kw = normalize_keyword(keyword)
        if not kw:
            scores.append(0.0)
            continue
        phrase_len = max(1, len(kw.split()))
        title_hit = 2.0 if kw in title else 0.0
        freq = full.count(kw)
        scores.append(1.0 + title_hit + float(freq) + 0.15 * phrase_len)
    total = sum(scores)
    if total <= 0:
        return [1.0 / len(keywords)] * len(keywords) if keywords else []
    return [s / total for s in scores]


def normalize_salience(values: List[float], n: int) -> List[float]:
    vals = [max(0.0, float(v)) for v in values[:n]]
    if len(vals) < n:
        vals.extend([0.0] * (n - len(vals)))
    total = sum(vals)
    if total <= 0:
        return [1.0 / n] * n if n else []
    return [v / total for v in vals]


def compute_pair_scores(keywords: List[str], salience: List[float], q_matrix: np.ndarray,
                        d_hist: np.ndarray, d_static: np.ndarray, top_k: int = 20) -> dict:
    if len(keywords) < 2:
        return {
            "historical_pair_distance_score": 0.0,
            "static_2021_pair_distance_score": 0.0,
            "temporal_drift_pair_distance_score": 0.0,
            "top_keyword_pairs": [],
            "top_temporal_drift_keyword_pairs": [],
        }
    b_hist = q_matrix @ d_hist @ q_matrix.T
    b_static = q_matrix @ d_static @ q_matrix.T
    drift = np.maximum(0.0, b_hist - b_static)
    hist_score = 0.0
    static_score = 0.0
    drift_score = 0.0
    pairs = []
    drift_pairs = []
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            w = float(salience[i] * salience[j])
            hist = float(b_hist[i, j])
            static = float(b_static[i, j])
            dr = float(drift[i, j])
            contribution = w * hist
            hist_score += contribution
            static_score += w * static
            drift_score += w * dr
            row = {
                "keyword_i": keywords[i],
                "keyword_j": keywords[j],
                "salience_i": float(salience[i]),
                "salience_j": float(salience[j]),
                "historical_distance": hist,
                "static_2021_distance": static,
                "temporal_drift": dr,
                "weighted_contribution": contribution,
            }
            pairs.append(row)
            if dr > 0:
                drift_pairs.append(dict(row))
    pairs.sort(key=lambda row: row["weighted_contribution"], reverse=True)
    drift_pairs.sort(key=lambda row: row["temporal_drift"] * row["salience_i"] * row["salience_j"], reverse=True)
    return {
        "historical_pair_distance_score": float(hist_score),
        "static_2021_pair_distance_score": float(static_score),
        "temporal_drift_pair_distance_score": float(drift_score),
        "top_keyword_pairs": pairs[:top_k],
        "top_temporal_drift_keyword_pairs": drift_pairs[:top_k],
    }


def softmax_attention_bias(salience: List[float], b_matrix: np.ndarray, lambda_b: float = 1.0,
                           temporal_penalty: Optional[List[float]] = None,
                           gamma: float = 0.0) -> np.ndarray:
    """Keyword-level analogue of softmax(attention_score + lambda * B_t a - gamma * q)."""
    a = np.asarray(salience, dtype=float)
    if a.size == 0:
        return a
    a = np.maximum(a, 0.0)
    total = float(a.sum())
    if total <= 0:
        a = np.ones_like(a) / float(a.size)
    else:
        a = a / total
    if b_matrix.shape != (a.size, a.size):
        return a
    bias = b_matrix @ a
    penalty = np.zeros_like(a)
    if temporal_penalty is not None:
        penalty = np.asarray(temporal_penalty, dtype=float)
        if penalty.shape != a.shape:
            penalty = np.zeros_like(a)
        penalty = np.clip(penalty, 0.0, 1.0)
    logits = a + float(lambda_b) * bias - float(gamma) * penalty
    logits = logits - float(np.max(logits))
    out = np.exp(logits)
    denom = float(out.sum())
    if denom <= 0:
        return a
    return out / denom


def compute_entropy_vs_from_subject_weights(subject_weights: np.ndarray,
                                            distance_matrix: np.ndarray,
                                            threshold: float = 0.321) -> Tuple[Dict[str, float], float, int]:
    weights = np.asarray(subject_weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if weights.size == 0 or total <= 0:
        return {}, 0.0, -1
    weights = weights / total
    active = np.where(weights > 1e-12)[0]
    if active.size == 0:
        return {}, 0.0, -1
    anchor = int(active[np.argmax(weights[active])])
    adjusted = weights.copy()
    for idx in active:
        dis = float(distance_matrix[anchor, idx])
        if dis > threshold:
            adjusted[idx] = weights[idx] * (1.0 + dis - threshold)
        else:
            adjusted[idx] = weights[idx] * (1.0 - dis + threshold)
    adjusted = np.maximum(adjusted, 0.0)
    adjusted_total = float(adjusted.sum())
    if adjusted_total <= 0:
        return {}, 0.0, anchor
    adjusted = adjusted / adjusted_total
    active_adjusted = adjusted[adjusted > 1e-12]
    if active_adjusted.size <= 1:
        vs = 0.0
    else:
        entropy = -float(np.sum(active_adjusted * np.log(np.maximum(active_adjusted, 1e-12))))
        vs = float(entropy / math.log(active_adjusted.size))
    return {str(i): float(adjusted[i]) for i in range(adjusted.size) if adjusted[i] > 1e-12}, vs, anchor


def compute_framework_scores(salience: List[float], quantiles: List[float], q_matrix: np.ndarray,
                             d_matrix: np.ndarray, lambda_b: float, threshold: float,
                             temporal_penalty_gamma: float = 0.0,
                             vt_attention_scale: float = 1.0) -> dict:
    b_matrix = q_matrix @ d_matrix @ q_matrix.T
    base_a = np.asarray(normalize_salience(salience, len(salience)), dtype=float)
    quant = np.asarray(quantiles, dtype=float)
    if quant.size != base_a.size:
        quant_for_penalty = np.zeros_like(base_a)
    else:
        quant_for_penalty = np.clip(quant, 0.0, 1.0)
    modulated = softmax_attention_bias(
        base_a.tolist(),
        b_matrix,
        lambda_b=lambda_b,
        temporal_penalty=quant_for_penalty.tolist(),
        gamma=temporal_penalty_gamma,
    )
    if quant.size != modulated.size:
        quant = np.zeros_like(modulated)
    vt_penalty = float(vt_attention_scale) * float(np.sum(modulated * quant))
    vt = 1.0 - vt_penalty
    vt = max(0.0, min(1.0, vt))
    subject_weights = modulated @ q_matrix
    adjusted, vs, anchor = compute_entropy_vs_from_subject_weights(
        subject_weights,
        d_matrix,
        threshold=threshold,
    )
    return {
        "attention": [float(x) for x in modulated.tolist()],
        "attention_bias": [float(x) for x in (b_matrix @ base_a).tolist()],
        "temporal_penalty": [float(x) for x in quant_for_penalty.tolist()],
        "attention_logits_adjustment": [float(x) for x in (float(lambda_b) * (b_matrix @ base_a) - float(temporal_penalty_gamma) * quant_for_penalty).tolist()],
        "subject_weights": {str(i): float(subject_weights[i]) for i in range(subject_weights.size) if subject_weights[i] > 1e-12},
        "adjusted_subject_weights": adjusted,
        "anchor_subject_index": int(anchor) if anchor >= 0 else None,
        "Vs": float(vs),
        "Vt": float(vt),
        "vt_attention_scale": float(vt_attention_scale),
        "vt_penalty": float(vt_penalty),
        "xindex": float(vs * vt),
    }



def build_identity_plus_offdiag_matrix(subject_matrix: np.ndarray, offdiag_lambda: float = 1.0) -> np.ndarray:
    """Build M = I + lambda * Norm(S - Diag(diag(S))).

    The normalization is max-absolute scaling over off-diagonal entries. This
    removes potentially over-strong self-relations from the observed subject
    matrix, preserves a stable identity self-channel, and uses only cross-subject
    relations as a bounded perturbation.
    """
    s = np.asarray(subject_matrix, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        return s
    off = s.copy()
    np.fill_diagonal(off, 0.0)
    off = np.maximum(off, 0.0)
    max_val = float(np.max(np.abs(off))) if off.size else 0.0
    if max_val > 0:
        norm_off = off / max_val
    else:
        norm_off = np.zeros_like(off)
    ident = np.eye(s.shape[0], dtype=float)
    return ident + float(offdiag_lambda) * norm_off

def compute_framework_scores_hard_roots(salience: List[float], quantiles: List[float], q_matrix: np.ndarray,
                                        d_matrix: np.ndarray, subject_labels: List[str],
                                        lambda_b: float, threshold: float,
                                        temporal_penalty_gamma: float = 0.0,
                                        vt_attention_scale: float = 1.0) -> dict:
    b_matrix = q_matrix @ d_matrix @ q_matrix.T
    base_a = np.asarray(normalize_salience(salience, len(salience)), dtype=float)
    quant = np.asarray(quantiles, dtype=float)
    if quant.size != base_a.size:
        quant_for_penalty = np.zeros_like(base_a)
    else:
        quant_for_penalty = np.clip(quant, 0.0, 1.0)
    modulated = softmax_attention_bias(
        base_a.tolist(),
        b_matrix,
        lambda_b=lambda_b,
        temporal_penalty=quant_for_penalty.tolist(),
        gamma=temporal_penalty_gamma,
    )
    if quant.size != modulated.size:
        quant = np.zeros_like(modulated)
    vt_penalty = float(vt_attention_scale) * float(np.sum(modulated * quant))
    vt = 1.0 - vt_penalty
    vt = max(0.0, min(1.0, vt))

    labels = [str(x or "").strip() for x in subject_labels[:len(modulated)]]
    if len(labels) < len(modulated):
        labels.extend([""] * (len(modulated) - len(labels)))
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label:
            groups[label].append(idx)
    if not groups:
        return {
            "attention": [float(x) for x in modulated.tolist()],
            "attention_bias": [float(x) for x in (b_matrix @ base_a).tolist()],
            "temporal_penalty": [float(x) for x in quant_for_penalty.tolist()],
            "attention_logits_adjustment": [float(x) for x in (float(lambda_b) * (b_matrix @ base_a) - float(temporal_penalty_gamma) * quant_for_penalty).tolist()],
            "subject_weights": {},
            "adjusted_subject_weights": {},
            "anchor_subject": None,
            "subject_distance_from_anchor": {},
            "root_count": 0,
            "nonzero_root_count": 0,
            "xindex_zero_reason": "no_valid_root",
            "Vs": 0.0,
            "Vt": float(vt),
            "vt_attention_scale": float(vt_attention_scale),
            "vt_penalty": float(vt_penalty),
            "xindex": 0.0,
        }

    raw_weights = {label: float(np.sum(modulated[idxs])) for label, idxs in groups.items()}
    total_subject = sum(raw_weights.values())
    if total_subject <= 0:
        norm_weights = {label: 1.0 / len(raw_weights) for label in raw_weights}
    else:
        norm_weights = {label: weight / total_subject for label, weight in raw_weights.items()}
    anchor = max(norm_weights, key=lambda label: norm_weights[label])

    def grouped_b_distance(label_a: str, label_b: str) -> float:
        if label_a == label_b:
            return 0.0
        ia = groups[label_a]
        ib = groups[label_b]
        wa = modulated[ia]
        wb = modulated[ib]
        denom = float(np.sum(wa) * np.sum(wb))
        if denom <= 0:
            return 0.0
        sub = b_matrix[np.ix_(ia, ib)]
        return float(np.sum(np.outer(wa, wb) * sub) / denom)

    adjusted = {}
    subject_distance = {}
    for subject, weight in norm_weights.items():
        dis = grouped_b_distance(anchor, subject)
        subject_distance[subject] = dis
        if dis > threshold:
            adjusted[subject] = weight * (1.0 + dis - threshold)
        else:
            adjusted[subject] = weight * (1.0 - dis + threshold)
    total_adj = sum(max(0.0, v) for v in adjusted.values())
    if total_adj <= 0:
        adjusted = {label: 1.0 / len(adjusted) for label in adjusted}
    else:
        adjusted = {label: max(0.0, value) / total_adj for label, value in adjusted.items()}
    vals = [v for v in adjusted.values() if v > 1e-12]
    nonzero_root_count = len(vals)
    if nonzero_root_count <= 1:
        vs = 0.0
        xindex_zero_reason = "single_valid_root"
    else:
        vs = -sum(v * math.log(max(v, 1e-12)) for v in vals) / math.log(nonzero_root_count)
        xindex_zero_reason = ""
    xindex = float(vs * vt) if nonzero_root_count > 1 and vs > 0 else 0.0
    return {
        "attention": [float(x) for x in modulated.tolist()],
        "attention_bias": [float(x) for x in (b_matrix @ base_a).tolist()],
        "temporal_penalty": [float(x) for x in quant_for_penalty.tolist()],
        "attention_logits_adjustment": [float(x) for x in (float(lambda_b) * (b_matrix @ base_a) - float(temporal_penalty_gamma) * quant_for_penalty).tolist()],
        "subject_weights": {str(k): float(v) for k, v in norm_weights.items()},
        "adjusted_subject_weights": {str(k): float(v) for k, v in adjusted.items()},
        "anchor_subject": str(anchor),
        "subject_distance_from_anchor": {str(k): float(v) for k, v in subject_distance.items()},
        "root_count": int(len(adjusted)),
        "nonzero_root_count": int(nonzero_root_count),
        "xindex_zero_reason": xindex_zero_reason,
        "Vs": float(vs),
        "Vt": float(vt),
        "vt_attention_scale": float(vt_attention_scale),
        "vt_penalty": float(vt_penalty),
        "xindex": float(xindex),
    }

def compute_historical_temporal_xindex(doc: dict, details: List[dict], prompt_keywords: List[str],
                                       es_client, args, cache: Optional[HistoricalQCache] = None,
                                       distance_loader: Optional[TemporalDistanceLoader] = None) -> dict:
    if distance_loader is None:
        distance_loader = TemporalDistanceLoader(args.distance_resource_dir)
    root_fields = distance_loader.root_fields()
    paper_year = doc.get("year")
    if not paper_year:
        return {
            "method_profile": args.xindex_profile,
            "keyword_field_mode": args.keyword_field_mode,
            "distance_mode": args.distance_mode,
            "historical_q_status": "missing_year_fallback",
        }
    paper_year = int(paper_year)
    d_hist_raw, d_meta = distance_loader.load(paper_year - 1)
    d_static_raw, static_meta = distance_loader.load(STATIC_DISTANCE_YEAR)
    subject_matrix_lambda = float(getattr(args, "subject_matrix_lambda", 1.0))
    d_hist = build_identity_plus_offdiag_matrix(d_hist_raw, subject_matrix_lambda)
    d_static = build_identity_plus_offdiag_matrix(d_static_raw, subject_matrix_lambda)

    detail_by_name = {}
    for row in details or []:
        name = str(row.get("keyword_name") or "").strip()
        if name and name not in detail_by_name:
            detail_by_name[name] = row
    keywords = [kw for kw in detail_by_name if kw]
    if not keywords:
        keywords = [str(kw).strip() for kw in prompt_keywords or [] if str(kw).strip()]
    keywords = list(dict.fromkeys(keywords))

    if not keywords:
        return {
            "method_profile": args.xindex_profile,
            "keyword_field_mode": args.keyword_field_mode,
            "distance_mode": args.distance_mode,
            **d_meta,
            "static_distance_year": int(static_meta["distance_year_used"]),
            "historical_q_status": "empty_keywords",
            "historical_formula_version": "identity_offdiag_subject_matrix_b_modulated_temporal_penalty_final",
            "historical_attention_lambda": float(getattr(args, "historical_attention_lambda", 1.0)),
            "historical_temporal_penalty_gamma": float(getattr(args, "historical_temporal_penalty_gamma", 0.0)),
            "subject_matrix_mode": "identity_plus_maxnorm_offdiag",
            "subject_matrix_lambda": float(getattr(args, "subject_matrix_lambda", 1.0)),
            "historical_root_filter_mode": str(getattr(args, "historical_root_filter_mode", "baseline_compatible")),
            "historical_Vs": 0.0,
            "historical_Vt": 0.0,
            "static_2021_Vs": 0.0,
            "static_2021_Vt": 0.0,
            "xindex_historical": 0.0,
            "xindex_static_2021": 0.0,
            "xindex_temporal_drift": 0.0,
            "historical_pair_distance_score": 0.0,
            "static_2021_pair_distance_score": 0.0,
            "temporal_drift_pair_distance_score": 0.0,
            "keyword_field_distributions": [],
            "top_keyword_pairs": [],
            "top_temporal_drift_keyword_pairs": [],
            "historical_q_audit": {
                "keyword_count": 0,
                "direct_match_keywords": 0,
                "fallback_keywords": 0,
                "uniform_keywords": 0,
                "mean_match_count": 0.0,
            },
        }

    q_rows = get_historical_acemap_field_distributions_batch(
        keywords,
        paper_year,
        root_fields,
        es_client=es_client,
        alpha=float(args.historical_q_alpha),
        cache=cache,
        index=args.es_index,
    )

    # Paper-context fallback for keywords that reached a broad prior while other
    # keywords in the same paper have direct/component evidence.
    evidence_rows = [row for row in q_rows if row.get("fallback_level") in {"none", "phrase_components"}]
    if evidence_rows:
        weights = [max(1.0, float(row.get("match_count") or 0.0)) for row in evidence_rows]
        total_w = sum(weights)
        context_probs = {
            field: sum(w * float(row["field_probs"].get(field, 0.0)) for row, w in zip(evidence_rows, weights)) / total_w
            for field in root_fields
        }
        for idx, row in enumerate(q_rows):
            if row.get("fallback_level") == "historical_prior":
                q_rows[idx] = {
                    **row,
                    "field_probs": context_probs,
                    "fallback_level": "paper_context",
                    "query_mode": "paper_context_average",
                    "top_fields": [[f, float(p)] for f, p in sorted(context_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]],
                }

    q_matrix = np.vstack([distribution_to_vector(row, root_fields) for row in q_rows])

    if args.keyword_salience_mode == "existing":
        salience = normalize_salience(
            [float(detail_by_name.get(keyword, {}).get("a_hat") or 0.0) for keyword in keywords],
            len(keywords),
        )
        mode_used = "existing"
    elif args.keyword_salience_mode == "lexical":
        salience = lexical_salience(keywords, doc)
        mode_used = "lexical"
    else:
        salience = [1.0 / len(keywords)] * len(keywords)
        mode_used = "uniform"

    quantiles = [float(detail_by_name.get(keyword, {}).get("keyword_quantile") or 0.0) for keyword in keywords]
    lambda_b = float(getattr(args, "historical_attention_lambda", 1.0))
    temporal_penalty_gamma = float(getattr(args, "historical_temporal_penalty_gamma", 0.0))
    vt_attention_scale = float(getattr(args, "historical_vt_attention_scale", 1.0))
    threshold = float(getattr(args, "threshold_subjects", 0.321))
    root_filter_mode = str(getattr(args, "historical_root_filter_mode", "baseline_compatible") or "baseline_compatible")
    root_labels = []
    pre_roots = []
    dropped_by_revised_gate_count = 0
    dropped_by_quality_gate_count = 0
    dropped_by_root_cap_count = 0
    for keyword in keywords:
        detail = detail_by_name.get(keyword, {})
        root = str(detail.get("final_root") or "").strip()
        quality_or_cap_drop = bool(detail.get("dropped_by_quality_gate")) or bool(detail.get("dropped_by_root_cap"))
        if detail.get("dropped_by_quality_gate"):
            dropped_by_quality_gate_count += 1
        if detail.get("dropped_by_root_cap"):
            dropped_by_root_cap_count += 1
        if quality_or_cap_drop:
            root = ""
        elif detail.get("dropped_by_root_gate"):
            dropped_by_revised_gate_count += 1
            if root_filter_mode == "revised_v5_strict":
                root = ""
        root_labels.append(root)
        if root:
            pre_roots.append(root)
    root_count_before_singleton_gate = len(set(pre_roots))
    singleton_roots = {root for root in pre_roots if pre_roots.count(root) == 1}
    if root_filter_mode == "revised_v5_strict":
        root_labels = ["" if root in singleton_roots else root for root in root_labels]
    root_count_after_singleton_gate = len(set(root for root in root_labels if root))
    hist_framework = compute_framework_scores_hard_roots(
        salience, quantiles, q_matrix, d_hist, root_labels, lambda_b, threshold,
        temporal_penalty_gamma=temporal_penalty_gamma,
        vt_attention_scale=vt_attention_scale,
    )
    static_framework = compute_framework_scores_hard_roots(
        salience, quantiles, q_matrix, d_static, root_labels, lambda_b, threshold,
        temporal_penalty_gamma=temporal_penalty_gamma,
        vt_attention_scale=vt_attention_scale,
    )
    scores = compute_pair_scores(keywords, salience, q_matrix, d_hist, d_static)
    fallback_count = sum(1 for row in q_rows if row.get("fallback_level") != "none")
    uniform_count = sum(1 for row in q_rows if row.get("fallback_level") == "uniform")
    match_counts = [float(row.get("match_count") or 0.0) for row in q_rows]
    bad_prob = 0
    for row in q_rows:
        vals = [float(row.get("field_probs", {}).get(field, 0.0)) for field in root_fields]
        if len(vals) != 26 or abs(sum(vals) - 1.0) > 1e-6 or any(v < 0 for v in vals):
            bad_prob += 1

    return {
        "method_profile": args.xindex_profile,
        "keyword_field_mode": args.keyword_field_mode,
        "distance_mode": args.distance_mode,
        "distance_resource_dir": str(distance_loader.distance_dir),
        **d_meta,
        "static_distance_year": int(static_meta["distance_year_used"]),
        "historical_formula_version": "identity_offdiag_subject_matrix_b_modulated_temporal_penalty_final",
        "historical_attention_lambda": lambda_b,
        "historical_temporal_penalty_gamma": temporal_penalty_gamma,
        "historical_vt_attention_scale": vt_attention_scale,
        "subject_matrix_mode": "identity_plus_maxnorm_offdiag",
        "subject_matrix_lambda": subject_matrix_lambda,
        "historical_root_filter_mode": root_filter_mode,
        "historical_dropped_by_revised_gate_count": int(dropped_by_revised_gate_count),
        "historical_dropped_by_quality_gate_count": int(dropped_by_quality_gate_count),
        "historical_dropped_by_root_cap_count": int(dropped_by_root_cap_count),
        "historical_singleton_root_count_before_filter": int(len(singleton_roots)),
        "historical_Vs": hist_framework["Vs"],
        "historical_Vt": hist_framework["Vt"],
        "historical_vt_penalty": hist_framework.get("vt_penalty", 0.0),
        "static_2021_Vs": static_framework["Vs"],
        "static_2021_Vt": static_framework["Vt"],
        "static_2021_vt_penalty": static_framework.get("vt_penalty", 0.0),
        "historical_root_count_before_singleton_gate": int(root_count_before_singleton_gate),
        "historical_root_count_after_singleton_gate": int(root_count_after_singleton_gate),
        "historical_nonzero_root_count": hist_framework.get("nonzero_root_count", 0),
        "static_2021_nonzero_root_count": static_framework.get("nonzero_root_count", 0),
        "historical_xindex_zero_reason": hist_framework.get("xindex_zero_reason", ""),
        "static_2021_xindex_zero_reason": static_framework.get("xindex_zero_reason", ""),
        "xindex_historical": hist_framework["xindex"],
        "xindex_static_2021": static_framework["xindex"],
        "xindex_temporal_drift": float(hist_framework["xindex"] - static_framework["xindex"]),
        "historical_pair_distance_score": scores["historical_pair_distance_score"],
        "static_2021_pair_distance_score": scores["static_2021_pair_distance_score"],
        "temporal_drift_pair_distance_score": scores["temporal_drift_pair_distance_score"],
        "historical_distance_score": scores["historical_pair_distance_score"],
        "static_2021_distance_score": scores["static_2021_pair_distance_score"],
        "temporal_drift_score": scores["temporal_drift_pair_distance_score"],
        "keyword_salience_mode": mode_used,
        "historical_modulated_attention": hist_framework["attention"],
        "static_2021_modulated_attention": static_framework["attention"],
        "historical_attention_bias": hist_framework["attention_bias"],
        "static_2021_attention_bias": static_framework["attention_bias"],
        "historical_temporal_penalty": hist_framework.get("temporal_penalty", []),
        "static_2021_temporal_penalty": static_framework.get("temporal_penalty", []),
        "historical_attention_logits_adjustment": hist_framework.get("attention_logits_adjustment", []),
        "static_2021_attention_logits_adjustment": static_framework.get("attention_logits_adjustment", []),
        "historical_subject_weights": hist_framework["subject_weights"],
        "static_2021_subject_weights": static_framework["subject_weights"],
        "historical_adjusted_subject_weights": hist_framework["adjusted_subject_weights"],
        "static_2021_adjusted_subject_weights": static_framework["adjusted_subject_weights"],
        "historical_anchor_subject": hist_framework["anchor_subject"],
        "static_2021_anchor_subject": static_framework["anchor_subject"],
        "historical_subject_distance_from_anchor": hist_framework["subject_distance_from_anchor"],
        "static_2021_subject_distance_from_anchor": static_framework["subject_distance_from_anchor"],
        "keyword_field_distributions": q_rows,
        "top_keyword_pairs": scores["top_keyword_pairs"],
        "top_temporal_drift_keyword_pairs": scores["top_temporal_drift_keyword_pairs"],
        "historical_q_status": "ok",
        "historical_q_audit": {
            "keyword_count": len(q_rows),
            "direct_match_keywords": sum(1 for row in q_rows if row.get("fallback_level") == "none"),
            "fallback_keywords": fallback_count,
            "uniform_keywords": uniform_count,
            "mean_match_count": float(sum(match_counts) / len(match_counts)) if match_counts else 0.0,
            "bad_probability_count": bad_prob,
        },
    }
