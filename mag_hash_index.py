
#!/usr/bin/env python3
"""Level-aware Random Hyperplane LSH index for MAG concept grounding."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def l2n_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-10
    return a / n


def hash_embeddings(embs: np.ndarray, hyperplanes: np.ndarray) -> np.ndarray:
    bits = (np.asarray(embs, dtype=np.float32) @ np.asarray(hyperplanes, dtype=np.float32)) >= 0
    codes = np.zeros(bits.shape[0], dtype=np.uint64)
    for bit_idx in range(bits.shape[1]):
        codes |= bits[:, bit_idx].astype(np.uint64) << np.uint64(bit_idx)
    return codes


def build_level_arrays(ids: np.ndarray, names: List[str], embeddings: np.ndarray,
                       hyperplanes: np.ndarray) -> dict:
    embeddings = l2n_rows(embeddings)
    codes = hash_embeddings(embeddings, hyperplanes)
    order = np.lexsort((ids, codes))
    ids = ids[order].astype(np.int64, copy=False)
    embeddings = embeddings[order].astype(np.float32, copy=False)
    codes = codes[order].astype(np.uint64, copy=False)
    names = [names[int(i)] for i in order]

    unique_codes, starts, counts = np.unique(codes, return_index=True, return_counts=True)
    centroids = []
    for start, count in zip(starts, counts):
        centroids.append(l2n_rows(embeddings[start:start + count].mean(axis=0))[0])
    centroids = np.asarray(centroids, dtype=np.float32)
    return {
        "ids": ids,
        "names": names,
        "embeddings": embeddings,
        "codes": codes,
        "unique_codes": unique_codes.astype(np.uint64),
        "starts": starts.astype(np.int64),
        "counts": counts.astype(np.int64),
        "centroids": centroids,
    }


class MAGHashIndex:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        meta_path = self.index_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"MAG hash index meta not found: {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.levels = [int(x) for x in self.meta.get("levels", [])]
        self.hash_bits = int(self.meta.get("hash_bits", 14))
        self.level_data: Dict[int, dict] = {}
        for level in self.levels:
            self.level_data[level] = self._load_level(level)

    def _load_level(self, level: int) -> dict:
        npz_path = self.index_dir / f"level_{level}.npz"
        nodes_path = self.index_dir / f"level_{level}_nodes.jsonl"
        if not npz_path.exists() or not nodes_path.exists():
            raise FileNotFoundError(f"MAG hash index level files missing for level {level}")
        materialized_keys = ["ids", "embeddings", "codes", "unique_codes", "starts", "counts", "centroids"]
        materialized_paths = {key: self.index_dir / f"level_{level}_{key}.npy" for key in materialized_keys}
        if all(path.exists() for path in materialized_paths.values()):
            arr = {
                "ids": np.load(materialized_paths["ids"], mmap_mode="r"),
                "embeddings": np.load(materialized_paths["embeddings"], mmap_mode="r"),
                "codes": np.load(materialized_paths["codes"], mmap_mode="r"),
                "unique_codes": np.load(materialized_paths["unique_codes"], mmap_mode="r"),
                "starts": np.load(materialized_paths["starts"], mmap_mode="r"),
                "counts": np.load(materialized_paths["counts"], mmap_mode="r"),
                "centroids": np.load(materialized_paths["centroids"], mmap_mode="r"),
            }
        else:
            npz = np.load(npz_path)
            arr = {key: npz[key] for key in materialized_keys}
            for key, value in arr.items():
                try:
                    np.save(materialized_paths[key], value)
                except Exception:
                    pass
        ids = np.asarray(arr["ids"], dtype=np.int64)
        names_cache_path = self.index_dir / f"level_{level}_names.npy"
        if names_cache_path.exists():
            names = np.load(names_cache_path, allow_pickle=True).tolist()
            if len(names) != len(ids):
                names_cache_path.unlink(missing_ok=True)
                names = None
        else:
            names = None
        if names is None:
            id_to_name = {}
            with nodes_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    fid = int(row["field_id"])
                    name = str(row.get("name") or "")
                    id_to_name[fid] = name
            names = [id_to_name.get(int(fid), "") for fid in ids.tolist()]
            try:
                np.save(names_cache_path, np.asarray(names, dtype=object), allow_pickle=True)
            except Exception:
                pass
        code_to_slice = {}
        unique_codes = np.asarray(arr["unique_codes"], dtype=np.uint64)
        starts = np.asarray(arr["starts"], dtype=np.int64)
        counts = np.asarray(arr["counts"], dtype=np.int64)
        for code, start, count in zip(unique_codes, starts, counts):
            code_to_slice[int(code)] = slice(int(start), int(start + count))
        return {
            "ids": ids,
            "names": names,
            "embeddings": np.asarray(arr["embeddings"], dtype=np.float32),
            "codes": np.asarray(arr["codes"], dtype=np.uint64),
            "unique_codes": np.ascontiguousarray(unique_codes),
            "centroids": np.asarray(arr["centroids"], dtype=np.float32),
            "starts": starts,
            "counts": counts,
            "code_to_slice": code_to_slice,
        }


    def query_batch(self, query_vecs: np.ndarray, top_clusters_per_level: int = 8,
                    target_level_top_clusters: int = 12,
                    max_candidate_nodes: int = 512) -> List[dict]:
        qs = np.asarray(query_vecs, dtype=np.float32)
        if qs.ndim == 1:
            qs = qs.reshape(1, -1)
        qs = l2n_rows(qs)
        n_query = int(qs.shape[0])
        if n_query == 0:
            return []
        level_votes_by_query = [[] for _ in range(n_query)]
        level_top_by_query = [dict() for _ in range(n_query)]
        for level in self.levels:
            data = self.level_data[level]
            centroids = data["centroids"]
            if centroids.size == 0:
                continue
            score_mat = qs @ centroids.T
            k = max(1, min(int(top_clusters_per_level), score_mat.shape[1]))
            if score_mat.shape[1] <= k:
                top_idx_mat = np.argsort(score_mat, axis=1)[:, ::-1]
            else:
                top_idx_mat = np.argpartition(score_mat, -k, axis=1)[:, -k:]
                row_order = np.argsort(np.take_along_axis(score_mat, top_idx_mat, axis=1), axis=1)[:, ::-1]
                top_idx_mat = np.take_along_axis(top_idx_mat, row_order, axis=1)
            top_scores_mat = np.take_along_axis(score_mat, top_idx_mat, axis=1)
            level_scores = top_scores_mat.mean(axis=1)
            for qi in range(n_query):
                level_votes_by_query[qi].append({"level": int(level), "score": float(level_scores[qi])})
                level_top_by_query[qi][int(level)] = top_idx_mat[qi]

        out = []
        for qi in range(n_query):
            votes = level_votes_by_query[qi]
            if not votes:
                out.append({"target_level": None, "level_votes": [], "candidates": []})
                continue
            target_level = max(votes, key=lambda row: (row["score"], -row["level"]))["level"]
            data = self.level_data[int(target_level)]
            cluster_idx = np.asarray(level_top_by_query[qi][int(target_level)], dtype=np.int64)
            q = qs[qi]
            cluster_scores = data["centroids"][cluster_idx] @ q
            order = np.argsort(cluster_scores)[::-1][:max(1, int(target_level_top_clusters))]
            chosen_cluster_idx = cluster_idx[order]
            chosen_cluster_scores = cluster_scores[order]
            node_indices = []
            cluster_score_by_code = {}
            for cidx, cscore in zip(chosen_cluster_idx, chosen_cluster_scores):
                code = int(data["unique_codes"][int(cidx)])
                cluster_score_by_code[code] = float(cscore)
                sl = data["code_to_slice"].get(code)
                if sl is not None:
                    node_indices.extend(range(sl.start, sl.stop))
            if not node_indices:
                out.append({"target_level": int(target_level), "level_votes": votes, "candidates": []})
                continue
            node_indices = np.asarray(sorted(set(node_indices)), dtype=np.int64)
            node_embs = data["embeddings"][node_indices]
            node_scores = node_embs @ q
            max_nodes = max(1, int(max_candidate_nodes))
            if len(node_scores) > max_nodes:
                keep = np.argpartition(node_scores, -max_nodes)[-max_nodes:]
                keep = keep[np.argsort(node_scores[keep])[::-1]]
                node_indices = node_indices[keep]
                node_scores = node_scores[keep]
            else:
                keep = np.argsort(node_scores)[::-1]
                node_indices = node_indices[keep]
                node_scores = node_scores[keep]
            candidates = []
            for idx, score in zip(node_indices.tolist(), node_scores.tolist()):
                fid = int(data["ids"][idx])
                code = int(data["codes"][idx])
                candidates.append({
                    "field_id": fid,
                    "mag_name": data["names"][idx],
                    "level": int(target_level),
                    "score": float(score),
                    "hash_code": code,
                    "cluster_score": float(cluster_score_by_code.get(code, 0.0)),
                })
            out.append({"target_level": int(target_level), "level_votes": votes, "candidates": candidates})
        return out

    def query(self, query_vec: np.ndarray, top_clusters_per_level: int = 8,
              target_level_top_clusters: int = 12,
              max_candidate_nodes: int = 512) -> dict:
        return self.query_batch(
            np.asarray(query_vec, dtype=np.float32).reshape(1, -1),
            top_clusters_per_level=top_clusters_per_level,
            target_level_top_clusters=target_level_top_clusters,
            max_candidate_nodes=max_candidate_nodes,
        )[0]

    def query_legacy(self, query_vec: np.ndarray, top_clusters_per_level: int = 8,
              target_level_top_clusters: int = 12,
              max_candidate_nodes: int = 512) -> dict:
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q = q / (float(np.linalg.norm(q)) + 1e-10)
        level_votes = []
        level_top = {}
        for level in self.levels:
            data = self.level_data[level]
            centroids = data["centroids"]
            if centroids.size == 0:
                continue
            scores = centroids @ q
            k = max(1, min(int(top_clusters_per_level), len(scores)))
            if len(scores) <= k:
                idx = np.argsort(scores)[::-1]
            else:
                idx = np.argpartition(scores, -k)[-k:]
                idx = idx[np.argsort(scores[idx])[::-1]]
            top_scores = scores[idx]
            level_score = float(np.mean(top_scores)) if top_scores.size else -1.0
            level_votes.append({"level": int(level), "score": level_score})
            level_top[int(level)] = idx
        if not level_votes:
            return {"target_level": None, "level_votes": [], "candidates": []}
        target_level = max(level_votes, key=lambda row: (row["score"], -row["level"]))["level"]
        data = self.level_data[int(target_level)]
        cluster_idx = level_top[int(target_level)]
        cluster_scores = data["centroids"][cluster_idx] @ q
        order = np.argsort(cluster_scores)[::-1][:max(1, int(target_level_top_clusters))]
        chosen_cluster_idx = cluster_idx[order]
        chosen_cluster_scores = cluster_scores[order]
        node_indices = []
        cluster_score_by_code = {}
        for cidx, cscore in zip(chosen_cluster_idx, chosen_cluster_scores):
            code = int(data["unique_codes"][int(cidx)])
            cluster_score_by_code[code] = float(cscore)
            sl = data["code_to_slice"].get(code)
            if sl is not None:
                node_indices.extend(range(sl.start, sl.stop))
        if not node_indices:
            return {"target_level": int(target_level), "level_votes": level_votes, "candidates": []}
        node_indices = np.asarray(sorted(set(node_indices)), dtype=np.int64)
        node_embs = data["embeddings"][node_indices]
        node_scores = node_embs @ q
        max_nodes = max(1, int(max_candidate_nodes))
        if len(node_scores) > max_nodes:
            keep = np.argpartition(node_scores, -max_nodes)[-max_nodes:]
            keep = keep[np.argsort(node_scores[keep])[::-1]]
            node_indices = node_indices[keep]
            node_scores = node_scores[keep]
        else:
            keep = np.argsort(node_scores)[::-1]
            node_indices = node_indices[keep]
            node_scores = node_scores[keep]
        candidates = []
        for idx, score in zip(node_indices.tolist(), node_scores.tolist()):
            fid = int(data["ids"][idx])
            code = int(data["codes"][idx])
            candidates.append({
                "field_id": fid,
                "mag_name": data["names"][idx],
                "level": int(target_level),
                "score": float(score),
                "hash_code": code,
                "cluster_score": float(cluster_score_by_code.get(code, 0.0)),
            })
        return {"target_level": int(target_level), "level_votes": level_votes, "candidates": candidates}
