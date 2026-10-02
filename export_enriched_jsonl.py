#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl_map(path: Path, key: str = "doc_id"):
    out = {}
    if not path.exists():
        return out
    for row in iter_jsonl(path):
        value = row.get(key)
        if value is not None:
            out[str(value)] = row
    return out


def simplify_authors(authors, author_id_mode: str = "sequence"):
    if not isinstance(authors, list):
        return []
    simplified = []
    for idx, author in enumerate(authors, start=1):
        if not isinstance(author, dict):
            continue
        affiliations = []
        for aff in author.get("affiliations", []) or []:
            if isinstance(aff, str):
                text = aff.strip()
                if text:
                    affiliations.append(text)
                continue
            if not isinstance(aff, dict):
                continue
            org = str(aff.get("organization") or "").strip()
            unit = str(aff.get("unit") or "").strip()
            country = str(aff.get("country") or "").strip()
            parts = [x for x in [org, unit, country] if x]
            if parts:
                affiliations.append(" | ".join(parts))
        dedup_affiliations = list(dict.fromkeys(affiliations))
        author_id = author.get("id")
        if author_id is None and author_id_mode == "sequence":
            author_id = idx
        simplified.append({
            "id": author_id,
            "name": author.get("name"),
            "country": author.get("country"),
            "affiliations": dedup_affiliations,
        })
    return simplified


def count_citations(value):
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return 1 if value else 0
    text = str(value).strip()
    if not text:
        return 0
    tagged_count = text.count("citation_id=")
    if tagged_count > 0:
        return tagged_count
    return 1


def reshape_source(row, author_id_mode: str = "sequence"):
    source = dict(row)
    source["authors"] = simplify_authors(source.get("authors"), author_id_mode=author_id_mode)
    source.pop("doc_id", None)
    source.pop("original_ack", None)
    source.pop("extracted_ack_items", None)
    source.pop("matched_results", None)

    citation_count = count_citations(source.get("citation"))
    reshaped = {}
    for key, value in source.items():
        reshaped[key] = value
        if key == "citation":
            reshaped["citation_count"] = citation_count
    if "citation" not in reshaped:
        reshaped["citation_count"] = citation_count
    reshaped["author_count"] = len(reshaped.get("authors", []))
    return reshaped


def normalize_text(value):
    return " ".join(str(value or "").strip().lower().split())


def parse_roots_level0(value):
    if isinstance(value, dict):
        parsed = value
    else:
        try:
            parsed = json.loads(value) if value else {}
        except Exception:
            parsed = {}
    roots = []
    if isinstance(parsed, dict):
        for root_list in parsed.values():
            if isinstance(root_list, list):
                roots.extend(root_list)
    return list(dict.fromkeys(str(x) for x in roots if str(x).strip()))


def build_matched_keywords(mag_row: dict, detail_row: dict):
    matches = mag_row.get("matches", []) if isinstance(mag_row, dict) else []
    details = detail_row.get("details", []) if isinstance(detail_row, dict) else []
    detail_by_name = {
        normalize_text(item.get("keyword_name")): item
        for item in details
        if isinstance(item, dict) and item.get("keyword_name")
    }

    grouped = defaultdict(lambda: {
        "mag_field_id": None,
        "matched_keyword": None,
        "source_terms": [],
        "match_scores": [],
        "final_level": None,
    })
    for row in matches:
        if not isinstance(row, dict):
            continue
        fid = row.get("mag_field_id")
        if fid is None or int(fid) < 0:
            continue
        key = int(fid)
        item = grouped[key]
        item["mag_field_id"] = key
        item["matched_keyword"] = row.get("mag_name")
        item["final_level"] = row.get("final_level")
        term = row.get("term")
        if term:
            item["source_terms"].append(term)
        score = row.get("score")
        if isinstance(score, (int, float)):
            item["match_scores"].append(float(score))

    merged = []
    for fid, item in grouped.items():
        detail = detail_by_name.get(normalize_text(item.get("matched_keyword")))
        merged.append({
            "matched_keyword": item.get("matched_keyword"),
            "llm_keywords": list(dict.fromkeys(item.get("source_terms") or [])),
            "final_level": item.get("final_level"),
            "attn_mass": detail.get("attn_mass") if detail else None,
            "final_root": detail.get("final_root") if detail else None,
            "roots_level0": parse_roots_level0(detail.get("roots_level0_dict")) if detail else [],
            "keyword_quantile": detail.get("keyword_quantile") if detail else None,
            "a_hat": detail.get("a_hat") if detail else None,
        })
    merged.sort(key=lambda x: ((x.get("matched_keyword") or "").lower(), x.get("mag_field_id") or -1))
    return merged


def main():
    ap = argparse.ArgumentParser(description="Export enriched X-index JSONL.")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--author_id_mode", choices=["sequence", "source_or_null"], default="sequence")
    ap.add_argument("--result_suffix", default="")
    args = ap.parse_args()

    input_jsonl = Path(args.input_jsonl)
    work_dir = Path(args.work_dir)
    out_jsonl = Path(args.out_jsonl)
    suffix = str(args.result_suffix or "").strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"

    summary_map = read_jsonl_map(work_dir / "summary" / f"crossindex_all{suffix}.jsonl")
    if not summary_map:
        raise SystemExit(f"missing summary rows under {work_dir / 'summary'}")

    keywords_map = {}
    mag_map = {}
    detail_map = {}
    for shard_dir in sorted((work_dir / "results").glob("shard_*")):
        keywords_map.update(read_jsonl_map(shard_dir / "keywords.jsonl"))
        mag_map.update(read_jsonl_map(shard_dir / "mag_matches.jsonl"))
        detail_map.update(read_jsonl_map(shard_dir / f"xindex_detail{suffix}.jsonl"))

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with input_jsonl.open("r", encoding="utf-8") as src, out_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id") or "").strip()
            if not doc_id or doc_id not in summary_map:
                continue

            source = reshape_source(row, author_id_mode=args.author_id_mode)

            keyword_row = keywords_map.get(doc_id, {})
            mag_row = mag_map.get(doc_id, {})
            detail_row = detail_map.get(doc_id, {})
            summary_row = summary_map[doc_id]

            output = {
                "doc_id": doc_id,
                "source": source,
                "llm_keywords": {
                    "keywords": keyword_row.get("keywords", []),
                    "keyword_roots": keyword_row.get("keyword_roots", []),
                    "decision": keyword_row.get("llm_cross_decision"),
                    "confidence": keyword_row.get("llm_cross_confidence"),
                    "reason": keyword_row.get("llm_cross_reason"),
                },
                "matched_keywords": build_matched_keywords(mag_row, detail_row),
                "xindex": {
                    "prompt_variant": summary_row.get("prompt_variant"),
                    "cutoff_year": summary_row.get("cutoff_year"),
                    "status": summary_row.get("status"),
                    "llm_cross_decision": summary_row.get("llm_cross_decision"),
                    "llm_cross_confidence": summary_row.get("llm_cross_confidence"),
                    "llm_cross_reason": summary_row.get("llm_cross_reason"),
                    "llm_keyword_roots": summary_row.get("llm_keyword_roots"),
                    "keyword_count": summary_row.get("keyword_count"),
                    "matched_keyword_count": summary_row.get("matched_keyword_count"),
                    "detail_rows": summary_row.get("detail_rows"),
                    "Vs": summary_row.get("Vs"),
                    "Vt": summary_row.get("Vt"),
                    "Vt_prime": summary_row.get("Vt_prime"),
                    "cross_index": summary_row.get("cross_index"),
                    "method_profile": summary_row.get("method_profile"),
                    "keyword_field_mode": summary_row.get("keyword_field_mode"),
                    "distance_mode": summary_row.get("distance_mode"),
                    "distance_resource_dir": summary_row.get("distance_resource_dir"),
                    "distance_year_requested": summary_row.get("distance_year_requested"),
                    "distance_year_used": summary_row.get("distance_year_used"),
                    "distance_year_reason": summary_row.get("distance_year_reason"),
                    "static_distance_year": summary_row.get("static_distance_year"),
                    "historical_formula_version": summary_row.get("historical_formula_version"),
                    "historical_attention_lambda": summary_row.get("historical_attention_lambda"),
                    "subject_matrix_mode": summary_row.get("subject_matrix_mode"),
                    "subject_matrix_lambda": summary_row.get("subject_matrix_lambda"),
                    "historical_Vs": summary_row.get("historical_Vs"),
                    "historical_Vt": summary_row.get("historical_Vt"),
                    "static_2021_Vs": summary_row.get("static_2021_Vs"),
                    "static_2021_Vt": summary_row.get("static_2021_Vt"),
                    "historical_pair_distance_score": summary_row.get("historical_pair_distance_score"),
                    "static_2021_pair_distance_score": summary_row.get("static_2021_pair_distance_score"),
                    "temporal_drift_pair_distance_score": summary_row.get("temporal_drift_pair_distance_score"),
                    "historical_modulated_attention": summary_row.get("historical_modulated_attention"),
                    "static_2021_modulated_attention": summary_row.get("static_2021_modulated_attention"),
                    "historical_attention_bias": summary_row.get("historical_attention_bias"),
                    "static_2021_attention_bias": summary_row.get("static_2021_attention_bias"),
                    "historical_subject_weights": summary_row.get("historical_subject_weights"),
                    "static_2021_subject_weights": summary_row.get("static_2021_subject_weights"),
                    "historical_adjusted_subject_weights": summary_row.get("historical_adjusted_subject_weights"),
                    "static_2021_adjusted_subject_weights": summary_row.get("static_2021_adjusted_subject_weights"),
                    "historical_anchor_subject": summary_row.get("historical_anchor_subject"),
                    "static_2021_anchor_subject": summary_row.get("static_2021_anchor_subject"),
                    "historical_subject_distance_from_anchor": summary_row.get("historical_subject_distance_from_anchor"),
                    "static_2021_subject_distance_from_anchor": summary_row.get("static_2021_subject_distance_from_anchor"),
                    "xindex_historical": summary_row.get("xindex_historical"),
                    "xindex_static_2021": summary_row.get("xindex_static_2021"),
                    "xindex_temporal_drift": summary_row.get("xindex_temporal_drift"),
                    "keyword_salience_mode": summary_row.get("keyword_salience_mode"),
                    "historical_q_status": summary_row.get("historical_q_status"),
                    "historical_q_audit": summary_row.get("historical_q_audit"),
                    "top_keyword_pairs": summary_row.get("top_keyword_pairs"),
                    "top_temporal_drift_keyword_pairs": summary_row.get("top_temporal_drift_keyword_pairs"),
                    "keyword_field_distributions": summary_row.get("keyword_field_distributions"),
                },
            }
            dst.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1

    print(json.dumps({
        "input_jsonl": str(input_jsonl),
        "work_dir": str(work_dir),
        "out_jsonl": str(out_jsonl),
        "written": written,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
