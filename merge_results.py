#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


SUMMARY_FIELDS = [
    "doc_id",
    "title",
    "year",
    "prompt_variant",
    "cutoff_year",
    "status",
    "llm_cross_decision",
    "llm_cross_confidence",
    "llm_cross_reason",
    "llm_keyword_roots",
    "keyword_count",
    "matched_keyword_count",
    "detail_rows",
    "Vs",
    "Vt",
    "Vt_prime",
    "cross_index",
    "method_profile",
    "keyword_field_mode",
    "distance_mode",
    "distance_resource_dir",
    "distance_year_requested",
    "distance_year_used",
    "distance_year_reason",
    "static_distance_year",
    "historical_formula_version",
    "historical_attention_lambda",
    "subject_matrix_mode",
    "subject_matrix_lambda",
    "historical_Vs",
    "historical_Vt",
    "static_2021_Vs",
    "static_2021_Vt",
    "historical_pair_distance_score",
    "static_2021_pair_distance_score",
    "temporal_drift_pair_distance_score",
    "historical_modulated_attention",
    "static_2021_modulated_attention",
    "historical_attention_bias",
    "static_2021_attention_bias",
    "historical_subject_weights",
    "static_2021_subject_weights",
    "historical_adjusted_subject_weights",
    "static_2021_adjusted_subject_weights",
    "historical_anchor_subject",
    "static_2021_anchor_subject",
    "historical_subject_distance_from_anchor",
    "static_2021_subject_distance_from_anchor",
    "xindex_historical",
    "xindex_static_2021",
    "xindex_temporal_drift",
    "historical_distance_score",
    "static_2021_distance_score",
    "temporal_drift_score",
    "keyword_salience_mode",
    "historical_q_status",
    "historical_q_audit",
    "top_keyword_pairs",
    "top_temporal_drift_keyword_pairs",
    "keyword_field_distributions",
]

DIAG_FIELDS = [
    "doc_id",
    "algo_profile",
    "vs_rows_before_filter",
    "vs_rows_after_filter",
    "singleton_root_count",
    "distinct_final_root_before_filter",
    "distinct_final_root_after_filter",
    "dominant_roots",
    "anchored_root_counts",
    "anchored_keyword_count",
    "weak_keyword_count",
    "root_gate_applied",
    "dropped_keyword_count",
    "kept_keyword_count_after_gate",
    "root_strength_by_root",
    "root_share_by_root",
    "dropped_roots_by_strength",
    "strong_anchor_count_by_root",
    "root_gate_mode",
    "llm_cross_decision",
    "llm_cross_confidence",
    "llm_cross_reason",
    "llm_keyword_roots",
    "kept_roots_after_distribution_gate",
    "dropped_roots_by_distribution_gate",
    "dropped_roots_by_singleton_gate",
    "vs_rows_before_root_gate",
    "vs_rows_after_distribution_gate",
    "vs_rows_after_singleton_gate",
]


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description="Merge shard X-index outputs.")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_jsonl", default="")
    ap.add_argument("--out_diag_jsonl", default="")
    ap.add_argument("--result_suffix", default="")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    suffix = str(args.result_suffix or "").strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    rows_by_doc = {}
    for fp in sorted(results_dir.glob(f"shard_*/xindex_partial{suffix}.jsonl")):
        for row in iter_jsonl(fp):
            doc_id = row.get("doc_id")
            if not doc_id:
                continue
            rows_by_doc[doc_id] = row

    rows = [rows_by_doc[k] for k in sorted(rows_by_doc)]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if args.out_jsonl:
        out_jsonl = Path(args.out_jsonl)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                summary_row = {k: row.get(k) for k in SUMMARY_FIELDS}
                f.write(json.dumps(summary_row, ensure_ascii=False) + "\n")

    if args.out_diag_jsonl:
        out_diag_jsonl = Path(args.out_diag_jsonl)
        out_diag_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_diag_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                diag = {k: row.get(k) for k in DIAG_FIELDS}
                f.write(json.dumps(diag, ensure_ascii=False) + "\n")

    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"[OK] merged={len(rows)} ok={ok} -> {out_csv}")


if __name__ == "__main__":
    main()
