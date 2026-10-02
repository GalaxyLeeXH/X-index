#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


VARIANTS = ["baseline_no_extra", "hash_only", "temporal_no_hash", "temporal_hash"]


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_map(path: Path, key: str = "doc_id"):
    out = {}
    for row in iter_jsonl(path) or []:
        value = row.get(key)
        if value is not None:
            out[str(value)] = row
    return out


def read_shard_map(results_dir: Path, filename: str):
    out = {}
    for fp in sorted(results_dir.glob(f"shard_*/{filename}")):
        out.update(read_map(fp))
    return out


def choose_metric(summary: dict, variant: str):
    if variant.startswith("temporal") and summary.get("xindex_historical") is not None:
        return {
            "xindex": summary.get("xindex_historical"),
            "vs": summary.get("historical_Vs"),
            "vt": summary.get("historical_Vt"),
        }
    return {
        "xindex": summary.get("cross_index"),
        "vs": summary.get("Vs"),
        "vt": summary.get("Vt"),
    }


def keyword_attention(detail_row: dict, attention_row: dict):
    details = detail_row.get("details") if isinstance(detail_row, dict) else None
    if isinstance(details, list) and details:
        rows = []
        for item in details:
            if not isinstance(item, dict):
                continue
            name = item.get("keyword_name")
            if not name:
                continue
            rows.append({
                "keyword": name,
                "attention": item.get("attn_mass"),
                "a_hat": item.get("a_hat"),
                "final_root": item.get("final_root"),
                "source_terms": item.get("source_terms"),
            })
        return rows
    masses = attention_row.get("attention_masses") if isinstance(attention_row, dict) else None
    if isinstance(masses, dict):
        return [
            {"keyword": key, "attention": value}
            for key, value in sorted(masses.items(), key=lambda kv: str(kv[0]).lower())
        ]
    return []


def json_cell(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser(description="Combine four X-index variants into one RQ2 paper-level table.")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--experiment_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    input_path = Path(args.input_jsonl)
    exp = Path(args.experiment_dir)
    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    variant_data = {}
    for variant in VARIANTS:
        work = exp / variant
        variant_data[variant] = {
            "summary": read_map(work / "summary" / "crossindex_all.jsonl"),
            "keywords": read_shard_map(work / "results", "keywords.jsonl"),
            "detail": read_shard_map(work / "results", "xindex_detail.jsonl"),
            "attention": read_shard_map(work / "results", "attention.jsonl"),
        }

    rows = []
    for src in iter_jsonl(input_path) or []:
        doc_id = str(src.get("doc_id") or src.get("paperid") or "").strip()
        if not doc_id:
            continue
        row = {
            "paperid": src.get("paperid") or doc_id,
            "title": src.get("title"),
            "keyword": [],
        }
        for variant in VARIANTS:
            data = variant_data[variant]
            kw_row = data["keywords"].get(doc_id, {})
            if not row["keyword"] and kw_row.get("keywords"):
                row["keyword"] = kw_row.get("keywords")
            summary = data["summary"].get(doc_id, {})
            metrics = choose_metric(summary, variant)
            row[f"{variant}_xindex"] = metrics["xindex"]
            row[f"{variant}_vs"] = metrics["vs"]
            row[f"{variant}_vt"] = metrics["vt"]
            row[f"{variant}_keyword_attention"] = keyword_attention(
                data["detail"].get(doc_id, {}),
                data["attention"].get(doc_id, {}),
            )
        rows.append(row)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = ["paperid", "title", "keyword"]
    for variant in VARIANTS:
        fieldnames.extend([
            f"{variant}_xindex",
            f"{variant}_vs",
            f"{variant}_vt",
            f"{variant}_keyword_attention",
        ])
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["keyword"] = json_cell(csv_row.get("keyword", []))
            for variant in VARIANTS:
                key = f"{variant}_keyword_attention"
                csv_row[key] = json_cell(csv_row.get(key, []))
            writer.writerow(csv_row)

    print(json.dumps({
        "experiment_dir": str(exp),
        "out_jsonl": str(out_jsonl),
        "out_csv": str(out_csv),
        "rows": len(rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
