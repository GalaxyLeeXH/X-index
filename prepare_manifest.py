#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def iter_json_records(path):
    decoder = json.JSONDecoder(strict=False)
    buffer = ""
    start_line = None
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip() and not buffer:
                continue
            if not buffer:
                start_line = lineno
            buffer += line
            try:
                obj, end = decoder.raw_decode(buffer.lstrip())
                rest = buffer.lstrip()[end:].strip()
                if rest:
                    continue
            except json.JSONDecodeError:
                continue
            yield start_line, obj
            buffer = ""
            start_line = None
    if buffer.strip():
        yield start_line or 0, None


def is_valid_year(value):
    if value is None:
        return False
    s = str(value).strip()
    return s.isdigit()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Prepare valid-paper manifest and shards for batch X-index.")
    ap.add_argument("--input", required=True, help="Input JSONL with doc_id/title/abs/year.")
    ap.add_argument("--work_dir", required=True, help="Task work directory.")
    ap.add_argument("--shard_size", type=int, default=100)
    ap.add_argument("--max_records", type=int, default=0, help="Debug limit after filtering; 0 means all.")
    args = ap.parse_args()

    src = Path(args.input)
    work = Path(args.work_dir)
    shard_dir = work / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    valid = []
    skipped = []
    seen = set()

    for lineno, obj in iter_json_records(src):
        if obj is None:
            skipped.append({"line": lineno, "reason": "json_error", "error": "incomplete_or_invalid_json_record"})
            continue

        doc_id = str(obj.get("doc_id") or "").strip()
        title = str(obj.get("title") or "").strip()
        abstract = str(obj.get("abs") or obj.get("abstract") or "").strip()
        year_raw = obj.get("year")
        year = str(year_raw).strip() if year_raw is not None else ""

        reason = None
        if not doc_id:
            reason = "missing_doc_id"
        elif doc_id in seen:
            reason = "duplicate_doc_id"
        elif not title:
            reason = "missing_title"
        elif not abstract:
            reason = "missing_abs"
        elif not is_valid_year(year):
            reason = "invalid_or_unknown_year"

        if reason:
            skipped.append({
                "doc_id": doc_id,
                "title": title,
                "year": year,
                "reason": reason,
                "line": lineno,
            })
            continue

        seen.add(doc_id)
        clean_obj = dict(obj)
        clean_obj["doc_id"] = doc_id
        clean_obj["title"] = title
        clean_obj["abs"] = abstract
        clean_obj["year"] = int(year)
        valid.append(clean_obj)

        if args.max_records and len(valid) >= args.max_records:
            break

    write_jsonl(work / "manifest.valid.jsonl", valid)
    write_jsonl(work / "manifest.skipped.jsonl", skipped)

    for old in shard_dir.glob("shard_*.jsonl"):
        old.unlink()

    shard_count = 0
    for i in range(0, len(valid), args.shard_size):
        rows = valid[i:i + args.shard_size]
        shard_path = shard_dir / f"shard_{shard_count:05d}.jsonl"
        write_jsonl(shard_path, rows)
        shard_count += 1

    summary = {
        "input": str(src),
        "work_dir": str(work),
        "valid": len(valid),
        "skipped": len(skipped),
        "shard_size": args.shard_size,
        "shards": shard_count,
    }
    (work / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
