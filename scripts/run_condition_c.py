
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from src.pipeline import (
    _load_or_compute_style_profile,
    translate_with_verification,
)

USER = "ganghee"
INPUT_PATH = Path("data/measure/team_evaluation.jsonl")
OUTPUT_DIR = Path("results/week7")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUTPUT_DIR / "condition_c_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "condition_c_summary.csv"

MAX_RETRIES = 2


def load_cases():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"평가 데이터가 없습니다: {INPUT_PATH}\n"
            "팀 공통 평가 문장을 JSONL 형식으로 준비하세요."
        )

    cases = []

    with INPUT_PATH.open(encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            case = json.loads(line)
            source = case.get("source", "").strip()

            if not source:
                raise ValueError(
                    f"{line_number}번째 줄에 source가 없습니다."
                )

            cases.append({
                "id": case.get("id", len(cases) + 1),
                "source": source,
            })

    if not cases:
        raise ValueError("평가 문장이 비어 있습니다.")

    return cases


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    cases = load_cases()
    profile = _load_or_compute_style_profile(USER)

    rows = []

    for index, case in enumerate(cases, 1):
        print(f"\n[C] {index}/{len(cases)} | {case['id']}")

        try:
            result = translate_with_verification(
                korean_text=case["source"],
                style_profile=profile,
                user=USER,
                max_retries=MAX_RETRIES,
            )

            row = {
                "id": case["id"],
                "condition": "C",
                "source": case["source"],
                "translation": result["translation"],
                "verdict": result["verdict"],
                "meaning_score": result.get("meaning_score"),
                "meaning_score_status": result.get(
                    "meaning_score_status"
                ),
                "meaning_score_method": result.get(
                    "meaning_score_method"
                ),
                "semantic_nuance_error": result.get(
                    "semantic_nuance_error"
                ),
                "attempts": result.get("attempts"),
                "error": "",
            }

        except Exception as exc:
            row = {
                "id": case["id"],
                "condition": "C",
                "source": case["source"],
                "translation": "",
                "verdict": "ERROR",
                "meaning_score": "",
                "meaning_score_status": "",
                "meaning_score_method": "",
                "semantic_nuance_error": "",
                "attempts": "",
                "error": f"{type(exc).__name__}: {exc}",
            }

        rows.append(row)

        # 중간에 중단되어도 이전 결과 보존
        save_csv(RESULT_PATH, rows, list(row.keys()))

        print(
            f"Verdict={row['verdict']} | "
            f"Score={row['meaning_score']}"
        )

    scores = [
        float(row["meaning_score"])
        for row in rows
        if row["meaning_score"] not in ("", None)
    ]

    verdict_counts = Counter(
        row["verdict"] for row in rows
    )

    summary = {
        "condition": "C",
        "total": len(rows),
        "scored": len(scores),
        "mean": statistics.mean(scores) if scores else "",
        "median": statistics.median(scores) if scores else "",
        "std": (
            statistics.stdev(scores)
            if len(scores) >= 2 else ""
        ),
        "min": min(scores) if scores else "",
        "max": max(scores) if scores else "",
        "pass": verdict_counts["PASS"],
        "review": verdict_counts["REVIEW"],
        "fail": verdict_counts["FAIL"],
        "error": verdict_counts["ERROR"],
        "similarity_only": sum(
            row["meaning_score_method"] == "SIMILARITY_ONLY"
            for row in rows
        ),
    }

    save_csv(SUMMARY_PATH, [summary], list(summary.keys()))

    print("\n=== CONDITION C COMPLETE ===")
    print(f"Total: {len(rows)}")
    print(f"Scored: {len(scores)}")
    print(f"Mean: {summary['mean']}")
    print(f"PASS: {summary['pass']}")
    print(f"REVIEW: {summary['review']}")
    print(f"FAIL: {summary['fail']}")
    print(f"ERROR: {summary['error']}")
    print(f"\nResults: {RESULT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
