"""조건 C(우리 팀 파이프라인)를 평가 문장 전체에 돌려 결과 CSV를 만듭니다.

사용법 (프로젝트 루트): python scripts/run_condition_c.py --user ganghee [--retries 2] [--out results/v2]
- 7주차 결과(results/week7)는 레거시 검증으로 만든 것이라 덮어쓰지 않도록 기본 출력 폴더가 results/v2입니다.
- 무료 티어: 문장당 판정 모델(VERIFY_MODEL) 1~3회. 하루 한도가 끝나면 config.VERIFY_FALLBACK_MODEL로 자동 대체되고
  결과의 judge_model 열에 기록됩니다.
"""
import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import (  # noqa: E402
    _load_or_compute_style_profile,
    translate_with_verification,
)

ap = argparse.ArgumentParser()
ap.add_argument("--user", default="ganghee")
ap.add_argument("--retries", type=int, default=2)
ap.add_argument("--out", default="results/v2")
ap.add_argument("--input", default="data/measure/team_evaluation.jsonl")
ARGS = ap.parse_args()

USER = ARGS.user
INPUT_PATH = ROOT / ARGS.input
OUTPUT_DIR = ROOT / ARGS.out
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUTPUT_DIR / f"condition_c_results_{USER}.csv"
SUMMARY_PATH = OUTPUT_DIR / f"condition_c_summary_{USER}.csv"

MAX_RETRIES = ARGS.retries


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
                "meaning_score_method": result.get("meaning_score_method"),
                "issues": "; ".join(
                    f"{i['severity']}:{i['type']}" for i in result.get("issues", [])
                ),
                "attempts": result.get("attempts"),
                "best_attempt": result.get("best_attempt"),
                "judge_model": result.get("judge_model"),
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
                "meaning_score_method": "",
                "issues": "",
                "attempts": "",
                "best_attempt": "",
                "judge_model": "",
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
        "retranslated": sum(
            1 for row in rows if row["attempts"] not in ("", None) and int(row["attempts"]) > 1
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
