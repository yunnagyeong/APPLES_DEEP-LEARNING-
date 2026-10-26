"""조건 C(우리 팀 파이프라인)를 평가 문장 전체에 돌려 결과 CSV를 만듭니다.

사용법 (프로젝트 루트): python scripts/run_condition_c.py --user ganghee [--retries 2] [--out results/v2]
- 7주차 결과(results/week7)는 레거시 검증으로 만든 것이라 덮어쓰지 않도록 기본 출력 폴더가 results/v2입니다.
- 문장당 API: 번역 생성 1~3회 + 판정(VERIFY_MODEL) 1~3회 + 임베딩 2~4회. --workers개 문장을 동시에 처리.
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
ap.add_argument("--workers", type=int, default=4, help="동시에 처리할 문장 수")
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


def run_case(case, profile):
    try:
        result = translate_with_verification(
            korean_text=case["source"], style_profile=profile, user=USER, max_retries=MAX_RETRIES,
        )
        return {
            "id": case["id"], "condition": "C", "source": case["source"],
            "translation": result["translation"], "verdict": result["verdict"],
            "meaning_score": result.get("meaning_score"),
            "meaning_score_method": result.get("meaning_score_method"),
            "issues": "; ".join(f"{i['severity']}:{i['type']}" for i in result.get("issues", [])),
            "attempts": result.get("attempts"), "best_attempt": result.get("best_attempt"),
            "judge_model": result.get("judge_model"), "error": "",
        }
    except Exception as exc:
        return {
            "id": case["id"], "condition": "C", "source": case["source"], "translation": "",
            "verdict": "ERROR", "meaning_score": "", "meaning_score_method": "", "issues": "",
            "attempts": "", "best_attempt": "", "judge_model": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cases = load_cases()
    profile = _load_or_compute_style_profile(USER)
    order = {str(c["id"]): i for i, c in enumerate(cases)}
    rows = []

    # 문장 여러 개를 동시에 처리(--workers). 끝나는 대로 저장하되 파일은 원래 순서로 정렬.
    with ThreadPoolExecutor(max_workers=max(1, ARGS.workers)) as pool:
        futures = [pool.submit(run_case, c, profile) for c in cases]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            rows.sort(key=lambda r: order[str(r["id"])])
            save_csv(RESULT_PATH, rows, list(row.keys()))  # 중간에 중단돼도 끝난 결과는 보존
            print(f"[C] {n}/{len(cases)} | #{row['id']} Verdict={row['verdict']} | Score={row['meaning_score']}"
                  + (f" | {row['error']}" if row["error"] else ""))

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
