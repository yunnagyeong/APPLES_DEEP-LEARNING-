"""[레거시] 7주차 NLI 검증(src/verify_legacy.py)으로 error_cases.csv를 평가. 새 검증은 scripts/eval_verifier.py."""

import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.verify_legacy import verify_translation
from src.meaning_score import calculate_meaning_score


INPUT_PATH = ROOT / "error_cases.csv"
OUTPUT_DIR = ROOT / "data" / "measure"
RESULT_PATH = OUTPUT_DIR / "meaning_evaluation.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "meaning_evaluation_summary.csv"

# 사례 사이의 간격. 분당 API 요청 제한을 고려한 값.
DELAY_SECONDS = 2


def load_completed():
    completed = {}

    if not RESULT_PATH.exists():
        return completed

    with RESULT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                completed[str(item["id"])] = item

    return completed


def save_result(item):
    with RESULT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def export_summary(results):
    fields = [
        "id",
        "category",
        "human_error",
        "human_severity",
        "verdict",
        "meaning_score",
        "semantic_similarity",
        "claim_coverage",
        "translation_support_rate",
        "contradiction_rate",
        "failure_count",
    ]

    with SUMMARY_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for item in sorted(
            results.values(), key=lambda x: int(x["id"])
        ):
            v = item["verification"]
            m = item["meaning"]

            writer.writerow({
                "id": item["id"],
                "category": item["category"],
                "human_error": item["human_error"],
                "human_severity": item["human_severity"],
                "verdict": v["verdict"],
                "meaning_score": m["meaning_score"],
                "semantic_similarity": m["semantic_similarity"],
                "claim_coverage": m["claim_coverage"],
                "translation_support_rate":
                    m["translation_support_rate"],
                "contradiction_rate": m["contradiction_rate"],
                "failure_count": len(v["failure_reasons"]),
            })


def print_analysis(results):
    print("\n" + "=" * 60)
    print("MEANING SCORE EVALUATION SUMMARY")
    print("=" * 60)

    groups = {
        "정상": [],
        "하": [],
        "중": [],
        "상": [],
    }

    for item in results.values():
        severity = item["human_severity"] or "정상"
        score = item["meaning"]["meaning_score"]

        if severity in groups and score is not None:
            groups[severity].append(score)

    for severity, scores in groups.items():
        if scores:
            print(
                f"{severity}: "
                f"{len(scores)} cases, "
                f"average={mean(scores):.2f}, "
                f"min={min(scores):.2f}, "
                f"max={max(scores):.2f}"
            )

    normal = [
        item for item in results.values()
        if not item["human_severity"]
    ]
    errors = [
        item for item in results.values()
        if item["human_severity"]
    ]

    false_alarms = sum(
        item["verification"]["verdict"] != "PASS"
        for item in normal
    )

    missed_errors = sum(
        item["verification"]["verdict"] == "PASS"
        for item in errors
    )

    print(f"\nCompleted: {len(results)}/32")
    print(f"Normal cases tested: {len(normal)}")
    print(f"Error cases tested: {len(errors)}")
    print(f"False alarms: {false_alarms}")
    print(f"Missed errors: {missed_errors}")
    print(f"\nDetailed results: {RESULT_PATH}")
    print(f"Summary CSV: {SUMMARY_PATH}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        cases = list(csv.DictReader(f))

    completed = load_completed()

    print(f"Total cases: {len(cases)}")
    print(f"Previously completed: {len(completed)}")

    for index, case in enumerate(cases, start=1):
        case_id = case["번호"]

        if case_id in completed:
            print(f"[{index}/{len(cases)}] #{case_id} SKIPPED")
            continue

        print(f"\n[{index}/{len(cases)}] Testing #{case_id}")
        print(f"Severity: {case['심각도'] or '정상'}")

        try:
            verification = verify_translation(
                case["원문"],
                case["번역"]
            )

            meaning = calculate_meaning_score(verification)

        except Exception as exc:
            print(f"\n[ERROR] Case #{case_id}: {exc}")
            print("Completed results have been saved.")
            print("Run this script again to resume.")
            break

        item = {
            "id": case_id,
            "category": case["카테고리"],
            "source": case["원문"],
            "translation": case["번역"],
            "human_error": case["문제유형"],
            "human_severity": case["심각도"],
            "verification": verification,
            "meaning": meaning,
        }

        save_result(item)
        completed[case_id] = item

        print(f"Verdict: {verification['verdict']}")
        print(f"Meaning Score: {meaning['meaning_score']}")
        print(f"Saved: #{case_id}")

        export_summary(completed)

        if index < len(cases):
            time.sleep(DELAY_SECONDS)

    export_summary(completed)
    print_analysis(completed)


if __name__ == "__main__":
    main()
