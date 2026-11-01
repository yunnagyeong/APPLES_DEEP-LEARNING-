
import json
import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "measure"

INPUT = DATA_DIR / "meaning_evaluation.jsonl"
OUTPUT = DATA_DIR / "meaning_formula_comparison.csv"

SEVERITIES = ["정상", "하", "중", "상"]


def load_cases():
    with INPUT.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def calculate_candidates(meaning):
    s = meaning["semantic_similarity"]
    c = meaning["claim_coverage"]
    t = meaning["translation_support_rate"]
    r = meaning["contradiction_rate"]

    if any(v is None for v in (s, c, t, r)):
        return {"A": None, "B": None, "C": None}

    # 기존 공식
    a = 100 * (0.2*s + 0.4*c + 0.4*t) * (1-r)

    # 기하평균
    b = 100 * (s*c*t)**(1/3) * (1-r)

    # 조화평균
    if min(s, c, t) <= 0:
        harmonic = 0.0
    else:
        harmonic = 3 / (1/s + 1/c + 1/t)

    cc = 100 * harmonic * (1-r)

    return {
        "A": round(max(0, min(100, a)), 2),
        "B": round(max(0, min(100, b)), 2),
        "C": round(max(0, min(100, cc)), 2),
    }


def severity_of(case):
    return case["human_severity"] or "정상"


def pairwise_accuracy(rows, formula):
    """
    오류 사례의 점수가 정상 사례보다 낮은
    쌍의 비율. 동점은 0.5로 계산.
    """
    normal = [
        r[formula] for r in rows
        if r["severity"] == "정상"
        and r[formula] is not None
    ]
    errors = [
        r[formula] for r in rows
        if r["severity"] != "정상"
        and r[formula] is not None
    ]

    if not normal or not errors:
        return None

    correct = 0
    total = 0

    for n in normal:
        for e in errors:
            total += 1
            if e < n:
                correct += 1
            elif e == n:
                correct += 0.5

    return correct / total


def main():
    cases = load_cases()
    rows = []

    for case in cases:
        scores = calculate_candidates(case["meaning"])

        rows.append({
            "id": int(case["id"]),
            "severity": severity_of(case),
            "verdict": case["verification"]["verdict"],
            "A": scores["A"],
            "B": scores["B"],
            "C": scores["C"],
        })

    rows.sort(key=lambda x: x["id"])

    with OUTPUT.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "severity", "verdict",
                "A", "B", "C"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== FORMULA COMPARISON ===")

    for formula in ["A", "B", "C"]:
        print(f"\nFormula {formula}")

        for severity in SEVERITIES:
            values = [
                r[formula] for r in rows
                if r["severity"] == severity
                and r[formula] is not None
            ]

            if values:
                print(
                    f"{severity}: n={len(values)}, "
                    f"mean={mean(values):.2f}, "
                    f"min={min(values):.2f}, "
                    f"max={max(values):.2f}"
                )

        accuracy = pairwise_accuracy(rows, formula)
        if accuracy is not None:
            print(
                f"Normal-error pairwise separation: "
                f"{accuracy:.3f}"
            )

    print("\n=== IMPORTANT CASES ===")

    for case_id in [3, 11, 13, 15, 17, 20, 23, 26, 30]:
        r = next(x for x in rows if x["id"] == case_id)
        print(
            f"#{case_id:02d} "
            f"severity={r['severity']} "
            f"A={r['A']} "
            f"B={r['B']} "
            f"C={r['C']}"
        )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
