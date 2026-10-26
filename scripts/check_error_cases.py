import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

files = list(ROOT.rglob("error_cases.csv"))

if not files:
    print("error_cases.csv 파일을 찾지 못했습니다.")
else:
    for path in files:
        print(f"\n파일: {path.relative_to(ROOT)}")

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"총 사례 수: {len(rows)}")
        print(f"컬럼: {reader.fieldnames}")

        for i, row in enumerate(rows, 1):
            print(f"\n[{i}] {row}")