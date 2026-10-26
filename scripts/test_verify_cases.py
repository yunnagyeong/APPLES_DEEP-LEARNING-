import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.verify_legacy import verify_translation


# 우선 2개 사례만 테스트
TEST_IDS = {"2", "23"}

with (ROOT / "error_cases.csv").open(
    encoding="utf-8-sig", newline=""
) as f:
    cases = list(csv.DictReader(f))


for case in cases:
    if case["번호"] not in TEST_IDS:
        continue

    print(f"\n{'=' * 60}")
    print(f"CASE #{case['번호']}")
    print(f"원문: {case['원문']}")
    print(f"번역: {case['번역']}")
    print(f"사람 평가: {case['문제유형']}")
    print(f"심각도: {case['심각도'] or '정상'}")

    result = verify_translation(
        case["원문"],
        case["번역"]
    )

    print("\n[Verification 결과]")
    print(json.dumps(
        result,
        ensure_ascii=False,
        indent=2
    ))