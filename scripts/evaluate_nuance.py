
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.verify_legacy import check_semantic_nuance

INPUT = ROOT / "data" / "measure" / "meaning_evaluation.jsonl"
OUTPUT = ROOT / "data" / "measure" / "nuance_evaluation.jsonl"

completed = {}

if OUTPUT.exists():
    with OUTPUT.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                completed[str(item["id"])] = item

with INPUT.open("r", encoding="utf-8") as f:
    cases = [json.loads(line) for line in f if line.strip()]

for case in cases:
    case_id = str(case["id"])

    if case_id in completed:
        print(f"SKIP #{case_id}")
        continue

    print(f"Checking #{case_id}...", flush=True)

    try:
        result = check_semantic_nuance(
            case["source"],
            case["translation"]
        )

        record = {
            "id": case_id,
            "human_error": case["human_error"],
            "human_severity": case["human_severity"],
            "nuance": result
        }

        with OUTPUT.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

        print(
            f"#{case_id}: "
            f"{result['has_meaning_error']} "
            f"{result['error_types']}",
            flush=True
        )

        time.sleep(2)

    except Exception as e:
        print(f"ERROR #{case_id}: {e}")
        break

print(f"\nSaved to: {OUTPUT}")
