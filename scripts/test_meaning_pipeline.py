"""파이프라인 1문장 실행 확인. 사용법: python scripts/test_meaning_pipeline.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import (  # noqa: E402
    _load_or_compute_style_profile,
    translate_with_verification,
)

USER = "ganghee"

SOURCE = "이 결과는 우연일 수도 있다."


def main():
    profile = _load_or_compute_style_profile(USER)

    result = translate_with_verification(
        korean_text=SOURCE,
        style_profile=profile,
        user=USER,
        max_retries=1,
    )

    print("\n========== FINAL RESULT ==========")
    print("Original:", SOURCE)
    print("Translation:", result["translation"])
    print("Verdict:", result["verdict"])
    print("Meaning Score:", result["meaning_score"])
    print("Judge model:", result["judge_model"])

    print("\nMeaning Components:")
    print(
        json.dumps(
            result["meaning_components"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nIssues:")
    print(json.dumps(result["issues"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
