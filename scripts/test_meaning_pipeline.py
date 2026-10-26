
import json
from src.pipeline import (
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
        max_retries=0,
    )

    print("\n========== FINAL RESULT ==========")
    print("Original:", SOURCE)
    print("Translation:", result["translation"])
    print("Verdict:", result["verdict"])
    print("Meaning Score:", result["meaning_score"])
    print("Meaning Status:", result["meaning_score_status"])
    print("Semantic Error:", result["semantic_nuance_error"])

    print("\nMeaning Components:")
    print(
        json.dumps(
            result["meaning_components"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nVerification Failure Reasons:")
    print(
        json.dumps(
            result["verification_history"][-1].get(
                "failure_reasons", []
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
