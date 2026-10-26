
def _label_counts(claims: list[dict]) -> dict:
    counts = {
        "entailment": 0,
        "neutral": 0,
        "contradiction": 0,
    }

    for claim in claims:
        label = str(claim["label"]).lower()

        if label not in counts:
            raise ValueError(f"Unknown NLI label: {label}")

        counts[label] += 1

    return counts


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def calculate_meaning_components(verification: dict) -> dict:
    source = _label_counts(verification["source_claims"])
    translation = _label_counts(verification["translation_claims"])

    source_total = sum(source.values())
    translation_total = sum(translation.values())
    total = source_total + translation_total

    coverage = (
        source["entailment"] / source_total
        if source_total > 0 else None
    )

    support_rate = (
        translation["entailment"] / translation_total
        if translation_total > 0 else None
    )

    contradiction_rate = (
        (
            source["contradiction"]
            + translation["contradiction"]
        ) / total
        if total > 0 else None
    )

    return {
        "semantic_similarity": verification["semantic_similarity"],
        "claim_coverage": coverage,
        "translation_support_rate": support_rate,
        "contradiction_rate": contradiction_rate,
        "source_claim_count": source_total,
        "translation_claim_count": translation_total,
    }


def calculate_meaning_score(verification: dict) -> dict:
    """
    Evidence-Gated Meaning Score

    Standard:
        M = 100 * S^0.2 * C^0.4 * T^0.4 * (1-R)

    Fallback:
        M = 100 * S

    Fallback is used when claim-based evaluation
    is unavailable. It is explicitly marked as
    SIMILARITY_ONLY and is not equivalent to
    a fully verified Meaning Score.
    """

    components = calculate_meaning_components(verification)

    s = components["semantic_similarity"]
    c = components["claim_coverage"]
    t = components["translation_support_rate"]
    r = components["contradiction_rate"]

    nuance = verification.get("semantic_nuance") or {}

    nuance_errors = [
        reason
        for reason in verification.get("failure_reasons", [])
        if reason.get("type") == "SEMANTIC_NUANCE"
    ]

    nuance_error = (
        bool(nuance.get("has_meaning_error", False))
        or bool(nuance_errors)
    )

    # Semantic similarity 자체가 없으면 평가 불가
    if s is None:
        return {
            **components,
            "meaning_score": None,
            "status": "UNAVAILABLE",
            "score_method": "UNAVAILABLE",
            "semantic_nuance_error": nuance_error,
        }

    s = _clamp(s)

    # Claim이 추출되지 않은 경우
    if any(v is None for v in (c, t, r)):
        score = round(100.0 * s, 2)

        return {
            **components,
            "meaning_score": score,
            "status": (
                "SEMANTIC_ERROR_DETECTED"
                if nuance_error
                else "SIMILARITY_ONLY"
            ),
            "score_method": "SIMILARITY_ONLY",
            "semantic_nuance_error": nuance_error,
        }

    c = _clamp(c)
    t = _clamp(t)
    r = _clamp(r)

    score = (
        100.0
        * (s ** 0.2)
        * (c ** 0.4)
        * (t ** 0.4)
        * (1.0 - r)
    )

    return {
        **components,
        "meaning_score": round(score, 2),
        "status": (
            "SEMANTIC_ERROR_DETECTED"
            if nuance_error
            else "OK"
        ),
        "score_method": "CLAIM_BASED",
        "semantic_nuance_error": nuance_error,
    }


if __name__ == "__main__":
    # Claim이 없는 정상 번역 테스트
    sample = {
        "semantic_similarity": 0.91,
        "source_claims": [],
        "translation_claims": [],
        "failure_reasons": [],
    }

    result = calculate_meaning_score(sample)

    print("Meaning Score:", result["meaning_score"])
    print("Status:", result["status"])
    print("Method:", result["score_method"])
