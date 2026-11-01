"""Meaning Score — 의미 보존 점수 (0~100).

v2 (src/verify.py 결과용, 기본)
    M = min(100 × F × A × G, 상한)
    상한 = 가장 심한 문제가 critical이면 50, major면 75, minor면 95, 문제 없으면 100
    F (fidelity)   = 원문 claim별 점수의 평균. OK 1.0 / minor 0.85 / major 0.6 / critical 0.0
    A (additions)  = Π(1 − p). 번역문에만 있는 내용마다 감점. MINOR_ADDITION 0.03 / UNSUPPORTED_ADDITION 0.5
    G (rule guard) = Π 0.6. 규칙 검사(숫자 불일치·유사도 하한) 문제마다 major와 같은 비율로 감점

    설계 의도
    - 점수가 판정(verdict)과 같은 근거(오류 유형 → 심각도 표)에서 나오므로 둘이 어긋나지 않음.
      상한 덕분에 구간이 겹치지 않음: PASS 85~100, REVIEW ≤ 75, FAIL ≤ 50 (긴 문단에서도 동일).
    - 7주차 공식(S^0.2·C^0.4·T^0.4·(1−R))의 문제 해결:
      · C(원문 claim 보존율)가 32건 전부 1.0이라 변별력이 없었음 → claim마다 "어떻게" 바뀌었는지 반영
      · 임베딩 유사도 S(정상 0.85~0.95, 오류 0.83~0.91)는 차이가 작아 점수에서 뺌(규칙 하한으로만 사용)
      · Gemini가 찾은 오류가 점수에 반영되지 않아 "점수 98인데 REVIEW" 같은 어긋남이 있었음
    - 가중치는 error_cases.csv 사람 심각도(정상 > 하 > 중 > 상) 순서를 지키도록 정한 설계값이며,
      통계적으로 최적화한 값은 아님. 같은 32문장으로 정했으므로 scripts/eval_verifier.py로 새 문장에서 확인 필요.
      (예: 정상 100 / 하: 시제·정도 변화 85 / 중: 확신도 변화 60 / 상: 지어낸 근거 추가 ≤50)

legacy (7주차 verify_legacy 결과용)
    M = 100 × S^0.2 × C^0.4 × T^0.4 × (1 − R), claim이 없으면 100 × S
"""

from __future__ import annotations

CLAIM_CREDIT = {"none": 1.0, "minor": 0.85, "major": 0.6, "critical": 0.0}
ADDITION_PENALTY = {"MINOR_ADDITION": 0.03, "UNSUPPORTED_ADDITION": 0.5}
RULE_FACTOR = 0.6
# 가장 심한 문제의 심각도별 점수 상한. claim이 많은 긴 문단에서 오류 하나가 평균에 묻혀
# "FAIL인데 91점"이 되는 것을 막아 점수 구간이 판정과 겹치지 않게 함 (새 평가셋에서 발견).
SEVERITY_CAP = {"minor": 95.0, "major": 75.0, "critical": 50.0}
_RANK = {"none": 0, "minor": 1, "major": 2, "critical": 3}


def calculate_meaning_score(verification: dict) -> dict:
    """verification이 v2 형식(issues 키)이면 v2 공식, 아니면 레거시 공식."""
    if "issues" in verification and "source_claims" in verification and \
            all("severity" in c for c in verification["source_claims"]):
        return _score_v2(verification)
    return calculate_legacy_meaning_score(verification)


def _score_v2(v: dict) -> dict:
    claims = v["source_claims"]
    fidelity = sum(CLAIM_CREDIT[c["severity"]] for c in claims) / len(claims) if claims else 0.0
    additions = 1.0
    for a in v.get("additions", []):
        additions *= 1.0 - ADDITION_PENALTY.get(a["label"], 0.5)
    guard = 1.0
    for i in v.get("issues", []):
        if i.get("origin") == "rule":
            guard *= RULE_FACTOR
    raw = 100.0 * fidelity * additions * guard
    worst = max((i["severity"] for i in v.get("issues", [])), key=lambda x: _RANK.get(x, 0), default="none")
    cap = SEVERITY_CAP.get(worst, 100.0)
    score = round(min(raw, cap), 1)
    return {
        "meaning_score": score,
        "score_method": "CLAIM_ALIGNMENT_V2",
        "meaning_components": {
            "fidelity": round(fidelity, 3),
            "addition_factor": round(additions, 3),
            "rule_factor": round(guard, 3),
            "severity_cap": cap,
            "semantic_similarity": v.get("semantic_similarity"),
            "source_claim_count": len(claims),
            "addition_count": len(v.get("additions", [])),
        },
    }


# ---------------------------------------------------------------- 레거시 (7주차)

def _label_counts(claims: list[dict]) -> dict:
    counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
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
    return {
        "semantic_similarity": verification["semantic_similarity"],
        "claim_coverage": source["entailment"] / source_total if source_total > 0 else None,
        "translation_support_rate": translation["entailment"] / translation_total if translation_total > 0 else None,
        "contradiction_rate": (source["contradiction"] + translation["contradiction"]) / total if total > 0 else None,
        "source_claim_count": source_total,
        "translation_claim_count": translation_total,
    }


def calculate_legacy_meaning_score(verification: dict) -> dict:
    """7주차 Evidence-Gated Meaning Score: 100·S^0.2·C^0.4·T^0.4·(1−R), claim이 없으면 100·S."""
    components = calculate_meaning_components(verification)
    s, c, t, r = (components[k] for k in
                  ("semantic_similarity", "claim_coverage", "translation_support_rate", "contradiction_rate"))
    nuance = verification.get("semantic_nuance") or {}
    nuance_error = bool(nuance.get("has_meaning_error", False)) or any(
        reason.get("type") == "SEMANTIC_NUANCE" for reason in verification.get("failure_reasons", []))

    if s is None:
        return {**components, "meaning_score": None, "status": "UNAVAILABLE",
                "score_method": "UNAVAILABLE", "semantic_nuance_error": nuance_error}
    s = _clamp(s)
    if any(v is None for v in (c, t, r)):
        return {**components, "meaning_score": round(100.0 * s, 2),
                "status": "SEMANTIC_ERROR_DETECTED" if nuance_error else "SIMILARITY_ONLY",
                "score_method": "SIMILARITY_ONLY", "semantic_nuance_error": nuance_error}
    score = 100.0 * (s ** 0.2) * (_clamp(c) ** 0.4) * (_clamp(t) ** 0.4) * (1.0 - _clamp(r))
    return {**components, "meaning_score": round(score, 2),
            "status": "SEMANTIC_ERROR_DETECTED" if nuance_error else "OK",
            "score_method": "CLAIM_BASED", "semantic_nuance_error": nuance_error}
