import json
import numpy as np
from google import genai
import torch
from .embeddings import embed_text
from .retry import call_with_retry
from transformers import AutoTokenizer, AutoModelForSequenceClassification


client = genai.Client()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)

    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )


def semantic_similarity(source_text: str, translated_text: str) -> float:
    source_vector = embed_text(
        source_text,
        task_type="RETRIEVAL_DOCUMENT"
    )

    translated_vector = embed_text(
        translated_text,
        task_type="RETRIEVAL_DOCUMENT"
    )

    return cosine_similarity(source_vector, translated_vector)


def extract_claims(text: str, language: str = "ko") -> list[str]:
    prompt = f"""
You are extracting atomic claims from text.

Rules:
1. Extract only claims explicitly supported by the input.
2. Do not add new information.
3. Preserve negation, numbers, modality, conditions, and comparisons.
4. Each claim should contain only one independently verifiable proposition.
5. Return only JSON in this format:
{{"claims": ["claim 1", "claim 2"]}}

Language of the input: {language}

Text:
{text}
"""

    response = call_with_retry(
        lambda: client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        ),
        label=f"claim extraction ({language})"
    )

    raw_text = response.text.strip()

    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]

    if raw_text.startswith("```"):
        raw_text = raw_text[3:]

    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]

    raw_text = raw_text.strip()

    result = json.loads(raw_text)

    return result["claims"]


NLI_MODEL_NAME = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
)

nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL_NAME
)


def nli_check(premise: str, hypothesis: str) -> dict:
    inputs = nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = nli_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    labels = nli_model.config.id2label

    scores = {
        labels[i]: probabilities[i].item()
        for i in range(len(probabilities))
    }

    best_label = max(scores, key=scores.get)

    return {
        "label": best_label,
        "confidence": scores[best_label],
        "scores": scores
    }


def verify_claims(source_text: str, translated_text: str) -> dict:
    source_claims = extract_claims(
        source_text,
        language="ko"
    )

    translated_claims = extract_claims(
        translated_text,
        language="en"
    )

    source_results = []
    translated_results = []

    # Direction 1
    # 번역문이 원문의 Claim을 지원하는가?
    # -> Claim Coverage / Omission 검사
    for claim in source_claims:
        result = nli_check(
            translated_text,
            claim
        )

        source_results.append({
            "claim": claim,
            "label": result["label"],
            "confidence": result["confidence"]
        })

    # Direction 2
    # 원문이 번역문의 Claim을 지원하는가?
    # -> Unsupported Addition / 과장 검사
    for claim in translated_claims:
        result = nli_check(
            source_text,
            claim
        )

        translated_results.append({
            "claim": claim,
            "label": result["label"],
            "confidence": result["confidence"]
        })

    return {
        "source_claims": source_results,
        "translation_claims": translated_results
    }


def build_failure_reasons(claim_results: dict) -> list[dict]:
    failure_reasons = []

    # Direction 1: Source -> Translation
    # 원문의 claim이 번역에서 보존되지 않은 경우
    for result in claim_results["source_claims"]:
        if result["label"] != "entailment":

            if result["label"] == "contradiction":
                error_type = "CONTRADICTION"
            else:
                error_type = "OMISSION_OR_UNSUPPORTED"

            failure_reasons.append({
                "type": error_type,
                "direction": "source_to_translation",
                "claim": result["claim"],
                "label": result["label"],
                "confidence": result["confidence"]
            })

    # Direction 2: Translation -> Source
    # 번역의 claim이 원문에서 지원되지 않는 경우
    for result in claim_results["translation_claims"]:
        if result["label"] != "entailment":

            if result["label"] == "contradiction":
                error_type = "CONTRADICTION"
            else:
                error_type = "UNSUPPORTED_ADDITION"

            failure_reasons.append({
                "type": error_type,
                "direction": "translation_to_source",
                "claim": result["claim"],
                "label": result["label"],
                "confidence": result["confidence"]
            })

    return failure_reasons



def verify_translation(
    source_text: str,
    translated_text: str
) -> dict:
    similarity = semantic_similarity(
        source_text, translated_text
    )

    claim_results = verify_claims(
        source_text, translated_text
    )

    failure_reasons = build_failure_reasons(claim_results)

    # 기존 NLI 판정
    nli_verdict, critical_errors = determine_verdict(
        failure_reasons
    )

    # Gemini 보조 검증
    nuance = check_semantic_nuance(
        source_text, translated_text
    )

    # Gemini가 의미 오류를 발견한 경우
    if nuance["has_meaning_error"]:
        failure_reasons.append({
            "type": "SEMANTIC_NUANCE",
            "direction": "source_to_translation",
            "claim": nuance["explanation"],
            "label": "gemini_review",
            "confidence": None,
            "error_types": nuance["error_types"]
        })

    # 기존 FAIL은 유지하고, Gemini가 오류를 발견하면 REVIEW
    if nli_verdict == "FAIL":
        verdict = "FAIL"
    elif failure_reasons:
        verdict = "REVIEW"
    else:
        verdict = "PASS"

    return {
        "semantic_similarity": similarity,
        "source_claims": claim_results["source_claims"],
        "translation_claims": claim_results["translation_claims"],
        "failure_reasons": failure_reasons,
        "critical_errors": critical_errors,
        "verdict": verdict,
        "semantic_nuance": nuance
    }

def determine_verdict(failure_reasons: list[dict]) -> tuple[str, list[dict]]:
    critical_errors = []

    for reason in failure_reasons:
        if reason["type"] == "CONTRADICTION":
            critical_errors.append(reason)

    if critical_errors:
        verdict = "FAIL"
    elif failure_reasons:
        verdict = "REVIEW"
    else:
        verdict = "PASS"

    return verdict, critical_errors


def check_semantic_nuance(
    source_text: str,
    translated_text: str
) -> dict:
    """
    NLI가 놓치기 쉬운 의미 변화 확인.
    특히 헤징, 확신 정도, 시제, 부정, 조건, 수치 등을 검증.
    """

    prompt = f"""
You are a bilingual Korean-English translation evaluator.

Evaluate whether the English translation preserves the
meaning of the Korean source.

Focus ONLY on meaning preservation, not fluency or style.


IMPORTANT EVALUATION PRINCIPLES:

Evaluate semantic equivalence, not word-for-word similarity.

Do NOT flag:
- Natural paraphrasing that preserves the original meaning.
- Minor emphasis or discourse modifiers that do not materially
  change the meaning.
- Words absent from the source that introduce no new factual claim.

DO flag:
- Unsupported factual claims, explanations, or justifications.
- Changes in modality or certainty (may, might, must, inevitably).
- Changes in negation, tense, conditions, comparisons, or numbers
  that alter the original meaning.
- Omissions of meaningful information.

UNSUPPORTED ADDITION CRITERION:

An unsupported addition must introduce materially new semantic
content, not merely a word absent from the source.

Distinguish harmless lexical elaboration from the introduction
of a new proposition, causal explanation, or factual assertion.

When uncertain, do not automatically classify a minor wording
difference as an error.

Check carefully:
1. Modality and certainty (may, might, must, inevitably)
2. Subjectivity and attribution (I think, it seems)
3. Negation
4. Tense and temporal reference
5. Conditions and comparisons
6. Numbers and named entities
7. Unsupported additions and omissions

Return ONLY valid JSON:
{{
  "has_meaning_error": true,
  "error_types": ["MODALITY_SHIFT"],
  "explanation": "Briefly explain the error in Korean."
}}

If no meaning error exists, return:
{{
  "has_meaning_error": false,
  "error_types": [],
  "explanation": ""
}}

Korean source:
{source_text}

English translation:
{translated_text}
"""

    response = call_with_retry(
        lambda: client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        ),
        label="semantic nuance check"
    )

    raw = response.text.strip()

    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]

    result = json.loads(raw.strip())

    return {
        "has_meaning_error": bool(
            result.get("has_meaning_error", False)
        ),
        "error_types": result.get("error_types", []),
        "explanation": result.get("explanation", "")
    }
