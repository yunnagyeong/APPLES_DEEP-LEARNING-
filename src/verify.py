"""의미 보존 검증 (Embedding 유사도 + Claim 단위 양방향 NLI + Gemini 뉘앙스 점검).

경량화 (config.VERIFY_SINGLE_CALL = True, 기본값):
- 원문 claim 추출 · 번역문 claim 추출 · 뉘앙스 점검을 Gemini 호출 1번(JSON 응답)으로 묶습니다.
  검증 1회당 3.6-flash 호출 3회 → 1회. 이 모델은 무료 티어 한도(분당 5회·하루 20회)가 빡빡해서
  호출 수가 곧 지연시간(429 대기)과 하루 데모 가능 횟수를 정합니다.
- 재번역 루프에서는 원문 쪽 결과(원문 임베딩·원문 claim)를 source_cache에 담아 재사용합니다.
  원문은 시도마다 같으므로 다시 계산할 이유가 없고, 시도 간 원문 claim이 같아 비교도 공정해집니다.
- 원문·번역문 임베딩은 한 번의 API 호출로 묶고, Gemini 분석 호출과 동시에(스레드) 보냅니다.
- NLI는 모든 (전제, 가설) 쌍을 한 번의 배치로 추론합니다. 모델은 처음 쓸 때 한 번만 로딩합니다.
config.VERIFY_SINGLE_CALL = False면 예전처럼 extract_claims ×2 + check_semantic_nuance로 동작합니다
(7주차 결과 재현용). 판정 규칙(build_failure_reasons / determine_verdict)은 두 방식이 같습니다.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from google import genai
from google.genai import types

from . import config
from .embeddings import embed_text, embed_texts
from .retry import call_with_retry

NLI_MODEL_NAME = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
)

_client = None
_nli = None  # (tokenizer, model)
_nli_lock = threading.Lock()


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GOOGLE_API_KEY or None)
    return _client


def load_nli_model():
    """NLI 모델을 (처음 한 번만) 로딩해 (tokenizer, model)을 반환. app.py에서 미리 불러 둘 때도 사용."""
    global _nli
    with _nli_lock:
        if _nli is None:
            import torch  # noqa: F401  (무거운 import를 실제 사용 시점으로 미룸)
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
            model.eval()
            _nli = (tok, model)
    return _nli


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return json.loads(raw.strip())


def _generate_json(prompt: str, label: str) -> dict:
    """VERIFY_MODEL을 JSON 모드로 호출하고 파싱까지 한 결과. 파싱 실패도 재시도 대상에 포함."""
    def _once():
        response = _get_client().models.generate_content(
            model=config.VERIFY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return _parse_json(response.text)

    return call_with_retry(_once, label=label)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)

    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )


def semantic_similarity(source_text: str, translated_text: str) -> float:
    source_vector, translated_vector = embed_texts(
        [source_text, translated_text], task_type="RETRIEVAL_DOCUMENT"
    )
    return cosine_similarity(source_vector, translated_vector)


# ---------------------------------------------------------------- 예전 방식(호출 3번)

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
    return _generate_json(prompt, label=f"claim extraction ({language})")["claims"]


# 뉘앙스 점검 기준. 예전 방식(check_semantic_nuance)과 묶음 호출(analyze_meaning)이 같은 문구를 씁니다.
_NUANCE_CRITERIA = """Focus ONLY on meaning preservation, not fluency or style.

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
7. Unsupported additions and omissions"""


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

{_NUANCE_CRITERIA}

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
    return _normalize_nuance(_generate_json(prompt, label="semantic nuance check"))


def _normalize_nuance(result: dict) -> dict:
    return {
        "has_meaning_error": bool(result.get("has_meaning_error", False)),
        "error_types": result.get("error_types", []) or [],
        "explanation": result.get("explanation", "") or "",
    }


# ---------------------------------------------------------------- 묶음 호출(호출 1번)

def analyze_meaning(
    source_text: str,
    translated_text: str,
    source_claims: list[str] | None = None,
) -> dict:
    """원문 claim · 번역문 claim · 뉘앙스 점검을 Gemini 호출 1번으로 받습니다.

    source_claims를 주면(재번역 시도) 원문 claim은 다시 뽑지 않고 그대로 씁니다.
    반환: {"source_claims", "translation_claims", "semantic_nuance"}
    """
    if source_claims is None:
        claim_task = """Task 1 - Atomic claims.
Extract atomic claims separately from the Korean source (write them in Korean)
and from the English translation (write them in English)."""
        claim_json = '"source_claims": ["...", "..."],\n  "translation_claims": ["...", "..."],'
    else:
        listed = "\n".join(f"- {c}" for c in source_claims)
        claim_task = f"""Task 1 - Atomic claims.
The Korean source claims are already fixed:
{listed}
Extract atomic claims from the English translation only (write them in English)."""
        claim_json = '"translation_claims": ["...", "..."],'

    prompt = f"""You are a bilingual Korean-English translation evaluator.

{claim_task}
Claim rules:
1. Extract only claims explicitly supported by that text.
2. Do not add new information.
3. Preserve negation, numbers, modality, conditions, and comparisons.
4. Each claim should contain only one independently verifiable proposition.
5. Extract claims from each text independently; do not let one text shape the other's claims.

Task 2 - Meaning check.
Evaluate whether the English translation preserves the meaning of the Korean source.
{_NUANCE_CRITERIA}

Return ONLY valid JSON in this format:
{{
  {claim_json}
  "has_meaning_error": false,
  "error_types": [],
  "explanation": ""
}}
If there is a meaning error, set "has_meaning_error": true, list types such as
"MODALITY_SHIFT", "UNSUPPORTED_ADDITION", "OMISSION", "TENSE_SHIFT" in "error_types",
and briefly explain the error in Korean in "explanation".

Korean source:
{source_text}

English translation:
{translated_text}
"""
    result = _generate_json(prompt, label="meaning analysis (claims + nuance)")
    return {
        "source_claims": source_claims if source_claims is not None else list(result.get("source_claims", [])),
        "translation_claims": list(result.get("translation_claims", [])),
        "semantic_nuance": _normalize_nuance(result),
    }


# ---------------------------------------------------------------- NLI

def nli_check_batch(pairs: list[tuple[str, str]]) -> list[dict]:
    """(전제, 가설) 쌍 여러 개를 한 번의 배치 추론으로 판정. 결과 형식은 nli_check와 같음."""
    if not pairs:
        return []
    import torch

    tok, model = load_nli_model()
    inputs = tok(
        [p for p, _ in pairs],
        [h for _, h in pairs],
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    labels = model.config.id2label

    out = []
    for row in probs:
        scores = {labels[i]: row[i].item() for i in range(len(row))}
        best = max(scores, key=scores.get)
        out.append({"label": best, "confidence": scores[best], "scores": scores})
    return out


def nli_check(premise: str, hypothesis: str) -> dict:
    return nli_check_batch([(premise, hypothesis)])[0]


def _nli_claims(source_text: str, translated_text: str,
                source_claims: list[str], translation_claims: list[str]) -> dict:
    """Direction 1: 번역문이 원문 claim을 지원하는가 (누락)
    Direction 2: 원문이 번역문 claim을 지원하는가 (추가·과장). 두 방향을 한 배치로 추론."""
    pairs = [(translated_text, c) for c in source_claims] + [(source_text, c) for c in translation_claims]
    results = nli_check_batch(pairs)
    n = len(source_claims)

    def _rows(claims, res):
        return [{"claim": c, "label": r["label"], "confidence": r["confidence"]} for c, r in zip(claims, res)]

    return {
        "source_claims": _rows(source_claims, results[:n]),
        "translation_claims": _rows(translation_claims, results[n:]),
    }


def verify_claims(source_text: str, translated_text: str) -> dict:
    """예전 방식의 claim 검증(원문·번역문 claim을 각각 따로 추출)."""
    return _nli_claims(
        source_text, translated_text,
        extract_claims(source_text, language="ko"),
        extract_claims(translated_text, language="en"),
    )


# ---------------------------------------------------------------- 판정

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


def verify_translation(
    source_text: str,
    translated_text: str,
    source_cache: dict | None = None,
) -> dict:
    """번역 검증. 반환 형식은 예전과 같습니다.

    source_cache: 같은 원문을 여러 번 검증할 때(재번역 루프) 넘기는 빈 dict. 첫 호출에서
    원문 임베딩·원문 claim을 채워 두고, 이후 호출에서는 다시 계산하지 않습니다.
    """
    cache = source_cache if source_cache is not None else {}
    single = config.VERIFY_SINGLE_CALL

    def _embeddings():
        if "embedding" in cache:
            return cache["embedding"], embed_text(translated_text, task_type="RETRIEVAL_DOCUMENT")
        src_vec, tr_vec = embed_texts([source_text, translated_text], task_type="RETRIEVAL_DOCUMENT")
        cache["embedding"] = src_vec
        return src_vec, tr_vec

    def _analysis():
        if single:
            return analyze_meaning(source_text, translated_text, cache.get("claims"))
        src_claims = cache.get("claims")
        if src_claims is None:
            src_claims = extract_claims(source_text, language="ko")
        return {
            "source_claims": src_claims,
            "translation_claims": extract_claims(translated_text, language="en"),
            "semantic_nuance": check_semantic_nuance(source_text, translated_text),
        }

    # 임베딩(빠름)과 Gemini 분석(느림)을 동시에 보냄
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_emb = pool.submit(_embeddings)
        f_ana = pool.submit(_analysis)
        src_vec, tr_vec = f_emb.result()
        analysis = f_ana.result()

    cache["claims"] = analysis["source_claims"]
    similarity = cosine_similarity(src_vec, tr_vec)

    claim_results = _nli_claims(
        source_text, translated_text,
        analysis["source_claims"], analysis["translation_claims"],
    )
    failure_reasons = build_failure_reasons(claim_results)

    # 기존 NLI 판정
    nli_verdict, critical_errors = determine_verdict(failure_reasons)

    # Gemini 보조 검증
    nuance = analysis["semantic_nuance"]
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
