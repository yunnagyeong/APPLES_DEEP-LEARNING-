"""의미 보존 검증 v2 — claim 정렬 판정 + 규칙 기반 안전장치.

왜 바꿨나 (자세한 근거: docs/PIPELINE_V2.md)
- 7주차 레거시 검증(src/verify_legacy.py)은 claim 추출 → 다국어 NLI(mDeBERTa) → Gemini 뉘앙스 점검이었다.
  사람 라벨 32건(error_cases.csv)에서 NLI는 오류 10건 중 7건을 잡았는데, 그 7건은 모두 Gemini 뉘앙스
  점검도 잡았다(NLI만 잡은 오류 0건). 반대로 NLI는 정상 번역 #26을 잘못 걸렀고, 양태("수 있다→causes")·
  정도("크지 않다→negligible")·시제 변화는 entailment로 통과시켰다. 원문(한국어)↔번역문(영어) 교차 언어
  NLI라 구조적으로 약한 부분이다. 그러면서 torch·transformers(수 GB)와 모델 다운로드가 필요했다.
- 그래서 판정은 Gemini 호출 1번으로 하되, "AI가 AI를 평가하는" 주관성을 줄이도록 설계했다.
  ① 판정 모델은 점수나 심각도를 매기지 않는다. 원문 claim마다 번역문의 대응 구절(evidence)을 인용하고,
     정해진 오류 유형 중 하나를 고르기만 한다(분류 문제).
  ② 유형 → 심각도 → 판정(verdict)·점수는 이 파일의 고정 규칙(코드)이 정한다.
  ③ 판정 모델과 무관한 규칙 검사 두 가지를 덧붙인다: 숫자 일치 검사, 임베딩 유사도 하한.

반환 dict (pipeline·app·meaning_score가 사용)
  verdict            PASS | REVIEW | FAIL
  meaning_score      0~100 (src/meaning_score.py)
  source_claims      [{claim, evidence, label, severity, note}]
  additions          [{text, label, severity, note}]
  issues             OK가 아닌 모든 항목 [{type, severity, origin, claim, evidence, note}]
  failure_reasons    issues 중 major·critical (재번역 피드백에 사용)
  critical_errors    issues 중 critical
  semantic_similarity, judge_model
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config, llm
from .embeddings import embed_text, embed_texts

# ---------------------------------------------------------------- 오류 유형과 심각도 (고정 규칙)
# critical: 의미가 뒤집히거나 없는 사실·근거가 생김 → FAIL
# major   : 확신도·주관성·범위·조건이 바뀌거나 내용이 빠짐 → REVIEW (재번역 대상)
# minor   : 시제·정도·군더더기 단어 → PASS (참고 표시만, 재번역 안 함)
SEVERITY = {
    "OK": "none",
    "TENSE_SHIFT": "minor",
    "INTENSITY_SHIFT": "minor",
    "MINOR_ADDITION": "minor",
    "MODALITY_SHIFT": "major",
    "ATTRIBUTION_LOST": "major",
    "SCOPE_CHANGE": "major",
    "CONDITION_DROPPED": "major",
    "OMISSION": "major",
    "NEGATION_FLIP": "critical",
    "NUMBER_CHANGE": "critical",
    "ENTITY_CHANGE": "critical",
    "COMPARISON_REVERSED": "critical",
    "CAUSAL_CHANGE": "critical",
    "CONTRADICTION": "critical",
    "UNSUPPORTED_ADDITION": "critical",
    # 규칙 검사(판정 모델과 별개)
    "NUMBER_CHECK": "major",
    "LOW_SIMILARITY": "major",
    "UNKNOWN_LABEL": "major",
}
CLAIM_LABELS = [k for k in SEVERITY if k not in ("MINOR_ADDITION", "UNSUPPORTED_ADDITION",
                                                  "NUMBER_CHECK", "LOW_SIMILARITY", "UNKNOWN_LABEL")]
ADDITION_LABELS = ["MINOR_ADDITION", "UNSUPPORTED_ADDITION"]
SEVERITY_RANK = {"none": 0, "minor": 1, "major": 2, "critical": 3}

ISSUE_KO = {
    "TENSE_SHIFT": "시제 변경", "INTENSITY_SHIFT": "정도 표현 변화", "MINOR_ADDITION": "사소한 단어 추가",
    "MODALITY_SHIFT": "확신도·가능성 변화", "ATTRIBUTION_LOST": "주관적 표현(생각한다 등) 변화",
    "SCOPE_CHANGE": "범위 변화", "CONDITION_DROPPED": "조건 누락·변경", "OMISSION": "내용 누락",
    "NEGATION_FLIP": "부정 반전", "NUMBER_CHANGE": "숫자 변경", "ENTITY_CHANGE": "고유명사 변경",
    "COMPARISON_REVERSED": "비교 방향 반전", "CAUSAL_CHANGE": "인과관계 변경", "CONTRADICTION": "의미 반전",
    "UNSUPPORTED_ADDITION": "원문에 없는 내용 추가", "NUMBER_CHECK": "숫자 불일치(규칙 검사)",
    "LOW_SIMILARITY": "전체 의미 유사도 낮음(규칙 검사)", "UNKNOWN_LABEL": "판정 불명확",
}


# ---------------------------------------------------------------- 판정 프롬프트
_JUDGE_PROMPT = """You are a meticulous bilingual Korean-to-English translation verifier.
Your ONLY job is meaning fidelity. The translation was deliberately rewritten in a specific person's
writing style, so differences in wording, register, sentence rhythm and word choice are EXPECTED and are NOT errors.

STEP 1 - Source claims.
{claim_step}

STEP 2 - Align each source claim to the translation.
Quote the exact span of the translation that expresses it as "evidence" ("" if absent), then choose ONE label:
- OK: same meaning (paraphrase, synonym, register or style change is fine)
- TENSE_SHIFT: only the time reference changed (past/present/future)
- INTENSITY_SHIFT: degree slightly stronger/weaker, claim otherwise the same (e.g. "not large" -> "negligible")
- MODALITY_SHIFT: certainty or possibility changed (e.g. "may/might/seems/could" <-> "is/will/does/must/inevitably"),
  including a hedge removed or added
- ATTRIBUTION_LOST: subjective framing ("I think", "in my opinion", "personally", "it seems to me") removed,
  turned into objective fact, or added where the source had none
- SCOPE_CHANGE: quantity or scope changed (some -> all, often -> always, adding/removing "only")
- CONDITION_DROPPED: a condition or restriction was removed or changed
- OMISSION: the claim is missing from the translation
- NEGATION_FLIP: negation reversed
- NUMBER_CHANGE: a number, percentage, date or quantity changed
- ENTITY_CHANGE: a name, place, organization or technical term changed into a different one
- COMPARISON_REVERSED: comparison direction changed
- CAUSAL_CHANGE: cause and effect reversed, correlation turned into causation, or a causal link invented
- CONTRADICTION: any other reversal of meaning

STEP 3 - Additions.
List every span of the translation that adds content not present in the source:
- MINOR_ADDITION: a word that adds no new proposition (intensifier, connective, clarifying word), e.g. "simply", "accordingly"
- UNSUPPORTED_ADDITION: a new proposition, reason, explanation, justification, example or fact
Do not list pure style changes. If nothing was added, return an empty list.

Write every "note" as one short sentence in Korean explaining the problem ("" for OK).
Return ONLY JSON:
{{"source_claims": [{{"claim": "...", "evidence": "...", "label": "OK", "note": ""}}],
  "additions": [{{"text": "...", "label": "UNSUPPORTED_ADDITION", "note": "..."}}]}}

Korean source:
{source}

English translation:
{translation}
"""

_SPLIT_STEP = """Split the Korean source into atomic claims: one proposition each, written in Korean.
Keep negation, numbers, modality and hedges (수 있다, 것 같다, 보인다), conditions, comparisons and
attribution (생각한다, 내 생각엔) inside the claim they belong to. A command or request is one claim.
Never return zero claims."""

_FIXED_STEP = """The source claims are already fixed. Use exactly these claims, in this order, without changing them:
{claims}"""


def _judge_prompt(source: str, translation: str, fixed_claims: list[str] | None) -> str:
    if fixed_claims:
        step = _FIXED_STEP.format(claims="\n".join(f"- {c}" for c in fixed_claims))
    else:
        step = _SPLIT_STEP
    return _JUDGE_PROMPT.format(claim_step=step, source=source.strip(), translation=translation.strip())


def _validate_judgement(result: dict) -> None:
    claims = result.get("source_claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("판정 결과에 source_claims가 없음")
    if not isinstance(result.get("additions", []), list):
        raise ValueError("additions 형식 오류")


def judge(source: str, translation: str, fixed_claims: list[str] | None = None) -> tuple[dict, str]:
    """판정 모델 호출. (판정 dict, 실제 사용 모델) 반환."""
    return llm.generate(
        _judge_prompt(source, translation, fixed_claims),
        model=config.VERIFY_MODEL,
        fallback_model=config.VERIFY_FALLBACK_MODEL,
        thinking_level=config.VERIFY_THINKING_LEVEL,
        json_mode=True,
        label="meaning judge",
        validate=_validate_judgement,
    )


# ---------------------------------------------------------------- 규칙 검사: 숫자
_EN_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000,
    "half": 0.5, "double": 2, "twice": 2, "single": 1, "dozen": 12,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
}
# 달 이름(3월 → March). 조동사 may와 헷갈리지 않게 대문자로 시작할 때만 인정
_EN_MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                           "August", "September", "October", "November", "December"], 1)}
_KO_NATIVE = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5, "여섯": 6,
              "일곱": 7, "여덟": 8, "아홉": 9, "열": 10, "스무": 20, "스물": 20}
_KO_COUNTERS = "가지|개|명|번|살|시간|달|해|권|편|장|차례|배|단계|대|군데|마리|주|일|년|분|초|층|줄|쪽|부분"
_KO_NATIVE_RE = re.compile(r"(?<![가-힣])(" + "|".join(sorted(_KO_NATIVE, key=len, reverse=True)) +
                           r")\s?(?:" + _KO_COUNTERS + r")")
_KO_UNIT = {"천": 1_000, "만": 10_000, "억": 100_000_000, "조": 1_000_000_000_000}
_NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def _norm(x: float) -> str:
    return f"{x:g}"


_KO_ORDINAL_RE = re.compile(r"제\s?(\d+)|(\d+)\s?차(?![가-힣]*량)")  # 제1차, 2차 같은 서수


def _ko_ordinals(text: str) -> set[str]:
    """서수(제1차 세계대전, 2차 시험). 영어에서 First/I/1st 등으로 다양하게 쓰여 '원문 숫자 누락' 검사에서 뺌."""
    return {_norm(float(m.group(1) or m.group(2))) for m in _KO_ORDINAL_RE.finditer(text)}


def _ko_numbers(text: str) -> set[str]:
    out = set()
    for m in re.finditer(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s?([천만억조])?", text):
        v = float(m.group(1).replace(",", ""))
        out.add(_norm(v))
        if m.group(2):
            out.add(_norm(v * _KO_UNIT[m.group(2)]))
    return out


def _ko_native_numbers(text: str) -> set[str]:
    return {_norm(_KO_NATIVE[m.group(1)]) for m in _KO_NATIVE_RE.finditer(text)}


def _en_numbers(text: str) -> set[str]:
    out = set()
    for m in _NUM_RE.finditer(text):
        tail = text[m.end():m.end() + 2].lower()
        if tail in ("st", "nd", "rd", "th"):  # 서수(1st 등)는 원문 표현이 다양해서 제외
            continue
        out.add(_norm(float(m.group(0).replace(",", ""))))
    return out


def _en_number_words(text: str) -> set[str]:
    words = {_norm(v) for w, v in _EN_WORDS.items() if re.search(rf"\b{w}\b", text, re.I)}
    months = {_norm(v) for m, v in _EN_MONTHS.items() if re.search(rf"\b{m}\b", text)}
    return words | months


def check_numbers(source: str, translation: str) -> list[dict]:
    """원문 숫자(아라비아 숫자)가 번역문에 있는지, 번역문에 원문에 없는 숫자가 생겼는지."""
    ko, en = _ko_numbers(source), _en_numbers(translation)
    en_any = en | _en_number_words(translation)
    ordinals = _ko_ordinals(source)
    ko_any = ko | _ko_native_numbers(source)
    issues = []
    missing = sorted(n for n in ko if n not in en_any and n not in ordinals)
    # 원문 단위 표기(1만)와 영어(10,000)처럼 같은 수가 다르게 보이는 경우: 한 묶음 중 하나만 있으면 통과
    for m in re.finditer(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s?([천만억조])", source):
        v = float(m.group(1).replace(",", ""))
        pair = {_norm(v), _norm(v * _KO_UNIT[m.group(2)])}
        if pair & en_any:
            missing = [n for n in missing if n not in pair]
    extra = sorted(n for n in en if n not in ko_any)
    if missing:
        issues.append({"type": "NUMBER_CHECK", "claim": "", "evidence": "",
                       "note": f"원문의 숫자 {', '.join(missing)}이(가) 번역문에서 보이지 않습니다."})
    if extra:
        issues.append({"type": "NUMBER_CHECK", "claim": "", "evidence": "",
                       "note": f"번역문에 원문에 없는 숫자 {', '.join(extra)}이(가) 있습니다."})
    return issues


# ---------------------------------------------------------------- 검증
def cosine_similarity(vec_a, vec_b) -> float:
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _issue(kind: str, origin: str, *, claim="", evidence="", note="") -> dict:
    return {"type": kind, "severity": SEVERITY.get(kind, "major"), "origin": origin,
            "label_ko": ISSUE_KO.get(kind, kind), "claim": claim, "evidence": evidence, "note": note}


def build_issues(judgement: dict, rule_issues: list[dict], similarity: float | None) -> tuple[list, list, list]:
    """판정 결과 + 규칙 검사 → (source_claims, additions, issues). 심각도는 SEVERITY 표로만 정함."""
    source_claims, additions, issues = [], [], []
    for c in judgement.get("source_claims", []):
        label = str(c.get("label", "")).strip().upper()
        if label not in CLAIM_LABELS:
            label = "UNKNOWN_LABEL"
        row = {"claim": str(c.get("claim", "")), "evidence": str(c.get("evidence", "") or ""),
               "label": label, "severity": SEVERITY[label], "note": str(c.get("note", "") or "")}
        source_claims.append(row)
        if label != "OK":
            issues.append(_issue(label, "judge", claim=row["claim"], evidence=row["evidence"], note=row["note"]))
    for a in judgement.get("additions", []) or []:
        label = str(a.get("label", "")).strip().upper()
        if label not in ADDITION_LABELS:
            label = "UNSUPPORTED_ADDITION" if label else "MINOR_ADDITION"
        row = {"text": str(a.get("text", "")), "label": label, "severity": SEVERITY[label],
               "note": str(a.get("note", "") or "")}
        additions.append(row)
        issues.append(_issue(label, "judge", evidence=row["text"], note=row["note"]))
    judge_flagged_numbers = any(c["label"] == "NUMBER_CHANGE" for c in source_claims)
    for r in rule_issues:
        if r["type"] == "NUMBER_CHECK" and judge_flagged_numbers:
            continue  # 판정 모델이 이미 숫자 변경으로 잡음
        issues.append(_issue(r["type"], "rule", note=r["note"]))
    if similarity is not None and similarity < config.SIMILARITY_GUARD:
        issues.append(_issue("LOW_SIMILARITY", "rule",
                             note=f"원문·번역문 임베딩 유사도 {similarity:.2f} < {config.SIMILARITY_GUARD}"))
    return source_claims, additions, issues


def decide_verdict(issues: list[dict]) -> str:
    worst = max((SEVERITY_RANK[i["severity"]] for i in issues), default=0)
    return "FAIL" if worst >= 3 else "REVIEW" if worst == 2 else "PASS"


def verify_translation(source_text: str, translated_text: str, source_cache: dict | None = None) -> dict:
    """번역 검증. source_cache(빈 dict)를 넘기면 원문 claim·원문 임베딩을 재번역 시도 간 재사용합니다
    (시도마다 같은 원문 claim으로 판정해야 시도끼리 점수를 공정하게 비교할 수 있음)."""
    from .meaning_score import calculate_meaning_score

    cache = source_cache if source_cache is not None else {}

    def _similarity():
        try:
            if "embedding" in cache:
                return cosine_similarity(cache["embedding"], embed_text(translated_text))
            src, tr = embed_texts([source_text, translated_text])
            cache["embedding"] = src
            return cosine_similarity(src, tr)
        except Exception as e:  # 유사도는 보조 지표라 실패해도 판정은 계속
            print(f"[verify] 유사도 계산 실패, 건너뜀: {e}")
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_sim = pool.submit(_similarity)
        f_judge = pool.submit(judge, source_text, translated_text, cache.get("claims"))
        similarity = f_sim.result()
        judgement, judge_model = f_judge.result()

    rule_issues = check_numbers(source_text, translated_text)
    source_claims, additions, issues = build_issues(judgement, rule_issues, similarity)
    cache.setdefault("claims", [c["claim"] for c in source_claims])

    result = {
        "verdict": decide_verdict(issues),
        "source_claims": source_claims,
        "additions": additions,
        "issues": issues,
        "failure_reasons": [i for i in issues if SEVERITY_RANK[i["severity"]] >= 2],
        "critical_errors": [i for i in issues if i["severity"] == "critical"],
        "semantic_similarity": similarity,
        "judge_model": judge_model,
    }
    result.update(calculate_meaning_score(result))
    return result
