"""문체 특징 추출 (규칙/통계 기반).

docs/style_profile_spec.md에서 정의한 StyleProfile 스키마를 실제로 계산하는 모듈입니다.

의도적으로 LLM을 쓰지 않습니다 — 스타일 수치를 LLM 주관 판단에 맡기면 의미 검증 단계와
같은 "AI가 AI를 평가하는" 약점이 반복되기 때문입니다 (docs/style_profile_spec.md 참고).

형태소 분석/문장 분리는 Kiwi(kiwipiepy)를 씁니다: pip install kiwipiepy
"""

from __future__ import annotations

import re
import sys

from kiwipiepy import Kiwi

# 팀 회의에서 확정된 표현으로 계속 보강해나가는 리스트.
LOGICAL_CONNECTIVES = [
    "그러므로", "따라서", "하지만", "그러나", "왜냐하면",
    "즉", "반면에", "그리고", "결과적으로", "그럼에도",
]

HEDGING_EXPRESSIONS = [
    "것 같다", "수 있다", "수도 있다", "일 수 있다",
    "아마", "대체로", "인 듯하다", "라고 볼 수 있다",
]

EXAMPLE_MARKERS = [
    "예를 들어", "가령", "예컨대", "예시로",
]

CONCLUSION_MARKERS = ["결론적으로", "요컨대", "따라서"]

# 격식체/단정형 종결어미 판별용 정규식. 실제 데이터를 보면서 다듬는다.
FORMAL_ENDING_PATTERN = re.compile(r"(습니다|합니다|입니다|됩니다)\.?$")
DIRECT_ENDING_PATTERN = re.compile(r"(이다|한다|하다)\.?$")

# lexical_diversity 계산에 쓸 "내용어" 품사 태그 (Kiwi 태그셋 기준).
# NNG/NNP: 일반/고유명사, VV: 동사, VA: 형용사, MAG: 일반부사, XR: 어근.
CONTENT_TAGS = {"NNG", "NNP", "VV", "VA", "MAG", "XR"}

# 기능어 선택 프로필(투두 24번, D-003 B 작업 3): 같은 기능(대조·결론·이유·목적)을 한국어에서
# 어떤 표현으로 쓰는지의 빈도. 값은 Kiwi 형태소 1000개당 등장 횟수. 규칙은 (형태, 품사 태그).
# 그룹마다 영어 지시문(generate.py FUNCTION_WORD_INSTRUCTIONS)이 대응한다. 주제 의존 형태소
# (시/NNB, 재/XPN 등)와 영어에서 선택지가 없는 것(와/과 병렬, 것 구문, 인용 -다고)은 뺐다.
FUNCTION_WORD_GROUPS: dict[str, list[tuple[str, str]]] = {
    "however": [("그러나", "MAJ")],
    "but": [("하지만", "MAJ"), ("그렇지만", "MAJ")],
    "whereas": [("반면", "NN")],
    "therefore": [("따라서", "MAJ"), ("그러므로", "MAJ")],
    "causal": [("므로", "EC"), ("으므로", "EC"), ("어서", "EC"), ("아서", "EC"), ("그래서", "MAJ")],
    "purpose_formal": [("고자", "EC")],
}

# 기능어 지시문 적용 조건(D-013): 입력 한국어에 아래 표현이 실제로 있을 때만 그 그룹의 영어 줄을 지시문에
# 넣는다(generate._style_profile_to_instructions의 source_text). 지표(FUNCTION_WORD_GROUPS)와 달리
# ① 대조(however/but)는 그러나·하지만·그렇지만 어느 것이든(같은 기능이라 선택을 정하는 줄이 필요),
# ② -어서·-아서는 뺀다(인과와 시간 순서가 섞여 "원문에 이유가 있다"를 장담할 수 없음). 지표 계산은 바꾸지 않았다.
_CONTRAST_GATE = [("그러나", "MAJ"), ("하지만", "MAJ"), ("그렇지만", "MAJ")]
FUNCTION_WORD_GATE: dict[str, list[tuple[str, str]]] = {
    "however": _CONTRAST_GATE,
    "but": _CONTRAST_GATE,
    "whereas": [("반면", "NN")],
    "therefore": [("따라서", "MAJ"), ("그러므로", "MAJ")],
    "causal": [("므로", "EC"), ("으므로", "EC"), ("그래서", "MAJ")],
    "purpose_formal": [("고자", "EC")],
}

_kiwi: Kiwi | None = None


def _get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        print("[style_features] Kiwi 형태소 분석기 초기화 중...", file=sys.stderr)
        _kiwi = Kiwi()
    return _kiwi


def split_sentences(text: str) -> list[str]:
    """Kiwi로 문장을 분리한다."""
    if not text or not text.strip():
        return []
    kiwi = _get_kiwi()
    return [s.text.strip() for s in kiwi.split_into_sents(text) if s.text.strip()]


def split_paragraphs(text: str) -> list[str]:
    """빈 줄(문단 구분) 기준으로 문단을 나눈다."""
    if not text:
        return []
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def sentence_length_avg(sentences: list[str]) -> float:
    """평균 어절 수. docs/style_profile_spec.md 1번 항목."""
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def formality_score(sentences: list[str]) -> float:
    """격식체 종결 문장 비율. docs/style_profile_spec.md 2번 항목."""
    if not sentences:
        return 0.0
    matched = sum(1 for s in sentences if FORMAL_ENDING_PATTERN.search(s))
    return matched / len(sentences)


def directness_score(sentences: list[str]) -> float:
    """단정형 종결 문장 비율. docs/style_profile_spec.md 3번 항목."""
    if not sentences:
        return 0.0
    matched = sum(1 for s in sentences if DIRECT_ENDING_PATTERN.search(s))
    return matched / len(sentences)


def _marker_rate(sentences: list[str], markers: list[str]) -> float:
    if not sentences:
        return 0.0
    count = sum(s.count(m) for s in sentences for m in markers)
    return count / len(sentences)


def logical_connective_rate(sentences: list[str]) -> float:
    """문장당 접속 표현 개수. docs/style_profile_spec.md 4번 항목."""
    return _marker_rate(sentences, LOGICAL_CONNECTIVES)


def hedging_rate(sentences: list[str]) -> float:
    """문장당 hedging 표현 개수. docs/style_profile_spec.md 5번 항목."""
    return _marker_rate(sentences, HEDGING_EXPRESSIONS)


def example_usage_rate(sentences: list[str]) -> float:
    """문장당 예시 표지어 개수. spec 추가 권장 항목."""
    return _marker_rate(sentences, EXAMPLE_MARKERS)


def conclusion_position(paragraphs: list[str]) -> str:
    """결론 표지어가 처음 등장하는 문단 위치. "front" | "back" | "mixed".

    spec 6번 항목: 결론 표지어가 등장하는 첫 문단이 전체의 앞쪽 1/3이면 "front",
    뒤쪽 1/3이면 "back", 그 사이거나 아예 없으면 "mixed"로 분류한다.
    """
    if not paragraphs:
        return "mixed"
    n = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if any(marker in p for marker in CONCLUSION_MARKERS):
            if i < n / 3:
                return "front"
            if i >= 2 * n / 3:
                return "back"
            return "mixed"
    return "mixed"


def calculate_mtld(tokens: list[str], ttr_threshold: float = 0.72) -> float:
    """MTLD (Measure of Textual Lexical Diversity). evaluate_metrics.py의 calculate_mtld와 같은 알고리즘.

    토큰을 앞에서부터 읽으며 TTR이 ttr_threshold 아래로 떨어질 때마다 한 구간(factor)을 끊고
    새로 시작한다. 결과는 "전체 토큰 수 / 구간 수", 즉 새 단어가 반복 없이 평균 몇 토큰이나
    이어지는지다. 앞→뒤, 뒤→앞 두 방향의 평균을 쓴다. 값이 클수록 어휘가 다양하다.
    TTR과 달리 글 길이에 거의 영향을 받지 않는다. 토큰이 10개 미만이면 0.0.
    """
    if len(tokens) < 10:
        return 0.0

    def _factor_mean(token_list: list[str]) -> float:
        factors = 0.0
        current_types: set[str] = set()
        token_count = 0
        for token in token_list:
            token_count += 1
            current_types.add(token)
            if len(current_types) / token_count <= ttr_threshold:
                factors += 1.0
                current_types = set()
                token_count = 0
        if token_count > 0:  # 끝까지 못 채운 마지막 구간은 비율로 보정 (McCarthy & Jarvis)
            current_ttr = len(current_types) / token_count
            if current_ttr < 1.0:
                factors += (1.0 - current_ttr) / (1.0 - ttr_threshold)
        return len(token_list) / factors if factors > 0 else float(len(token_list))

    return round((_factor_mean(tokens) + _factor_mean(tokens[::-1])) / 2.0, 2)


def lexical_diversity(sentences: list[str]) -> float:
    """MTLD, 내용어(명사/동사/형용사/부사/어근) 기준. 키 이름은 호환을 위해 lexical_diversity 유지.

    예전에는 TTR(0~1)이었으나, TTR은 글이 길수록 기계적으로 떨어져 코퍼스 길이가 다른 사람끼리
    비교할 수 없어서 MTLD로 교체했다(투두 5번). 값의 범위가 0~1이 아니라 대략 수십 단위이므로
    generate.py의 임계값(LEXICAL_DIVERSITY_*)도 이 척도 기준이어야 한다. 7주차 AI 획일화
    평가(evaluate_metrics.py)의 어휘 다양성과 같은 알고리즘이다.
    """
    if not sentences:
        return 0.0
    kiwi = _get_kiwi()
    full_text = " ".join(sentences)
    tokens = kiwi.tokenize(full_text)
    lemmas = [t.form for t in tokens if t.tag in CONTENT_TAGS]
    if not lemmas:
        return 0.0
    return calculate_mtld(lemmas)


def paragraph_length_avg(paragraphs: list[str]) -> float:
    """문단당 평균 문장 수. spec 추가 권장 항목."""
    if not paragraphs:
        return 0.0
    counts = [len(split_sentences(p)) for p in paragraphs]
    return sum(counts) / len(counts)


def count_function_word_groups(tokens) -> dict[str, int]:
    """Kiwi 토큰 리스트에서 FUNCTION_WORD_GROUPS 그룹별 등장 횟수."""
    counts = {g: 0 for g in FUNCTION_WORD_GROUPS}
    for t in tokens:
        for g, rules in FUNCTION_WORD_GROUPS.items():
            if any(t.form == f and t.tag.startswith(tag) for f, tag in rules):
                counts[g] += 1
    return counts


def count_gate_groups(tokens) -> dict[str, int]:
    """Kiwi 토큰에서 FUNCTION_WORD_GATE(지시문 적용 조건) 그룹별 등장 횟수."""
    counts = {g: 0 for g in FUNCTION_WORD_GATE}
    for t in tokens:
        for g, rules in FUNCTION_WORD_GATE.items():
            if any(t.form == f and t.tag.startswith(tag) for f, tag in rules):
                counts[g] += 1
    return counts


def source_gate_groups(text: str) -> set[str]:
    """입력 한국어에 적용 조건 표현이 실제로 있는 기능어 그룹 집합."""
    return {g for g, c in count_gate_groups(_get_kiwi().tokenize(text)).items() if c}


def function_word_rates(texts: list[str]) -> dict[str, float]:
    """여러 글을 합쳐 그룹별 기능어 빈도(형태소 1000개당)를 계산한다."""
    kiwi = _get_kiwi()
    totals = {g: 0 for g in FUNCTION_WORD_GROUPS}
    n = 0
    for text in texts:
        toks = kiwi.tokenize(text)
        n += len(toks)
        for g, c in count_function_word_groups(toks).items():
            totals[g] += c
    return {g: round(1000 * c / n, 4) if n else 0.0 for g, c in totals.items()}


def extract_style_profile(texts: list[str]) -> dict:
    """여러 원문(과거 과제물 등)을 받아 StyleProfile 딕셔너리를 반환한다.

    반환 스키마는 docs/style_profile_spec.md의 StyleProfile 초안과 반드시 일치해야
    한다 — 팀원②·③의 코드가 이 키 이름을 그대로 참조하므로, 이름을 바꿀 일이 있으면
    반드시 팀 회의에서 먼저 공유할 것.
    """
    all_sentences: list[str] = []
    all_paragraphs: list[str] = []
    total = len(texts)
    for i, text in enumerate(texts, start=1):
        all_sentences.extend(split_sentences(text))
        all_paragraphs.extend(split_paragraphs(text))
        if i % 10 == 0 or i == total:
            print(f"[style_features] 예시 처리 중: {i}/{total}", file=sys.stderr)

    return {
        "sentence_length_avg": sentence_length_avg(all_sentences),
        "formality_score": formality_score(all_sentences),
        "directness_score": directness_score(all_sentences),
        "logical_connective_rate": logical_connective_rate(all_sentences),
        "hedging_rate": hedging_rate(all_sentences),
        "conclusion_position": conclusion_position(all_paragraphs),
        "example_usage_rate": example_usage_rate(all_sentences),
        "lexical_diversity": lexical_diversity(all_sentences),
        "paragraph_length_avg": paragraph_length_avg(all_paragraphs),
        "function_word_choice": function_word_rates(texts),
        "source_example_count": len(texts),
    }
