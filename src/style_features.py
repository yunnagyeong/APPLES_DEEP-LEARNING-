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


def lexical_diversity(sentences: list[str]) -> float:
    """TTR(Type-Token Ratio), 내용어(명사/동사/형용사/부사/어근) 기준.

    spec 추가 권장 항목 — 7주차 AI 획일화 평가의 어휘 다양성 계산과 같은 방식을
    쓸 수 있도록, 언어 무관하게 "토큰 리스트"를 받는 형태로 별도 유틸을 분리해도 좋다.
    """
    if not sentences:
        return 0.0
    kiwi = _get_kiwi()
    full_text = " ".join(sentences)
    tokens = kiwi.tokenize(full_text)
    lemmas = [t.form for t in tokens if t.tag in CONTENT_TAGS]
    if not lemmas:
        return 0.0
    return len(set(lemmas)) / len(lemmas)


def paragraph_length_avg(paragraphs: list[str]) -> float:
    """문단당 평균 문장 수. spec 추가 권장 항목."""
    if not paragraphs:
        return 0.0
    counts = [len(split_sentences(p)) for p in paragraphs]
    return sum(counts) / len(counts)


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
        "source_example_count": len(texts),
    }
