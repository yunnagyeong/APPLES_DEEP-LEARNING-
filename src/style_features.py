"""문체 특징 추출 (규칙/통계 기반).

docs/style_profile_spec.md에서 정의한 StyleProfile 스키마를 실제로 계산하는 모듈입니다.
1주차 회의에서 스펙이 확정되면 이 스텁을 채워 넣는 게 2주차 작업입니다.

의도적으로 LLM을 쓰지 않습니다 — 스타일 수치를 LLM 주관 판단에 맡기면 의미 검증 단계와
같은 "AI가 AI를 평가하는" 약점이 반복되기 때문입니다 (docs/style_profile_spec.md 참고).

형태소 분석기는 Kiwi(kiwipiepy)를 씁니다: pip install kiwipiepy
"""

from __future__ import annotations

# TODO(1~2주차): 팀 회의에서 확정된 표현을 여기 채운다.
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

# 격식체/단정형 종결어미 판별용 정규식은 실제 데이터를 보면서 다듬는다.
FORMAL_ENDING_PATTERN = r"(습니다|합니다|입니다|됩니다)\.?$"
DIRECT_ENDING_PATTERN = r"(이다|한다|하다)\.?$"

CONCLUSION_MARKERS = ["결론적으로", "요컨대", "따라서"]


def split_sentences(text: str) -> list[str]:
    """Kiwi로 문장을 분리한다. TODO: kiwipiepy 연동."""
    raise NotImplementedError("2주차: Kiwi 형태소 분석기 연동 후 구현")


def sentence_length_avg(sentences: list[str]) -> float:
    """평균 어절 수. docs/style_profile_spec.md 1번 항목."""
    raise NotImplementedError


def formality_score(sentences: list[str]) -> float:
    """격식체 종결 문장 비율. docs/style_profile_spec.md 2번 항목."""
    raise NotImplementedError


def directness_score(sentences: list[str]) -> float:
    """단정형 종결 문장 비율. docs/style_profile_spec.md 3번 항목."""
    raise NotImplementedError


def logical_connective_rate(sentences: list[str]) -> float:
    """문장당 접속 표현 개수. docs/style_profile_spec.md 4번 항목."""
    raise NotImplementedError


def hedging_rate(sentences: list[str]) -> float:
    """문장당 hedging 표현 개수. docs/style_profile_spec.md 5번 항목."""
    raise NotImplementedError


def conclusion_position(paragraphs: list[str]) -> str:
    """결론 표지어 위치. "front" | "back" | "mixed". spec 6번 항목."""
    raise NotImplementedError


def example_usage_rate(sentences: list[str]) -> float:
    """문장당 예시 표지어 개수. spec 추가 권장 항목."""
    raise NotImplementedError


def lexical_diversity(sentences: list[str]) -> float:
    """TTR(Type-Token Ratio). spec 추가 권장 항목.

    주의: 7주차 AI 획일화 정량 평가에서도 동일한 TTR 계산이 필요하다. 이 함수를
    evaluation 코드에서도 재사용할 수 있도록 언어(한국어/영어) 무관하게 토큰
    리스트를 받는 형태로 설계하는 걸 권장한다 (예: 별도 유틸로 분리해도 좋음).
    """
    raise NotImplementedError


def paragraph_length_avg(paragraphs: list[str]) -> float:
    """문단당 평균 문장 수. spec 추가 권장 항목."""
    raise NotImplementedError


def extract_style_profile(texts: list[str]) -> dict:
    """여러 원문(과거 과제물 등)을 받아 StyleProfile 딕셔너리를 반환한다.

    반환 스키마는 docs/style_profile_spec.md의 StyleProfile 초안과 반드시 일치해야
    한다 — 팀원②·③의 코드가 이 키 이름을 그대로 참조하므로, 이름을 바꿀 일이 있으면
    반드시 팀 회의에서 먼저 공유할 것.
    """
    raise NotImplementedError("1주차 회의에서 스펙 확정 후, 2주차에 구현")
