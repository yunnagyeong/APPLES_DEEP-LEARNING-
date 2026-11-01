"""비교 조건 A·B 번역기 (app.py, 7주차 A/B/C 비교용).

A · 일반 번역      : 예시·문체 지시·검증 없이 그대로 번역 (zero-shot)
B · 단순 개인화(RAG): 검색한 본인 한국어 예시만 프롬프트에 넣음. 문체 프로필 지시·보존 규칙·검증 없음
C · 우리 팀 방법    : src.pipeline.translate_with_verification (프로필 + RAG + 보존 규칙 + 검증·재번역)

A·B는 생성 모델 호출 1번(+ B는 검색 임베딩 1번)이라 C보다 훨씬 빠릅니다.
"""

from . import config
from .generate import _ensure_client, _format_examples
from .retrieve import retrieve_similar
from .retry import call_with_retry


def _generate(prompt: str, label: str) -> str:
    client = _ensure_client()
    response = call_with_retry(
        lambda: client.models.generate_content(model=config.GENERATION_MODEL, contents=prompt),
        label=label,
    )
    return response.text.strip()


def translate_zero_shot(korean_text: str) -> str:
    """조건 A: 일반 번역."""
    prompt = f"""Translate the following Korean text into natural, fluent English.

Korean text:
{korean_text}

Output only the English translation, with no preface, notes, or quotation marks.

English translation:"""
    return _generate(prompt, "baseline A 생성 호출")


def translate_in_context(korean_text: str, user: str, examples: list[dict] | None = None) -> str:
    """조건 B: 검색한 예시만 넣은 단순 개인화. examples를 주면 검색을 건너뜁니다."""
    if examples is None:
        examples = retrieve_similar(korean_text, user, k=config.DEFAULT_TOP_K)
    prompt = f"""Translate the following Korean text into English in the writer's personal style.

Reference examples of the writer's Korean style (to understand tone and approach):
{_format_examples(examples)}

Korean text:
{korean_text}

Output only the English translation, with no preface, notes, or quotation marks.

English translation:"""
    return _generate(prompt, "baseline B 생성 호출")
