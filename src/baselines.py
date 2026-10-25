"""7주차 평가용 비교 조건(baseline) A, B.

C(우리 팀 방법)는 pipeline.translate_with_verification()을 그대로 씁니다.
A와 B는 C와 같은 생성 모델(config.GENERATION_MODEL)을 써야 공정한 비교가 되므로,
여기서도 같은 클라이언트·모델·재시도 설정을 공유합니다.

- A. Zero-Shot 기본 번역      : 한국어 원문만 넣고 번역 (성능 하한선)
- B. In-Context 예시 기반 번역 : 원문 + RAG로 찾은 사용자 예시 k개만 넣고 번역
                                (Style Profile, NLI 검증·재번역 없음)
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
    """Condition A: 개인화 정보 없이 일반 번역."""
    prompt = f"""Translate the following Korean text into natural, fluent English.
Output only the translation.

Korean text:
{korean_text}

English translation:"""
    return _generate(prompt, label="baseline A 생성 호출")


def translate_in_context(korean_text: str, user: str, k: int = config.DEFAULT_TOP_K) -> str:
    """Condition B: 사용자 과거 글 예시 k개만 프롬프트에 넣는 단순 개인화 번역."""
    examples = retrieve_similar(korean_text, user, k=k)
    prompt = f"""Translate the following Korean text into English in the same style as the writer's examples below.
Output only the translation.

Writer's examples:
{_format_examples(examples)}

Korean text:
{korean_text}

English translation:"""
    return _generate(prompt, label="baseline B 생성 호출")
