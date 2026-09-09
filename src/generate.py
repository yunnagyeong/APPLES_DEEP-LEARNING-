"""프롬프트 구성 + 생성 모델 호출.

검색된 예시를 "이 문체를 참고해라"는 형태로 프롬프트에 끼워 넣고, 모델 가중치는 건드리지 않은 채
그 순간의 문맥(In-Context Learning)만으로 결과를 생성합니다.
"""

import sys

from google import genai

from . import config

# 유사도가 이 값보다 낮으면 "참고 예시가 부실하다"고 판단합니다. 데이터가 쌓이면
# 실제 분포를 보고 조정하세요. (0~1 범위, 코사인 유사도 기준)
LOW_SIMILARITY_THRESHOLD = 0.55

_client = None


def _ensure_client() -> genai.Client:
    global _client
    if _client is not None:
        return _client
    if not config.GOOGLE_API_KEY:
        print(
            "[generate] GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.",
            file=sys.stderr,
        )
        sys.exit(1)
    _client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _client


def _format_examples(examples: list[dict]) -> str:
    lines = []
    for i, ex in enumerate(examples, start=1):
        tag = f"[{ex.get('type', '예시')}" + (f" · {ex['topic']}]" if ex.get("topic") else "]")
        lines.append(f"예시{i} {tag}: {ex['text']}")
    return "\n".join(lines)


def _low_confidence_notice(examples: list[dict]) -> str:
    if not examples or examples[0]["score"] < LOW_SIMILARITY_THRESHOLD:
        return (
            "\n[안내] 지금 입력과 비슷한 참고 예시가 부족합니다. 결과가 평소 문체를 "
            "충분히 반영하지 못할 수 있습니다. data/examples/examples.jsonl에 관련 예시를 "
            "추가하면 다음부터 더 잘 반영됩니다.\n"
        )
    return ""


def transform_style(text: str, examples: list[dict]) -> str:
    """문체 변환: 내용(정보·주장·분량)은 그대로 두고 어휘·문장 구조·어조만 내 스타일로 바꿉니다."""
    client = _ensure_client()
    prompt = f"""아래는 사용자가 평소 글을 쓸 때 사용하는 어휘, 문장 구조, 어조 예시입니다.

{_format_examples(examples)}

위 예시들의 어휘 선택, 문장 길이, 어조를 참고해서 다음 텍스트를 다시 써줘.
지켜야 할 규칙:
- 내용(정보, 주장, 논리 구조)은 하나도 빠뜨리거나 더하지 말고 그대로 유지할 것
- 요약하거나 압축하지 말 것 (분량을 원문과 비슷하게 유지할 것)
- 문체(어휘, 문장 구조, 어조)만 위 예시에 맞게 바꿀 것

원문:
{text}"""
    response = client.models.generate_content(model=config.GENERATION_MODEL, contents=prompt)
    return _low_confidence_notice(examples) + response.text
