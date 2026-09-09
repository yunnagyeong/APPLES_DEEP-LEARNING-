"""임베딩 API 래퍼.

Google의 text-embedding 모델을 감싸서, 텍스트 하나를 벡터(숫자 리스트)로 바꿔줍니다.
API 키가 없을 때는 예외를 던지는 대신 안내 메시지를 출력하고 종료하도록 해서,
처음 설정하는 사람이 스택트레이스만 보고 당황하지 않게 했습니다.

google-generativeai(구 SDK)는 지원이 종료되어, 여기서는 후속 SDK인 google-genai를 씁니다.
"""

import sys

from google import genai
from google.genai import types

from . import config

_client = None


def _ensure_client() -> genai.Client:
    global _client
    if _client is not None:
        return _client
    if not config.GOOGLE_API_KEY:
        print(
            "[embeddings] GOOGLE_API_KEY가 설정되지 않았습니다. "
            ".env 파일을 만들고 API 키를 채워주세요 (.env.example 참고).",
            file=sys.stderr,
        )
        sys.exit(1)
    _client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _client


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """텍스트 하나를 임베딩 벡터로 변환합니다.

    task_type:
        - "RETRIEVAL_DOCUMENT": 예시 풀에 저장할 문서를 임베딩할 때
        - "RETRIEVAL_QUERY": 검색 질의(사용자 입력)를 임베딩할 때
    두 값을 구분해서 넣어주면 검색 품질이 조금 더 좋아집니다 (임베딩 모델의 권장 사용법).
    """
    client = _ensure_client()
    result = client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return list(result.embeddings[0].values)
