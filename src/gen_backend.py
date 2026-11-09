"""번역 생성 호출 한 곳으로 모으기: Gemini(기본) 또는 OpenAI 호환 서버(Ollama·OpenRouter 등).

기본 설정(환경변수 없음)에서는 예전과 똑같이 client.models.generate_content(model=GENERATION_MODEL, contents=prompt)를 부릅니다.
프롬프트는 건드리지 않습니다. 모델 크기별 비교 실험(11/02)에서 생성 모델만 바꾸려고 만든 얇은 층입니다.
추론형 모델이 내는 <think>…</think> 블록은 번역문에서 뺍니다.
"""
import json
import re
import urllib.request

from . import config

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def _gemini(prompt: str) -> str:
    from .generate import _ensure_client  # 순환 import 방지: 호출 시점에 가져옴

    client = _ensure_client()
    kwargs = {"model": config.GENERATION_MODEL, "contents": prompt}
    if config.GENERATION_TEMPERATURE is not None:
        from google.genai import types

        kwargs["config"] = types.GenerateContentConfig(temperature=config.GENERATION_TEMPERATURE)
    return client.models.generate_content(**kwargs).text


def _openai_compat(prompt: str) -> str:
    body = {"model": config.GENERATION_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False}
    if config.GENERATION_TEMPERATURE is not None:
        body["temperature"] = config.GENERATION_TEMPERATURE
    req = urllib.request.Request(
        config.OPENAI_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.OPENAI_API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read().decode("utf-8"))
    return _THINK.sub("", data["choices"][0]["message"]["content"] or "").strip()


def generate_text(prompt: str) -> str:
    """번역 생성 모델에 prompt를 보내 텍스트를 돌려받는다."""
    if config.GENERATION_BACKEND == "openai":
        return _openai_compat(prompt)
    return _gemini(prompt)
