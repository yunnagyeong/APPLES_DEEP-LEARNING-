"""Gemini 호출 공통 헬퍼 (검증 판정·재번역에서 사용).

- 클라이언트는 프로세스당 하나만 만듭니다.
- json_mode=True면 JSON 응답을 강제하고 파싱까지 해서 dict로 돌려줍니다(파싱 실패도 재시도 대상).
- thinking_level을 주면 생각(추론) 길이를 줄여 지연시간을 줄입니다. 모델이 이 옵션을 지원하지 않아
  400이 나면 옵션 없이 다시 보내고, 이후로는 그 모델에 보내지 않습니다.
- fallback_model을 주면 1순위 모델의 하루 한도가 끝났을 때(429 PerDay) 그 모델로 바꿔 보냅니다
  (config.VERIFY_FALLBACK_MODEL, 기본 None = 대체 안 함).
"""

from __future__ import annotations

import json
import sys

from google import genai
from google.genai import types

from . import config
from .retry import _is_daily_quota, call_with_retry

_client = None
_no_thinking_models: set[str] = set()
_exhausted_models: set[str] = set()  # 이번 실행에서 하루 한도가 끝난 모델


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GOOGLE_API_KEY:
            print("[llm] GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.", file=sys.stderr)
            sys.exit(1)
        _client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _client


def parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return json.loads(raw)


def _config_for(model: str, json_mode: bool, thinking_level: str | None):
    kwargs = {}
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    if thinking_level and model not in _no_thinking_models:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    return types.GenerateContentConfig(**kwargs) if kwargs else None


def _call_once(model: str, prompt: str, json_mode: bool, thinking_level: str | None):
    client = get_client()
    try:
        response = client.models.generate_content(
            model=model, contents=prompt, config=_config_for(model, json_mode, thinking_level)
        )
    except Exception as e:  # thinking 옵션 미지원 모델이면 옵션 없이 한 번 더
        if thinking_level and model not in _no_thinking_models and "thinking" in str(e).lower():
            _no_thinking_models.add(model)
            response = client.models.generate_content(
                model=model, contents=prompt, config=_config_for(model, json_mode, None)
            )
        else:
            raise
    text = response.text or ""
    return parse_json(text) if json_mode else text


def generate(
    prompt: str,
    *,
    model: str,
    label: str,
    json_mode: bool = False,
    thinking_level: str | None = None,
    fallback_model: str | None = None,
    validate=None,
):
    """(결과, 실제로 쓴 모델명)을 반환합니다.

    validate(result)가 주어지면 응답마다 호출해서, 예외를 던지면(형식이 틀린 판정 등) 다시 묻습니다.
    """
    models = [m for m in (model, fallback_model) if m]
    last_error: Exception | None = None
    for m in models:
        if m in _exhausted_models and m != models[-1]:
            continue
        try:
            def _attempt(m=m):
                r = _call_once(m, prompt, json_mode, thinking_level)
                if validate is not None:
                    validate(r)
                return r

            result = call_with_retry(_attempt, label=f"{label} [{m}]", attempts=3)
            return result, m
        except Exception as e:
            last_error = e
            if _is_daily_quota(e) and m != models[-1]:
                _exhausted_models.add(m)
                print(f"[llm] {m} 하루 한도 소진 → {models[-1]}로 대체합니다.", file=sys.stderr)
                continue
            raise
    raise last_error  # pragma: no cover
