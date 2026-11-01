"""Gemini API 호출 재시도 헬퍼.

503(과부하)·429(분당 한도) 같은 일시적 오류가 나면 대기시간을 늘려가며(5초 → 10초 → 20초 ...) 재시도합니다.

다만 기다려도 해결되지 않는 오류는 바로 실패시킵니다(데모에서 몇 분씩 멈춰 있지 않도록).
- 하루 한도 소진(429 메시지에 "PerDay"): 내일까지 안 풀림
- 400/401/403/404: 키·모델명·요청 형식 문제
429 응답에 서버가 권하는 대기시간(retryDelay)이 있으면 그 값을 씁니다.
"""

import re
import sys
import time

RETRY_ATTEMPTS = 5
INITIAL_DELAY = 5.0  # seconds
BACKOFF_FACTOR = 2
MAX_DELAY = 60.0

_NON_RETRYABLE_CODES = (400, 401, 403, 404)


def _status_code(e: Exception) -> int | None:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if isinstance(code, int):
        return code
    m = re.match(r"\s*(\d{3})\b", str(e))
    return int(m.group(1)) if m else None


def _is_daily_quota(e: Exception) -> bool:
    """하루 한도 소진(429 + PerDay). 기다려도 오늘은 안 풀림."""
    return _status_code(e) == 429 and "PerDay" in str(e)


def _is_fatal(e: Exception) -> bool:
    """재시도해도 소용없는 오류인가. JSON 파싱 실패(ValueError)는 다시 물으면 될 수 있어 재시도."""
    code = _status_code(e)
    if code in _NON_RETRYABLE_CODES:
        return True
    return _is_daily_quota(e)


def _server_retry_delay(e: Exception) -> float | None:
    """429 응답의 retryDelay(예: '17s')."""
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(e))
    return float(m.group(1)) if m else None


def call_with_retry(
    fn,
    *,
    label: str,
    attempts: int = RETRY_ATTEMPTS,
    initial_delay: float = INITIAL_DELAY,
    backoff_factor: float = BACKOFF_FACTOR,
):
    """fn()을 호출하고, 실패하면 원인 메시지를 출력한 뒤 대기시간을 늘려가며 재시도합니다.

    backoff_factor=1이면 대기시간이 늘어나지 않고 매번 initial_delay로 고정됩니다
    (여러 건을 빠르게 훑어야 하는 배치 처리용).
    """
    delay = initial_delay
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts or _is_fatal(e):
                raise
            wait = min(_server_retry_delay(e) or delay, MAX_DELAY)
            print(
                f"[retry] {label} 실패 ({attempt}/{attempts}회): {e} "
                f"— {wait:.0f}초 후 재시도합니다.",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay *= backoff_factor
