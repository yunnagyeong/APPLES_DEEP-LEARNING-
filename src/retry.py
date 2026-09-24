"""Gemini API 호출 재시도 헬퍼.

503(과부하) 등 일시적 오류가 나면 대기시간을 늘려가며(5초 → 10초 → 20초 ...) 재시도합니다.
"""

import sys
import time

RETRY_ATTEMPTS = 5
INITIAL_DELAY = 5.0  # seconds
BACKOFF_FACTOR = 2


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
            if attempt == attempts:
                raise
            print(
                f"[retry] {label} 실패 ({attempt}/{attempts}회): {e} "
                f"— {delay:.0f}초 후 재시도합니다.",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= backoff_factor
