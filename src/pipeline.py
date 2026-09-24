"""4주차: Generation 엔드투엔드 1차 파이프라인.

기존 코드(style_features.py의 extract_style_profile, generate.py의 translate_with_style)를
하나의 함수로 묶어서, "한국어 텍스트 → 개인화 영어 번역"을 원스톱으로 처리합니다.

핵심 포인트: 스타일 프로필은 매번 새로 계산하지 않고 캐싱합니다.
- Kiwi 형태소 분석 + 177개 예시 처리는 몇 초가 걸리는 작업이라,
  데모/실제 사용 시 매 요청마다 다시 계산하면 응답이 느려집니다.
- 프로필을 한 번 계산해서 JSON 파일로 저장해두고, 이후에는 그 파일을 읽기만 합니다.
- 데이터(examples.jsonl)가 바뀌었을 때만 force_recompute=True로 재계산하면 됩니다.
"""

import json
import sys

from . import config
from .generate import translate_with_style  # 3주차에서 만든 함수

# 계산된 스타일 프로필을 저장해둘 캐시 파일 경로
STYLE_PROFILE_CACHE_PATH = config.ROOT_DIR / "data" / "style_profile_cache.json"

# 배치 처리(여러 문장 연속 테스트) 시 실패한 문장을 남겨둘 로그 파일
FAILED_CASES_PATH = config.ROOT_DIR / "data" / "failed_cases.jsonl"


def _load_or_compute_style_profile(force_recompute: bool = False) -> dict:
    """캐싱된 스타일 프로필을 불러오거나, 없으면 새로 계산해서 저장합니다.

    force_recompute=True로 호출하면 캐시를 무시하고 examples.jsonl 전체를 다시 분석합니다.
    (팀 데이터가 갱신됐을 때 사용)
    """
    if not force_recompute and STYLE_PROFILE_CACHE_PATH.exists():
        with open(STYLE_PROFILE_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)

    # 캐시가 없거나 강제 재계산인 경우에만 무거운 연산 수행
    from . import style_features as sf
    from .store import read_examples

    examples = read_examples()
    texts = [ex["text"] for ex in examples]
    profile = sf.extract_style_profile(texts)

    STYLE_PROFILE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STYLE_PROFILE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    return profile


def generate_translation(
    korean_text: str,
    type_filter: str | None = None,
    force_recompute_profile: bool = False,
) -> str:
    """엔드투엔드 파이프라인: 한국어 텍스트 → 개인화 영어 번역.

    데모/CLI에서는 이 함수 하나만 호출하면 됩니다.
    (내부적으로 스타일 프로필 로드/계산 + RAG 검색 + 프롬프트 구성 + 생성 모델 호출까지 처리)

    Args:
        korean_text: 번역할 한국어 텍스트
        type_filter: RAG 검색 시 장르 필터 (예: "report", "essay"). 없으면 전체에서 검색.
        force_recompute_profile: True면 캐시 무시하고 스타일 프로필 재계산

    Returns:
        개인화된 영어 번역 텍스트
    """
    profile = _load_or_compute_style_profile(force_recompute=force_recompute_profile)
    return translate_with_style(korean_text, profile, type_filter=type_filter)


def generate_translation_batch(
    korean_texts: list[str],
    type_filter: str | None = None,
) -> list[str | None]:
    """여러 문장을 연속으로 번역합니다 (오류 사례 수집 등 배치 테스트용).

    문장마다 재시도를 짧게(최대 3회, 5초 고정 간격) 시도하고, 그래도 503 등으로 실패하면
    건너뛰고 다음 문장으로 넘어갑니다. 실패한 문장은 FAILED_CASES_PATH에 원문과 에러
    메시지를 한 줄씩(JSONL) 기록합니다.

    Returns:
        입력 순서와 같은 길이의 리스트. 실패한 자리는 None.
    """
    profile = _load_or_compute_style_profile()
    results: list[str | None] = []

    with open(FAILED_CASES_PATH, "w", encoding="utf-8") as failed_log:
        for i, text in enumerate(korean_texts, start=1):
            try:
                results.append(
                    translate_with_style(
                        text,
                        profile,
                        type_filter=type_filter,
                        attempts=3,
                        initial_delay=5.0,
                        backoff_factor=1,
                    )
                )
            except Exception as e:
                print(f"[pipeline] ({i}/{len(korean_texts)}) 재시도 소진, 건너뜀: {e}", file=sys.stderr)
                results.append(None)
                failed_log.write(json.dumps({"text": text, "error": str(e)}, ensure_ascii=False) + "\n")

    return results
