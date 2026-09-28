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
from .generate import _ensure_client, _format_examples, _style_profile_to_instructions, translate_with_style
from .retrieve import retrieve_similar
from .retry import call_with_retry

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

"""5주차: Generation-Verification 연결 + 재번역 트리거 로직.

verify.py의 verify_translation()을 호출해서 검증하고, PASS가 아니면
failure_reasons를 프롬프트에 구체적으로 넣어서 재번역합니다.

설계 결정 (강희님과 확인 필요):
- verdict가 FAIL일 때만이 아니라 REVIEW일 때도 재번역을 시도합니다.
  이유: verify.py의 determine_verdict()는 CONTRADICTION만 critical_errors로
  잡고 FAIL을 주는데, 우리가 4주차에 찾은 가장 심각한 사례(#23 Unsupported
  Addition)는 CONTRADICTION이 아니라서 FAIL이 아니라 REVIEW로 분류됩니다.
  FAIL만 재번역하면 Addition류 문제는 그냥 통과되어 버립니다.
"""

# 최초 1회 + 이 횟수만큼 추가 재번역 시도 (총 시도 = 1 + max_retries)
DEFAULT_MAX_RETRIES = 2


def _format_failure_reasons(failure_reasons: list[dict]) -> str:
    """verify.py의 failure_reasons를 모델이 이해할 수 있는 지시문으로 변환."""
    lines = []
    for reason in failure_reasons:
        error_type = reason["type"]
        claim = reason["claim"]

        if error_type == "CONTRADICTION":
            lines.append(
                f'- CONTRADICTION: the claim "{claim}" contradicts the meaning '
                f"of the original text. Fix this so the meaning matches the source."
            )
        elif error_type == "UNSUPPORTED_ADDITION":
            lines.append(
                f'- UNSUPPORTED ADDITION: your translation states "{claim}", '
                f"but this is NOT present in the original text. Remove this "
                f"and do not add explanations, justifications, or details that "
                f"are not explicitly in the source."
            )
        elif error_type == "OMISSION_OR_UNSUPPORTED":
            lines.append(
                f'- POSSIBLE OMISSION: the original claim "{claim}" does not '
                f"appear to be reflected in your translation. Make sure this "
                f"point is preserved."
            )

    return "\n".join(lines)


def _retranslate_with_feedback(
    korean_text: str,
    previous_translation: str,
    failure_reasons: list[dict],
    style_profile: dict,
    type_filter: str | None,
) -> str:
    """이전 번역 + 검증 실패 사유를 프롬프트에 넣어 재생성."""
    client = _ensure_client()
    examples = retrieve_similar(korean_text, k=4, type_filter=type_filter)

    # generate.py에 이미 있는 스타일 지시문 변환 함수 재사용
    from .generate import _style_profile_to_instructions

    style_instructions = _style_profile_to_instructions(style_profile)
    issues_text = _format_failure_reasons(failure_reasons)

    prompt = f"""You previously translated the Korean text below, but automated verification found issues with your translation.

Original Korean text:
{korean_text}

Your previous translation:
{previous_translation}

Issues found by verification:
{issues_text}

Please produce a corrected English translation that:
1. Fixes every issue listed above
2. Still preserves the writer's personal style:
{style_instructions}
3. Does not introduce any new problems (no fabricated content, no dropped claims, no reversed meaning)

Reference examples of the writer's Korean style:
{_format_examples(examples)}

Corrected translation:"""

    response = call_with_retry(
        lambda: client.models.generate_content(
            model=config.GENERATION_MODEL, contents=prompt
        ),
        label="retranslation with feedback",
    )
    return response.text


def translate_with_verification(
    korean_text: str,
    style_profile: dict,
    type_filter: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """개인화 번역 -> 검증 -> (필요시) 실패 사유 기반 재번역 까지 처리하는 통합 함수.

    주의: verify_translation()이 호출당 LLM claim 추출 2회 + NLI 검사를 수행하므로,
    max_retries=2면 최악의 경우 번역 3회 + 검증 3회(각 검증마다 LLM 호출 2회 이상)가
    발생합니다. 무료 티어 쿼터를 고려해서 max_retries를 너무 높이지 마세요.

    Returns:
        {
            "translation": str,               # 최종 번역 결과
            "verdict": str,                    # 최종 verdict (PASS/REVIEW/FAIL)
            "attempts": int,                   # 실제 시도 횟수
            "verification_history": list[dict],  # 각 시도의 verify_translation() 결과
        }
    """
    from .verify import verify_translation  # NLI 모델 로딩이 무거워서 실제 호출 시점에 import

    translation = translate_with_style(korean_text, style_profile, type_filter=type_filter)
    history = []

    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        verification = verify_translation(korean_text, translation)
        history.append(verification)
        verdict = verification["verdict"]
        print(f"[verify] 시도 {attempt}/{total_attempts}: verdict={verdict}")

        if verdict == "PASS":
            break

        if attempt == total_attempts:
            print(f"[verify] 최대 시도({total_attempts}회) 도달 - 마지막 결과를 그대로 반환")
            break

        # FAIL, REVIEW 둘 다 재번역 대상. failure_reasons(전체) 기준으로 피드백 구성.
        translation = _retranslate_with_feedback(
            korean_text,
            translation,
            verification["failure_reasons"],
            style_profile,
            type_filter,
        )

    return {
        "translation": translation,
        "verdict": history[-1]["verdict"],
        "attempts": len(history),
        "verification_history": history,
    }