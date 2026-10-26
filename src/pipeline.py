"""Generation 엔드투엔드 1차 파이프라인.

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
from .generate import (
    _ensure_client,
    _low_confidence_notice,
    translate_with_style,
)
from .retrieve import retrieve_similar
from .retry import call_with_retry


def _examples_fingerprint(user: str) -> str:
    import hashlib
    return hashlib.sha256(config.examples_path(user).read_bytes()).hexdigest()[:16]


def _load_or_compute_style_profile(user: str, force_recompute: bool = False) -> dict:
    """캐싱된 스타일 프로필을 불러오거나, 없으면 새로 계산해서 저장합니다.

    캐시에 예시 파일의 지문(_examples_hash)을 같이 저장해서, examples.jsonl이 바뀌면 자동으로 다시 계산합니다.
    force_recompute=True면 무조건 다시 계산합니다.
    """
    cache_path = config.style_profile_cache_path(user)
    fingerprint = _examples_fingerprint(user)
    if not force_recompute and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("_examples_hash") == fingerprint:  # 지문이 없거나 다르면(예전 캐시·예시 변경) 다시 계산
            cached.pop("_examples_hash", None)
            return cached

    from . import style_features as sf
    from .store import read_examples

    texts = [ex["text"] for ex in read_examples(user)]
    profile = sf.extract_style_profile(texts)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({**profile, "_examples_hash": fingerprint}, f, ensure_ascii=False, indent=2)
    return profile


def generate_translation(
    user: str,
    korean_text: str,
    type_filter: str | None = None,
    force_recompute_profile: bool = False,
) -> str:
    """엔드투엔드 파이프라인: 한국어 텍스트 → 개인화 영어 번역.

    데모/CLI에서는 이 함수 하나만 호출하면 됩니다.
    (내부적으로 스타일 프로필 로드/계산 + RAG 검색 + 프롬프트 구성 + 생성 모델 호출까지 처리)

    Args:
        user: 어느 사람의 예시 풀/스타일 프로필을 쓸지 (예: "ganghee")
        korean_text: 번역할 한국어 텍스트
        type_filter: RAG 검색 시 장르 필터 (예: "report", "essay"). 없으면 전체에서 검색.
        force_recompute_profile: True면 캐시 무시하고 스타일 프로필 재계산

    Returns:
        개인화된 영어 번역 텍스트
    """
    profile = _load_or_compute_style_profile(user, force_recompute=force_recompute_profile)
    return translate_with_style(korean_text, profile, user, type_filter=type_filter)


def generate_translation_batch(
    user: str,
    korean_texts: list[str],
    type_filter: str | None = None,
) -> list[str | None]:
    """여러 문장을 연속으로 번역합니다 (오류 사례 수집 등 배치 테스트용).

    문장마다 재시도를 짧게(최대 3회, 5초 고정 간격) 시도하고, 그래도 503 등으로 실패하면
    건너뛰고 다음 문장으로 넘어갑니다. 실패한 문장은 data/failed_cases/<user>.jsonl에
    원문과 에러 메시지를 한 줄씩(JSONL) 기록합니다.

    Returns:
        입력 순서와 같은 길이의 리스트. 실패한 자리는 None.
    """
    profile = _load_or_compute_style_profile(user)
    results: list[str | None] = []

    failed_path = config.failed_cases_path(user)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failed_path, "w", encoding="utf-8") as failed_log:
        for i, text in enumerate(korean_texts, start=1):
            try:
                results.append(
                    translate_with_style(
                        text,
                        profile,
                        user,
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

"""검증 + 재번역 루프 (v2).

translate_with_style()로 번역 → verify.verify_translation()으로 검증 → 고칠 문제(major·critical)가 있으면
문제 구절만 고치는 재번역 → 다시 검증. 시도 중 가장 좋은 번역을 돌려줍니다.

v2에서 바뀐 점 (근거: docs/PIPELINE_V2.md)
- 재번역은 FAIL·REVIEW일 때만. minor(시제·정도·군더더기 단어)는 PASS로 두고 화면에 참고로만 표시.
- 재번역 프롬프트는 "처음부터 다시 번역"이 아니라 "지적된 구절만 고치고 나머지 문장은 그대로" (최소 수정).
  다시 번역하면 이미 맞던 부분에 새 문제가 생기거나 문체가 흔들리기 때문.
- 마지막 시도가 아니라 가장 좋은 시도(판정 → 점수 순)를 반환. 재번역이 오히려 나빠지는 경우를 막음.
- 원문 claim은 첫 검증에서 정한 것을 이후 시도에도 그대로 써서 시도끼리 점수를 공정하게 비교.
"""

DEFAULT_MAX_RETRIES = config.DEFAULT_MAX_RETRIES
_VERDICT_RANK = {"PASS": 2, "REVIEW": 1, "FAIL": 0}


def _format_issues(issues: list[dict]) -> str:
    """검증 문제를 재번역 모델용 지시로. 판정 모델이 인용한 구절(evidence)을 그대로 보여줌."""
    lines = []
    for i in issues:
        parts = [f"- [{i['type']}]"]
        if i.get("claim"):
            parts.append(f'source claim: "{i["claim"]}"')
        if i.get("evidence"):
            parts.append(f'translation span: "{i["evidence"]}"')
        if i.get("note"):
            parts.append(f"problem: {i['note']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


_FIX_HINTS = {
    "UNSUPPORTED_ADDITION": "delete the added reason/explanation/fact entirely",
    "MODALITY_SHIFT": "restore the source's exact degree of certainty (keep may/might/seems if the source hedges)",
    "ATTRIBUTION_LOST": "restore the subjective framing exactly as in the source (e.g. I think, personally)",
    "OMISSION": "add the missing content",
    "SCOPE_CHANGE": "restore the source's exact scope or quantity",
    "CONDITION_DROPPED": "restore the condition",
}


def _retranslate_with_feedback(
    korean_text: str,
    previous_translation: str,
    issues: list[dict],
    style_profile: dict,
    examples: list[dict],
    user: str | None = None,
) -> str:
    """지적된 구절만 고치는 최소 수정 재번역."""
    client = _ensure_client()
    from .generate import HEDGE_PRESERVATION_RULE, _style_profile_to_instructions

    style_instructions = _style_profile_to_instructions(style_profile, user, source_text=korean_text)
    hints = sorted({f"- {t}: {_FIX_HINTS[t]}" for t in (i["type"] for i in issues) if t in _FIX_HINTS})
    prompt = f"""You translated a Korean text into English in a specific writer's style. An automated meaning check
found the problems listed below. Revise your translation with the SMALLEST possible edit.

Rules:
1. Change only the spans needed to fix the listed problems. Keep every other sentence and word exactly as it is,
   so the writer's style is preserved.
2. Do not add any content that is not in the Korean source; do not drop any content that is.
3. {HEDGE_PRESERVATION_RULE}
4. Output only the revised English translation, nothing else.

How to fix each problem type:
{chr(10).join(hints) if hints else "- make the meaning match the Korean source exactly"}

Writer's style notes (keep them):
{style_instructions or "- (none)"}

Korean source:
{korean_text}

Your previous translation:
{previous_translation}

Problems found:
{_format_issues(issues)}

Revised translation:"""

    response = call_with_retry(
        lambda: client.models.generate_content(model=config.GENERATION_MODEL, contents=prompt),
        label="retranslation with feedback",
    )
    return response.text.strip()


def _attempt_key(v: dict) -> tuple:
    return (_VERDICT_RANK[v["verdict"]], v["meaning_score"] or 0.0)


def translate_with_verification(
    korean_text: str,
    style_profile: dict,
    user: str,
    type_filter: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    examples: list[dict] | None = None,
) -> dict:
    """개인화 번역 → 검증 → (FAIL·REVIEW면) 최소 수정 재번역. 가장 좋은 시도를 반환.

    호출 수 (재번역 없음 / 1회 / 2회): 번역 생성 1/2/3 + 판정(VERIFY_MODEL) 1/2/3 + 임베딩 2/3/4
    (검색 1 + 유사도: 첫 검증 원문·번역문 묶음 1, 재검증마다 번역문 1). 판정과 유사도 임베딩은 동시에 보냄.
    examples를 주면 검색 임베딩 1회가 빠짐(app.py가 B·C에 같은 검색 결과를 씀).

    Returns:
        translation        가장 좋은 시도의 번역
        verdict            그 시도의 판정 PASS/REVIEW/FAIL
        meaning_score      그 시도의 Meaning Score (0~100)
        meaning_score_method, meaning_components
        issues             그 시도에서 찾은 문제 전체 (minor 포함, 화면 표시용)
        attempts           실제 시도 횟수
        best_attempt       반환한 번역이 몇 번째 시도인지 (1부터)
        translations       시도별 번역
        verification_history  시도별 verify_translation() 결과
        judge_model        판정에 쓴 모델 (하루 한도 소진 시 대체 모델일 수 있음)
        notice             검색 신뢰도 낮음 안내문 (없으면 "")
    """
    from .verify import verify_translation

    if examples is None:
        examples = retrieve_similar(korean_text, user, k=config.DEFAULT_TOP_K, type_filter=type_filter)
    notice = _low_confidence_notice(examples)

    translation = translate_with_style(
        korean_text, style_profile, user, type_filter=type_filter, examples=examples
    ).strip()
    history: list[dict] = []
    translations: list[str] = []
    source_cache: dict = {}

    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        verification = verify_translation(korean_text, translation, source_cache=source_cache)
        history.append(verification)
        translations.append(translation)
        print(f"[verify] 시도 {attempt}/{total_attempts}: verdict={verification['verdict']} "
              f"score={verification['meaning_score']}")

        if verification["verdict"] == "PASS" or attempt == total_attempts:
            break
        translation = _retranslate_with_feedback(
            korean_text, translation, verification["failure_reasons"], style_profile, examples, user,
        )

    best = max(range(len(history)), key=lambda i: (_attempt_key(history[i]), -i))  # 동점이면 앞 시도(원래 문체에 가까움)
    v = history[best]
    return {
        "translation": translations[best],
        "verdict": v["verdict"],
        "meaning_score": v["meaning_score"],
        "meaning_score_method": v["score_method"],
        "meaning_components": v["meaning_components"],
        "issues": v["issues"],
        "attempts": len(history),
        "best_attempt": best + 1,
        "translations": translations,
        "verification_history": history,
        "judge_model": v["judge_model"],
        "notice": notice,
    }
