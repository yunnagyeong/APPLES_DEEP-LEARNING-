"""프롬프트 구성 + 생성 모델 호출.

검색된 예시를 "이 문체를 참고해라"는 형태로 프롬프트에 끼워 넣고, 모델 가중치는 건드리지 않은 채
그 순간의 문맥(In-Context Learning)만으로 결과를 생성합니다.
"""

import json
import sys

from google import genai

from . import config 

from .retrieve import retrieve_similar  # 검색된 예시를 가져오는 함수
from .retry import BACKOFF_FACTOR, INITIAL_DELAY, RETRY_ATTEMPTS, call_with_retry

# 유사도가 이 값보다 낮으면 "참고 예시가 부실하다"고 판단합니다. 데이터가 쌓이면
# 실제 분포를 보고 조정하세요. (0~1 범위, 코사인 유사도 기준)
LOW_SIMILARITY_THRESHOLD = 0.55

# 스타일 지표는 절대 임계값이 아니라 효과 크기 d로 해석합니다(투두 23번, D-003):
#   d = (본인 평균 - 나머지 사람들 평균) / 사람 안(풀링) 표준편차
# 본인 값은 팀 작성자(ganghee/nakyung/suyoung)면 기준 코퍼스의 문서별 평균(기준 분포와 같은
# 계산 방식), 팀 밖 사용자면 프로필 값을 씁니다. 팀 작성자는 문서를 다시 뽑았을 때 d의 부호가
# STYLE_SIGN_KEEP_MIN 이상 유지되는 지표만 지시합니다(D-006, build_style_reference.py가 계산).
# 기준값은 data/style_reference.json (scripts/build_style_reference.py로 생성, 코퍼스 essay+report
# 문서별 값). |d| >= STYLE_D_THRESHOLD일 때만 그 방향으로 지시합니다. 0.5는 Cohen의 "중간 효과"
# 관례값을 잠정으로 쓴 것이고 이 코퍼스에서 검증한 값이 아닙니다(3명·67편 소표본).
STYLE_REFERENCE_PATH = config.DATA_DIR / "style_reference.json"
STYLE_D_THRESHOLD = 0.5
STYLE_SIGN_KEEP_MIN = 0.90

_style_reference_cache: dict | None = None


def _load_style_reference() -> dict | None:
    """data/style_reference.json의 metrics 딕셔너리. 없으면 None(경고 후 개인 지시 없음)."""
    global _style_reference_cache
    if _style_reference_cache is None:
        try:
            with open(STYLE_REFERENCE_PATH, encoding="utf-8") as f:
                _style_reference_cache = json.load(f)["metrics"]
        except FileNotFoundError:
            print(f"[generate] {STYLE_REFERENCE_PATH} 없음: scripts/build_style_reference.py를 "
                  "실행하세요. 스타일 지시문 없이 진행합니다.", file=sys.stderr)
            _style_reference_cache = {}
    return _style_reference_cache or None


# 기능어 선택 프로필(24번): 같은 기능을 어떤 표현으로 쓰는지가 사람마다 갈리는 정도(d)가 큰
# 그룹만 영어 지시로 옮깁니다. 지표 이름은 fw_<그룹>(style_features.FUNCTION_WORD_GROUPS).
# D-007: 원문에 그 기능이 있을 때만 "같은 기능의 표현 중 무엇을 고르는가"를 정하는 선택형 문구입니다
# ("Where the source ... render it with ..."). 원문에 없던 대조·결론·이유를 새로 넣게 하지 않습니다.
# d가 양수(팀 나머지보다 많이 씀)인 방향만 지시하고, 적게 쓰는 방향은 지시하지 않습니다(그 표현을
# 피하라고 하면 원문 의미 표지를 지울 수 있음). 대조 표현 그룹(however/but)은 둘 다 걸리면 d가 큰 쪽만.
FUNCTION_WORD_INSTRUCTIONS = {
    "however": "Where the source expresses contrast, render it with a formal marker such as 'however' rather than 'but'",
    "but": "Where the source expresses contrast, render it with plain 'but' rather than a formal marker",
    "whereas": "Where the source contrasts two things, render it with 'whereas' or 'on the other hand'",
    "therefore": "Where the source states a conclusion, render it with 'therefore' or 'thus'",
    "causal": "Where the source gives a reason, render it with 'since' or 'because'",
    "purpose_formal": "Where the source states a purpose, render it with 'in order to'",
}
FUNCTION_WORD_MAX_LINES = 2


def _profile_value(profile: dict, metric: str) -> float | None:
    """프로필에서 지표 값을 꺼낸다. fw_<그룹>은 profile["function_word_choice"] 안에 있다."""
    if metric.startswith("fw_"):
        return (profile.get("function_word_choice") or {}).get(metric[3:])
    return profile.get(metric)


def style_profile_effect_size(profile: dict, user: str | None = None) -> dict[str, float]:
    """효과 크기 d = (본인 - 나머지 사람들 평균) / 사람 안 표준편차. 기준 파일이 없으면 빈 dict.

    user가 기준 코퍼스 작성자면 미리 계산된 값(문서별 평균 기준 author_d)을 쓰고, 아니면(팀 밖
    사용자) 프로필 값을 세 사람 문서 평균의 단순 평균과 비교한다.
    """
    ref = _load_style_reference()
    if not ref:
        return {}
    out = {}
    for m, r in ref.items():
        if user in r.get("author_d", {}):
            out[m] = r["author_d"][user]
        elif _profile_value(profile, m) is not None and r["within_sd"] > 0:
            out[m] = (_profile_value(profile, m) - sum(r["author_mean"].values()) / len(r["author_mean"])) / r["within_sd"]
    return out


def _stable_metrics(user: str | None) -> set[str]:
    """지시문에 써도 되는 지표: 팀 작성자는 d 부호 유지 비율이 STYLE_SIGN_KEEP_MIN 이상인 것만.
    팀 밖 사용자는 부트스트랩 값이 없어 전부 허용(d 크기 조건만 적용)."""
    ref = _load_style_reference() or {}
    return {
        m for m, r in ref.items()
        if user not in r.get("author_sign_keep", {}) or r["author_sign_keep"][user] >= STYLE_SIGN_KEEP_MIN
    }


# 문장 경계·접속 표현 보존 공통 규칙(D-006). 문장 길이·접속어 사용량은 원문에 이미 들어 있어서
# 개인 지시가 비어도, 모델이 문장을 쪼개거나 합치고 접속어를 빠뜨려 평준화하는 것을 막습니다.
# 번역·재번역 프롬프트의 HEDGE_PRESERVATION_RULE 다음 항목으로 붙고, 비교 실험에서
# STRUCTURE_PRESERVATION_RULES = ()로 바꾸면 빠집니다.
STRUCTURE_PRESERVATION_RULES = (
    "Keep the source's sentence boundaries: do not split one sentence into several or merge several into one",
    "Keep every connective and transition expression of the source (e.g., however, therefore, in addition)",
)


def numbered_extra_rules(start: int) -> str:
    """공통 보존 규칙(문장 경계·접속 표현)을 start번부터 번호 붙여 줄바꿈으로 이은 문자열. 없으면 빈 문자열."""
    return "".join(f"\n{start + i}. {r}" for i, r in enumerate(STRUCTURE_PRESERVATION_RULES))


# 모든 사용자에게 공통으로 들어가는 의미 보존 규칙(개인 문체 지시가 아님). 스타일 지시문이
# 헤징을 지우거나 단언을 강화하는 방향으로 작동하지 않도록 프롬프트 본문에 따로 둡니다.
HEDGE_PRESERVATION_RULE = (
    "Preserve every hedge and qualifier in the source (e.g., 'I think', 'may', 'seems'); "
    "never strengthen or weaken a claim"
)

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
    response = call_with_retry(
        lambda: client.models.generate_content(model=config.GENERATION_MODEL, contents=prompt),
        label="transform_style 생성 호출",
    )
    return _low_confidence_notice(examples) + response.text


"""3주차 구현: translate_with_style() - 한국어 → 영어 개인화 번역

Cross-Lingual Style Mapping:
- StyleProfile(한국어 문체 특징) → 영어 자연언어 지시문으로 변환
- RAG로 유사 예시 검색 (한국어는 검색 쿼리용, 예시도 한국어로 "이런 톤")
- Gemini에 프롬프트 전달 → 영어 번역 생성
"""

def _style_profile_to_instructions(
    profile: dict, user: str | None = None, d_threshold: float | None = None,
    source_text: str | None = None,
) -> str:
    """StyleProfile 딕셔너리를 영어 스타일 지시문으로 변환.

    각 지표를 효과 크기 d(나머지 사람들 대비, 사람 안 표준편차 단위)로 바꿔
    |d| >= d_threshold(기본 STYLE_D_THRESHOLD)이고 부호가 안정적인(STYLE_SIGN_KEEP_MIN) 방향만 "평균보다 ~한 편" 식으로 지시합니다(23번). 한국어 어절 수 같은 절대값을 영어 단어 수로
    옮기던 이전 방식은 사람 구분 없이 같은 줄이 나왔습니다. 쓰는 지표는 sentence_length_avg,
    logical_connective_rate, lexical_diversity입니다.

    source_text(번역할 한국어 입력)를 주면 기능어 줄(10번)은 입력에 그 표현이 실제로 있을 때만 넣습니다
    (D-013: 28번에서 "원문에 그 기능이 있을 때만"이라는 문구를 모델이 지키지 못해 나열 "-며"를 "therefore"로,
    "만들고"를 "in order to"로 바꾼 사례가 나옴 → 있는지 판단을 모델이 아니라 Kiwi 규칙이 합니다).
    None이면 조건 없이 전부 넣습니다(프로필만 보는 확인·문서용).
    """
    t = STYLE_D_THRESHOLD if d_threshold is None else d_threshold
    stable = _stable_metrics(user)
    z = {m: v for m, v in style_profile_effect_size(profile, user).items() if m in stable}
    lines = []

    # 1. 문장 길이
    zs = z.get("sentence_length_avg", 0.0)
    if zs >= t:
        lines.append("Do not compress or simplify the source's long sentences")
    # d<0("Do not pad or expand ...") 줄은 삭제(D-013): 28번에서 조건 간 문장 수 비율 차이가 없었음(효과 미확인).

    # 2. 형식성: 합쇼체 비율이 높은 경우에만 지시합니다.
    #    예전에는 값이 낮으면 "casual, conversational tone"을 넣었는데, 한다체 학술·에세이
    #    글(값 0)에 구어체로 쓰라고 지시하는 오류였습니다(D-003).
    formality = profile.get("formality_score", 0)
    if formality >= 0.7:
        lines.append("Use formal, academic or professional tone")

    # 3. 직설성: 지시문에서 제외합니다. directness_score는 "이다/한다/하다 종결 비율"이라
    #    직설성이 아닌 서술어 유형이고, 낮을 때 "diplomatic and hedged"는 헤징 추가를
    #    유도해 의미 보존과 충돌합니다(D-003). 서술어 종결 성향(copula)으로 재정의되면 다시 넣습니다.

    # 4. 논리적 연결: 지시문에서 제외합니다(D-007). "접속어를 평균보다 많이"는 원문에 없던 접속어(인과 주장)를
    #    만들 수 있고, 원문 접속어 보존은 공통 규칙(STRUCTURE_PRESERVATION_RULES)이 맡습니다.
    #    사람별 접속어 선택 차이는 아래 기능어 선택(10번)이 "원문에 있을 때 무엇으로 옮기는가"로 다룹니다.

    # 5. 헤징: 개인 지표로 지시하지 않습니다. 모든 사람에게 공통인 의미 보존 규칙
    #    (HEDGE_PRESERVATION_RULE)을 프롬프트 본문에 넣었습니다. "Avoid hedging"은
    #    원문의 헤징 표현 손실(4주차 오류)과 같은 방향이라 삭제했습니다(D-003).

    # 6~7. 결론 위치·예시 사용: 지시문에서 제외합니다(프로필 계산은 유지, 설명용).
    #    "무엇을 어떤 순서로 쓰는가"라는 구성 층위라 번역에서 따르게 하면 내용 재배열·
    #    예시 추가를 유도합니다(D-002·D-003).

    # 8. 어휘 다양성 (MTLD)
    zl = z.get("lexical_diversity", 0.0)
    if zl >= t:
        lines.append("Where the source uses distinct terms, keep them distinct; do not collapse them into one word")
    elif zl <= -t:
        lines.append("Use consistent, repeated key terms for clarity")

    # 9. 단락 길이: 지시문에서 제외합니다. 원문 문단 구분이 사람마다 달라 문체 지표가
    #    아니고, 구성 층위입니다(D-003).

    # 10. 기능어 선택(24번): d가 크고 안정적인 그룹 중 상위 FUNCTION_WORD_MAX_LINES개
    fw = sorted(
        ((v, m[3:]) for m, v in z.items() if m.startswith("fw_") and v >= t and m[3:] in FUNCTION_WORD_INSTRUCTIONS),
        reverse=True,
    )
    if source_text is not None:  # 입력에 표현이 없는 그룹은 줄을 넣지 않음(상위 2개 자리도 차지하지 않게 먼저 거름)
        from .style_features import source_gate_groups
        present = source_gate_groups(source_text)
        fw = [(v, g) for v, g in fw if g in present]
    picked: list[str] = []
    for v, g in fw:
        if g in ("however", "but") and any(x in ("however", "but") for x in picked):
            continue
        picked.append(g)
    picked = picked[:FUNCTION_WORD_MAX_LINES]
    lines.extend(FUNCTION_WORD_INSTRUCTIONS[g] for g in picked)

    return "\n".join(f"- {line}" for line in lines)


def translate_with_style(
    korean_text: str,
    style_profile: dict,
    user: str,
    type_filter: str | None = None,
    attempts: int = RETRY_ATTEMPTS,
    initial_delay: float = INITIAL_DELAY,
    backoff_factor: float = BACKOFF_FACTOR,
    examples: list[dict] | None = None,
) -> str:
    """한국어 텍스트 → 영어 번역 (개인화된 문체 반영).

    Args:
        korean_text: 번역할 한국어 텍스트
        style_profile: style_features.extract_style_profile()에서 반환한 9항목 dict
        user: 어느 사람의 예시 풀(data/examples/<user>/)에서 검색할지
        type_filter: 검색할 예시의 장르 필터 (e.g., "report", "essay")
                    지정하면 그 장르 예시만 검색합니다.
        attempts, initial_delay, backoff_factor: API 503 재시도 설정.
            기본값은 실시간 데모용(넉넉하게 기다림). 여러 문장을 배치로 돌릴 때는
            attempts=3, initial_delay=5.0, backoff_factor=1 처럼 짧게 주면 됩니다.
        examples: 미리 검색해 둔 RAG 예시(retrieve_similar() 결과). 주면 검색을 건너뜁니다
            (translate_with_verification()이 첫 번역·재번역에 같은 검색 결과를 쓰려고 넘김).
            None이면 기존처럼 이 함수 안에서 직접 검색합니다.

    Returns:
        영어 번역 텍스트 (개인화된 문체 적용). 낮은 검색 신뢰도 안내문은 포함하지 않습니다.
    """
    client = _ensure_client()

    # 1. RAG: 입력 텍스트와 유사한 한국어 예시 검색
    #    (검색 쿼리는 한국어, 예시도 한국어로 제공 — 이 스타일로 쓰라는 레퍼런스)
    if examples is None:
        examples = retrieve_similar(
            korean_text, user, k=config.DEFAULT_TOP_K, type_filter=type_filter
        )

    # 2. StyleProfile → 영어 스타일 지시문
    style_instructions = _style_profile_to_instructions(style_profile, user, source_text=korean_text)

    # 3. 프롬프트 구성
    prompt = f"""You are translating Korean writing to English while preserving the writer's personal style.

The writer's style characteristics (extracted from their Korean writing):
{style_instructions}

Reference examples of the writer's Korean style (to understand tone and approach):
{_format_examples(examples)}

Your task:
1. Translate the Korean text to natural, fluent English
2. Preserve all content (information, arguments, structure)
3. Adapt the writing style to match the characteristics above
4. Do NOT summarize, paraphrase, or omit details—maintain the original scope and tone
5. {HEDGE_PRESERVATION_RULE}{numbered_extra_rules(6)}

Korean text to translate:
{korean_text}

Translated text in English:"""

    # 4. 생성 모델 호출
    response = call_with_retry(
        lambda: client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=prompt,
        ),
        label="translate_with_style 생성 호출",
        attempts=attempts,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
    )

    # 번역문만 반환합니다. 검색 신뢰도 낮음 안내(_low_confidence_notice)를 번역문 앞에
    # 붙이면 검증(verify_translation)이 안내문까지 번역의 일부로 보고 가짜
    # UNSUPPORTED_ADDITION을 만들 수 있어서, 안내문은 호출하는 쪽에서 따로 붙입니다.
    return response.text