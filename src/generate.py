"""프롬프트 구성 + 생성 모델 호출.

검색된 예시를 "이 문체를 참고해라"는 형태로 프롬프트에 끼워 넣고, 모델 가중치는 건드리지 않은 채
그 순간의 문맥(In-Context Learning)만으로 결과를 생성합니다.
"""

import sys

from google import genai

from . import config 

from .retrieve import retrieve_similar  # 검색된 예시를 가져오는 함수
from .retry import BACKOFF_FACTOR, INITIAL_DELAY, RETRY_ATTEMPTS, call_with_retry

# 유사도가 이 값보다 낮으면 "참고 예시가 부실하다"고 판단합니다. 데이터가 쌓이면
# 실제 분포를 보고 조정하세요. (0~1 범위, 코사인 유사도 기준)
LOW_SIMILARITY_THRESHOLD = 0.55

# lexical_diversity는 MTLD라서 0~1 척도가 아닙니다(대략 수십 단위). 아래 기준값은 강희·수영·나경
# raw 코퍼스 문서 52편의 문서별 MTLD를 3등분한 경계(50.5, 71.7)를 반올림한 잠정값입니다.
# 코퍼스가 늘거나 바뀌면 `python scripts/analyze_lexical_diversity.py`로 다시 계산하세요.
LEXICAL_DIVERSITY_LOW = 50.0
LEXICAL_DIVERSITY_HIGH = 72.0

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

def _style_profile_to_instructions(profile: dict) -> str:
    """StyleProfile 딕셔너리를 영어 스타일 지시문으로 변환.

    한국어 문체 특징을 영어로 설명하되, 영어 쓰기의 뉘앙스를 고려합니다.
    예: formality_score 0.8 → "Use formal, professional vocabulary"
    """
    lines = []

    # 1. 문장 길이
    sent_len = profile.get("sentence_length_avg", 0)
    if sent_len <= 3:
        lines.append("Use short, concise sentences (3-4 words on average)")
    elif sent_len <= 6:
        lines.append("Use moderate sentence length (5-7 words on average)")
    else:
        lines.append("Use longer, more complex sentences (8+ words on average)")

    # 2. 형식성: 합쇼체 비율이 높은 경우에만 지시합니다.
    #    예전에는 값이 낮으면 "casual, conversational tone"을 넣었는데, 한다체 학술·에세이
    #    글(값 0)에 구어체로 쓰라고 지시하는 오류였습니다(D-003).
    formality = profile.get("formality_score", 0)
    if formality >= 0.7:
        lines.append("Use formal, academic or professional tone")

    # 3. 직설성: 지시문에서 제외합니다. directness_score는 "이다/한다/하다 종결 비율"이라
    #    직설성이 아닌 서술어 유형이고, 낮을 때 "diplomatic and hedged"는 헤징 추가를
    #    유도해 의미 보존과 충돌합니다(D-003). 상대 기준(z-점수)으로 재정의되면 다시 넣습니다.

    # 4. 논리적 연결
    logical_conn = profile.get("logical_connective_rate", 0)
    if logical_conn >= 0.5:
        lines.append("Use explicit logical connectives (however, therefore, as a result)")
    else:
        lines.append("Use minimal connectives; let ideas flow implicitly")

    # 5. 헤징: 개인 지표로 지시하지 않습니다. 모든 사람에게 공통인 의미 보존 규칙
    #    (HEDGE_PRESERVATION_RULE)을 프롬프트 본문에 넣었습니다. "Avoid hedging"은
    #    원문의 헤징 표현 손실(4주차 오류)과 같은 방향이라 삭제했습니다(D-003).

    # 6~7. 결론 위치·예시 사용: 지시문에서 제외합니다(프로필 계산은 유지, 설명용).
    #    "무엇을 어떤 순서로 쓰는가"라는 구성 층위라 번역에서 따르게 하면 내용 재배열·
    #    예시 추가를 유도합니다(D-002·D-003).

    # 8. 어휘 다양성
    lex_div = profile.get("lexical_diversity", 0)
    if lex_div >= LEXICAL_DIVERSITY_HIGH:
        lines.append("Use varied vocabulary; avoid repetition")
    elif lex_div < LEXICAL_DIVERSITY_LOW:
        lines.append("Use consistent, repeated key terms for clarity")
    # 중간 구간은 특별한 지시 안 함

    # 9. 단락 길이: 지시문에서 제외합니다. 원문 문단 구분이 사람마다 달라 문체 지표가
    #    아니고, 구성 층위입니다(D-003).

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
    style_instructions = _style_profile_to_instructions(style_profile)

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
5. {HEDGE_PRESERVATION_RULE}

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