"""프로젝트 전역 설정.

모델 이름이나 경로를 바꾸고 싶을 때 이 파일만 고치면 되도록 한 곳에 모아둡니다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 경로
# 사람별로 코퍼스가 따로 있으므로(ganghee/nakyung/suyoung), 모든 데이터 경로는 user(영문
# 이름)로 구분합니다. 파일명은 "corpus_<user>_raw.jsonl" / "corpus_<user>_examples.jsonl"
# 형식으로 통일해서, 폴더를 안 열어봐도 ls만으로 누구 데이터인지 알 수 있게 합니다.
# 서로 다른 사람의 예시/스타일을 섞어 쓰는 실수를 막기 위해 기본값 없이 항상 명시적으로
# 넘기도록 함수로 둡니다.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"  # 사람별 원본 코퍼스 (예: raw/corpus_ganghee_raw.jsonl)
EXAMPLES_DIR = DATA_DIR / "examples"
STYLE_PROFILE_CACHE_DIR = DATA_DIR / "style_profile_cache"
FAILED_CASES_DIR = DATA_DIR / "failed_cases"


def raw_path(user: str) -> Path:
    return RAW_DIR / f"corpus_{user}_raw.jsonl"


def examples_path(user: str) -> Path:
    return EXAMPLES_DIR / f"corpus_{user}_examples.jsonl"


def index_cache_path(user: str) -> Path:
    return EXAMPLES_DIR / f"corpus_{user}_index_cache.json"


def style_profile_cache_path(user: str) -> Path:
    return STYLE_PROFILE_CACHE_DIR / f"{user}.json"


def failed_cases_path(user: str) -> Path:
    return FAILED_CASES_DIR / f"{user}.jsonl"

# API 키
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# 모델 (다른 제공자로 바꾸려면 embeddings.py / generate.py의 호출부와 함께 수정)
# 아래 두 모델명은 이 스캐폴드를 만든 시점 기준이며, Google이 모델 라인업을 자주 바꾸므로
# 에러가 나면 `python scripts/list_models.py`로 지금 쓸 수 있는 모델명을 먼저 확인하세요.
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.5-flash-lite"  # 팀이 실습 때 무료 티어로 이미 써본 모델
# 의미 검증 판정 모델. 무료 티어 한도가 빡빡함(분당 5회·하루 20회).
VERIFY_MODEL = "gemini-3.6-flash"
# VERIFY_MODEL의 하루 한도가 끝나면 이 모델로 판정(시연이 멈추지 않게). None이면 대체하지 않고 오류.
VERIFY_FALLBACK_MODEL = GENERATION_MODEL
# 판정 호출의 생각(추론) 길이. "low"면 지연시간이 크게 줄어듦. 지원하지 않는 모델이면 자동으로 무시됨.
VERIFY_THINKING_LEVEL = "low"

# 의미 검증 v2 (src/verify.py) 규칙 기반 안전장치
# 원문·번역문 임베딩 유사도가 이보다 낮으면 판정과 별개로 major 문제로 올림(판정 모델이 놓친 큰 이탈 대비).
# 지금까지 정상·오류 번역 모두 0.83~0.95였으므로 꽤 보수적인 값.
SIMILARITY_GUARD = 0.70

# 재번역 기본 횟수 (최초 번역 + 이 횟수). 앱 데모는 app.py의 DEMO_MAX_RETRIES를 씀.
DEFAULT_MAX_RETRIES = 2

# [레거시] src/verify_legacy.py(7주차 NLI 검증)용 설정
# True: claim 추출(원문·번역문)과 뉘앙스 점검을 Gemini 호출 1번에 묶음. False: 7주차와 같은 호출 3번.
VERIFY_SINGLE_CALL = True

# 검색 기본값
DEFAULT_TOP_K = 4
