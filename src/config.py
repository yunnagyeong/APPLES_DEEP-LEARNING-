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


# 평가 모드: LINGUA_CORPUS_SPLIT=train 이면 평가용 글을 뺀 학습용 코퍼스(corpus_<user>_train.jsonl)로
# 문체 프로필·RAG 예시를 만듭니다. 평가 원문을 이미 본 상태로 번역하는 것을 막기 위함.
# 비워두면(기본) 지금까지처럼 전체 코퍼스를 써서 데모·앱 동작은 바뀌지 않습니다.
# scripts/run_abc.py가 자동으로 train으로 설정하므로 직접 켤 일은 거의 없습니다.
CORPUS_SPLIT = os.getenv("LINGUA_CORPUS_SPLIT", "").strip()
_SPLIT_SUFFIX = f"_{CORPUS_SPLIT}" if CORPUS_SPLIT else ""


def raw_path(user: str) -> Path:
    return RAW_DIR / f"corpus_{user}_raw.jsonl"


def examples_path(user: str) -> Path:
    if CORPUS_SPLIT:
        return EXAMPLES_DIR / f"corpus_{user}{_SPLIT_SUFFIX}.jsonl"
    return EXAMPLES_DIR / f"corpus_{user}_examples.jsonl"


def index_cache_path(user: str) -> Path:
    return EXAMPLES_DIR / f"corpus_{user}{_SPLIT_SUFFIX}_index_cache.json"


def style_profile_cache_path(user: str) -> Path:
    return STYLE_PROFILE_CACHE_DIR / f"{user}{_SPLIT_SUFFIX}.json"


def failed_cases_path(user: str) -> Path:
    return FAILED_CASES_DIR / f"{user}.jsonl"

# API 키
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# 모델 (다른 제공자로 바꾸려면 embeddings.py / generate.py의 호출부와 함께 수정)
# 아래 두 모델명은 이 스캐폴드를 만든 시점 기준이며, Google이 모델 라인업을 자주 바꾸므로
# 에러가 나면 `python scripts/list_models.py`로 지금 쓸 수 있는 모델명을 먼저 확인하세요.
EMBEDDING_MODEL = "gemini-embedding-001"
# 번역 생성 모델. 크기별 비교 실험용으로 환경변수로 바꿀 수 있음(안 주면 기존과 같음):
#   LINGUA_GENERATION_MODEL, LINGUA_GENERATION_BACKEND("gemini" 기본 | "openai" = OpenAI 호환 서버: Ollama·OpenRouter 등),
#   LINGUA_OPENAI_BASE_URL, LINGUA_OPENAI_API_KEY, LINGUA_GENERATION_TEMPERATURE(비우면 모델 기본값)
GENERATION_MODEL = os.getenv("LINGUA_GENERATION_MODEL", "gemini-3.5-flash-lite")  # 번역 생성(빠르고 저렴)
GENERATION_BACKEND = os.getenv("LINGUA_GENERATION_BACKEND", "gemini").strip().lower()
OPENAI_BASE_URL = os.getenv("LINGUA_OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("LINGUA_OPENAI_API_KEY", "ollama")
_temp = os.getenv("LINGUA_GENERATION_TEMPERATURE", "").strip()
GENERATION_TEMPERATURE = float(_temp) if _temp else None
# 의미 검증 판정 모델 (유료 API 기준)
VERIFY_MODEL = "gemini-3.6-flash"
# 판정 모델 한도가 끝났을 때 대신 쓸 모델. None이면 대체하지 않음(기본).
# 대체 모델로 조용히 바뀌면 판정 기준이 달라져 평가 결과를 비교할 수 없으므로 유료 API에서는 끄는 게 맞음.
# 무료 티어로 시연할 때만 GENERATION_MODEL 등으로 켜세요(결과의 judge_model에 실제 모델이 기록됨).
VERIFY_FALLBACK_MODEL = None
# 판정 호출의 생각(추론) 길이. "low"면 지연시간이 크게 줄어듦. 지원하지 않는 모델이면 자동으로 무시됨.
VERIFY_THINKING_LEVEL = "low"

# 의미 검증 v2 (src/verify.py) 규칙 기반 안전장치
# 원문·번역문 임베딩 유사도가 이보다 낮으면 판정과 별개로 major 문제로 올림(판정 모델이 놓친 큰 이탈 대비).
# 지금까지 정상·오류 번역 모두 0.83~0.95였으므로 꽤 보수적인 값.
SIMILARITY_GUARD = 0.70

# 재번역 최대 횟수 (최초 번역 + 이 횟수). 앱·CLI·스크립트 공통.
DEFAULT_MAX_RETRIES = 2

# [레거시] src/verify_legacy.py(7주차 NLI 검증)용 설정
# True: claim 추출(원문·번역문)과 뉘앙스 점검을 Gemini 호출 1번에 묶음. False: 7주차와 같은 호출 3번.
VERIFY_SINGLE_CALL = True

# 검색 기본값
DEFAULT_TOP_K = 4
