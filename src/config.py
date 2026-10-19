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

# 검색 기본값
DEFAULT_TOP_K = 4
