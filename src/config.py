"""프로젝트 전역 설정.

모델 이름이나 경로를 바꾸고 싶을 때 이 파일만 고치면 되도록 한 곳에 모아둡니다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 경로
ROOT_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_PATH = ROOT_DIR / "data" / "examples" / "examples.jsonl"
INDEX_CACHE_PATH = ROOT_DIR / "data" / "examples" / "index_cache.json"

# API 키
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# 모델 (다른 제공자로 바꾸려면 embeddings.py / generate.py의 호출부와 함께 수정)
# 아래 두 모델명은 이 스캐폴드를 만든 시점 기준이며, Google이 모델 라인업을 자주 바꾸므로
# 에러가 나면 `python scripts/list_models.py`로 지금 쓸 수 있는 모델명을 먼저 확인하세요.
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.5-flash-lite"  # 팀이 실습 때 무료 티어로 이미 써본 모델

# 검색 기본값
DEFAULT_TOP_K = 4
