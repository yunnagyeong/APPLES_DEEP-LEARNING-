"""examples.jsonl을 기준으로 임베딩 캐시를 재생성하는 편의 스크립트.

사용법: python scripts/build_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.store import build_index

if __name__ == "__main__":
    build_index()
