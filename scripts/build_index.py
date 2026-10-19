"""examples.jsonl을 기준으로 임베딩 캐시를 재생성하는 편의 스크립트.

사용법: python scripts/build_index.py <user>  (예: python scripts/build_index.py ganghee)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.store import build_index

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python scripts/build_index.py <user>", file=sys.stderr)
        sys.exit(1)
    build_index(sys.argv[1])
