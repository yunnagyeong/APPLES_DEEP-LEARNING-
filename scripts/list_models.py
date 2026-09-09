"""지금 이 API 키로 실제 사용 가능한 모델 목록을 확인하는 스크립트.

config.py에 적어둔 모델명이 시간이 지나 더 이상 없을 수 있으므로, 에러가 나면 먼저 이걸 돌려서
embedContent / generateContent를 지원하는 모델명을 확인하세요.

사용법: python scripts/list_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai

from src import config

if __name__ == "__main__":
    if not config.GOOGLE_API_KEY:
        print("GOOGLE_API_KEY가 설정되지 않았습니다. .env를 확인하세요.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        print(f"{model.name:45s} {actions}")
