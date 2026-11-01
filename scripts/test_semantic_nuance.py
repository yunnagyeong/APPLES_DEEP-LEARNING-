
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.verify_legacy import check_semantic_nuance


cases = [
    (
        3,
        "이 결과는 우연일 수도 있다.",
        "This result may simply be a coincidence."
    ),
    (
        11,
        "성능 차이는 크지 않았다.",
        "The performance difference was negligible."
    ),
    (
        20,
        "예를 들어, 학습률이 너무 크면 발산할 수 있다.",
        "For instance, an excessively large learning rate causes divergence."
    ),
    (
        23,
        "모델 A가 모델 B보다 성능이 좋다.",
        "Model A demonstrates superior performance compared to Model B because optimization dictates that evaluation metrics invariably favor architectures with enhanced generalization capabilities."
    ),
    (
        26,
        "데이터가 늘어나면서 정확도가 향상되었다.",
        "As the amount of data increased, the accuracy improved accordingly."
    ),
]

for case_id, source, translation in cases:
    result = check_semantic_nuance(source, translation)

    print(f"\nCase #{case_id}")
    print(result)
