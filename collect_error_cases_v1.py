"""문체 반영 오류 사례 수집 스크립트 - 1차 테스트 케이스.

다양한 유형의 한국어 문장을 실제로 번역해보고 결과를 CSV로 저장합니다.
저장된 CSV를 열어서 '문제유형'/'심각도' 컬럼을 직접 채워 넣으면
팀 공유용 오류 사례 리스트가 완성됩니다 (verify.py 테스트 케이스로 활용 가능).
"""

import csv
import time

from src.pipeline import generate_translation  # 파일 위치에 맞게 수정하세요

# 오류를 유발하기 쉬운 유형별 테스트 문장.
# 각 카테고리는 특정 실패 패턴을 노리고 설계되었습니다.
TEST_CASES = [
    # 1. 헤징/불확실성 표현 -> 단정으로 바뀌는지 확인
    "이 방법이 효율적이라고 생각한다. 하지만 데이터가 부족하면 성능이 떨어질 수도 있다.",
    "아직 확실하지는 않지만, 이 접근법이 더 나을 것 같다.",

    # 2. 부정문 -> 의미가 반대로 뒤집히지 않는지 확인
    "이 모델은 아직 완성되지 않았다.",
    "결과가 기대에 미치지 못했다.",

    # 3. 단정적/직설적 문장 -> 오히려 애매해지지 않는지 확인
    "이 방법은 반드시 실패한다.",
    "데이터 전처리는 필수다.",

    # 4. 논리적 연결이 많은 복잡한 문장 -> 연결 관계가 깨지지 않는지 확인
    "정확도는 높아졌지만, 그러나 학습 시간이 늘어났고, 따라서 실제 서비스에는 적합하지 않을 수 있다.",

    # 5. 전문 용어 -> 용어 자체가 왜곡되지 않는지 확인
    "역전파(backpropagation)를 통해 가중치를 업데이트한다.",

    # 6. 짧고 간결한 명령/지시문 -> 불필요하게 길어지거나 어조가 바뀌지 않는지
    "데이터를 정규화하라.",

    # 7. 예시를 드는 문장 -> example_usage_rate 반영 확인
    "예를 들어, 학습률이 너무 크면 발산할 수 있다.",

    # 8. 개인적 의견이 강한 문장 -> 주관성이 유지되는지
    "나는 이 프레임워크가 제일 편하다고 생각한다.",
]


def run_error_case_collection(
    output_path: str = "error_cases.csv", delay_seconds: float = 5.0
) -> list[dict]:
    """테스트 문장들을 번역하고 결과를 CSV로 저장합니다.

    Args:
        output_path: 결과를 저장할 CSV 경로
        delay_seconds: 각 호출 사이 대기 시간 (무료 티어 RPM 보호 목적, 기본 5초)
    """
    rows = []

    for i, korean_text in enumerate(TEST_CASES, start=1):
        print(f"[{i}/{len(TEST_CASES)}] 번역 중: {korean_text[:30]}...")

        try:
            translation = generate_translation(korean_text)
            error_note = ""
        except Exception as e:
            # 하나 실패해도 전체 배치가 멈추지 않도록 처리
            translation = ""
            error_note = f"API 실패: {e}"
            print(f"  실패: {e}")

        rows.append(
            {
                "번호": i,
                "원문": korean_text,
                "번역": translation,
                "문제유형": "",  # 직접 읽어보고 채워넣기 (예: 헤징손실, 부정반전, 용어왜곡 등)
                "심각도": "",  # 상/중/하 직접 판단
                "비고": error_note,
            }
        )

        if i < len(TEST_CASES):
            time.sleep(delay_seconds)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["번호", "원문", "번역", "문제유형", "심각도", "비고"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n완료. {output_path}에 저장했습니다. 엑셀/구글시트로 열어서 검토하세요.")
    return rows


if __name__ == "__main__":
    run_error_case_collection()
