"""문체 반영 오류 사례 수집 - 2차 테스트 케이스.

error_cases.csv를 읽어서, 이미 있는 1차 결과(11개)는 재호출하지 않고 그대로
재사용하고, 2차(신규) 21문장만 이번에 실제로 API를 호출합니다. 최종 결과는
카테고리 순으로 정렬해서 같은 error_cases.csv 파일에 덮어씁니다.
(쿼터 절약 + 데이터/코드 분리 + 파일 하나로 유지)

준비물: error_cases.csv에 '카테고리' 컬럼이 있어야 합니다.

실행: 프로젝트 루트에서
    python collect_error_cases_v2.py
"""

import csv
import time

from src.pipeline import generate_translation  # 파일 위치에 맞게 수정하세요

EXISTING_RESULTS_PATH = "error_cases.csv"

# 카테고리 표시/정렬 순서. 여기 없는 카테고리는 뒤로 밀려납니다.
CATEGORY_ORDER = [
    "헤징/가능성",
    "주관성/개인의견",
    "부정문",
    "단정/직설",
    "명령문",
    "논리연결(복문)",
    "전문용어",
    "예시",
    "숫자/수치",
    "비교",
    "인과관계",
    "조건문",
    "개체명",
    "전문개념",
]

# 카테고리별 "이 유형을 왜 테스트하는가"를 자동으로 채워 넣기 위한 매핑.
# CSV에 매번 손으로 옮겨 적지 않도록, 여기서 한 번만 정의합니다.
CATEGORY_PURPOSE = {
    "헤징/가능성": "가능성/불확실성 표현이 단정적으로 바뀌는지 확인",
    "부정문": "부정의 의미가 반대로 뒤집히지 않는지 확인",
    "단정/직설": "단정적 문장이 오히려 애매해지지 않는지 확인",
    "명령문": "원문에 없는 설명/근거가 추가되는지 확인 (Unsupported Addition)",
    "논리연결(복문)": "논리적 연결 관계가 깨지지 않는지 확인",
    "전문용어": "전문 용어가 왜곡되지 않는지 확인",
    "예시": "예시 표현 속 가능성(헤징)이 유지되는지 확인",
    "주관성/개인의견": "'생각한다'/개인 의견 표현의 주관성이 유지되는지 확인",
    "숫자/수치": "숫자가 왜곡되지 않는지 확인",
    "비교": "비교 방향(A>B 등)이 뒤집히지 않는지 확인",
    "인과관계": "원인/결과 방향이 유지되는지 확인",
    "조건문": "조건 범위가 '항상'으로 과확장되지 않는지 확인",
    "개체명": "고유명사가 바뀌지 않는지 확인 (스타일 변환과 무관해야 정상)",
    "전문개념": "상관관계가 인과관계로 둔갑하지 않는지 확인",
}

# ── 2차 신규 테스트 케이스 (이번에 실제로 API 호출) ──
NEW_TEST_CASES = [
    ("헤징/가능성", "이 결과는 우연일 수도 있다."),
    ("헤징/가능성", "좀 더 검증이 필요해 보인다."),
    ("주관성/개인의견", "내 생각엔 이 설계가 더 낫다."),
    ("주관성/개인의견", "개인적으로는 이 결과에 만족한다."),
    ("명령문", "코드를 리팩토링하라."),
    ("명령문", "이 값을 초기화해야 한다."),
    ("명령문", "결과를 저장하라."),
    ("부정문", "이 실험은 재현되지 않았다."),
    ("부정문", "성능 차이는 크지 않았다."),
    ("숫자/수치", "정확도가 92%에서 87%로 떨어졌다."),
    ("숫자/수치", "샘플은 총 500개였다."),
    ("비교", "모델 A가 모델 B보다 성능이 좋다."),
    ("비교", "이 방법이 기존 방법보다 느리다."),
    ("인과관계", "과적합 때문에 검증 성능이 떨어졌다."),
    ("인과관계", "데이터가 늘어나면서 정확도가 향상되었다."),
    ("조건문", "데이터가 충분할 때만 이 방법이 효과적이다."),
    ("조건문", "조건이 맞으면 성능이 개선된다."),
    ("개체명", "이 논문은 스탠퍼드 대학에서 발표됐다."),
    ("개체명", "실험은 파이토치로 구현했다."),
    ("전문개념", "두 변수 사이에 상관관계가 있다."),
    ("전문개념", "이 지표는 참고용일 뿐, 인과관계를 의미하지 않는다."),
]


def load_existing_results(path: str = EXISTING_RESULTS_PATH) -> list[dict]:
    """이미 채워둔 1차 결과 CSV를 그대로 읽어옵니다. (재호출 없음)

    CSV에 '테스트목적' 컬럼이 없으면 카테고리 기준으로 자동 채워 넣습니다.
    """
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if not row.get("테스트목적"):
            row["테스트목적"] = CATEGORY_PURPOSE.get(row.get("카테고리", ""), "")

    return rows


def run_error_case_collection(
    output_path: str = "error_cases.csv", delay_seconds: float = 5.0
) -> list[dict]:
    rows = load_existing_results()  # 1차 결과 재사용

    total_new = len(NEW_TEST_CASES)
    for i, (category, korean_text) in enumerate(NEW_TEST_CASES, start=1):
        print(f"[{i}/{total_new}] ({category}) 번역 중: {korean_text[:30]}...")

        try:
            translation = generate_translation(korean_text)
            error_note = ""
        except Exception as e:
            translation = ""
            error_note = f"API 실패: {e}"
            print(f"  실패: {e}")

        rows.append(
            {
                "카테고리": category,
                "테스트목적": CATEGORY_PURPOSE.get(category, ""),
                "원문": korean_text,
                "번역": translation,
                "문제유형": "",
                "심각도": "",
                "비고": error_note,
            }
        )

        if i < total_new:
            time.sleep(delay_seconds)

    # 카테고리 순으로 정렬 (CATEGORY_ORDER 기준, 목록에 없는 카테고리는 맨 뒤로)
    def category_sort_key(row: dict) -> int:
        category = row.get("카테고리", "")
        return CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(CATEGORY_ORDER)

    rows.sort(key=category_sort_key)

    # 번호 다시 매기기 (정렬 후 1부터 순서대로)
    numbered_rows = [{"번호": i, **{k: v for k, v in row.items() if k != "번호"}}
                      for i, row in enumerate(rows, start=1)]

    fieldnames = ["번호", "카테고리", "테스트목적", "원문", "번역", "문제유형", "심각도", "비고"]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(numbered_rows)

    print(
        f"\n완료. {output_path}에 저장했습니다. "
        f"(총 {len(numbered_rows)}개: 기존 {len(rows) - total_new}개 + 신규 {total_new}개)"
    )
    return numbered_rows


if __name__ == "__main__":
    run_error_case_collection()
