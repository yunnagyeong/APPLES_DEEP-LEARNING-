# Verification Edge Case Test Results

## 목적

Verification 모듈이 번역 과정에서 발생할 수 있는 주요 의미 변화를
탐지할 수 있는지 확인하기 위해 엣지케이스 테스트를 수행하였다.

## 테스트 결과

| Case | Meaning Change | Verdict | Result |
|---|---|---|---|
| Normal Translation | 의미 변화 없음 | PASS | 정상 |
| Modality Change | 가능성 → 확정 | REVIEW | 탐지 성공 |
| Scope Change | 일부 → 모든 | FAIL | 탐지 성공 |
| Condition Omission | 조건절 삭제 | REVIEW | 탐지 성공 |
| Numerical Change | 30% → 20% | FAIL | 탐지 성공 |
| Important Information Omission | 영업이익 10% 감소 정보 삭제 | FAIL | 탐지 성공 |

## 발견된 Edge Cases

### 1. Compound Entity Claim Splitting

"small and medium-sized enterprises"와 같은 표현이
Claim Extraction 과정에서 다음과 같이 두 개의 claim으로 분리될 수 있었다.

- Small enterprises can adopt new technologies.
- Medium-sized enterprises can adopt new technologies.

의미 오류 탐지에는 성공했지만, 하나의 개념이 불필요하게 여러 claim으로
분리될 가능성이 있다.

### 2. Omission Classified as Contradiction

원문의 핵심 정보를 번역에서 완전히 삭제한 경우,
실제로는 omission에 해당하지만 NLI 모델이 contradiction으로
분류하는 사례가 관찰되었다.

예:

Source:
회사는 지난해 매출이 20% 증가했지만 영업이익은 10% 감소했다.

Translation:
The company's revenue increased by 20% last year.

누락된 영업이익 관련 claim이 contradiction으로 분류되었다.

현재 단계에서는 최종적으로 FAIL 판정이 이루어지므로 의미 오류 탐지에는
문제가 없지만, 향후 failure reason을 보다 세밀하게 분류할 때 보완이 필요하다.

## 현재 결론

테스트한 주요 의미 변화 유형에서는 Verification 모듈이 의미 오류를
PASS로 처리하는 심각한 실패 사례는 발견되지 않았다.

다만 Claim Extraction의 claim 분리 방식과
NLI 기반 omission/contradiction 구분은 향후 개선 대상으로 남는다.