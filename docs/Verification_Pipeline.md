# Verification Pipeline 설계안

## 1. 목적

개인 문체를 반영해 번역하는 과정에서 원문의 의미가 바뀌거나, 빠지거나, 새 내용이 추가되는 문제를 자동으로 검증한다.

전체 구조:

```
한국어 원문
→ Personalized Translation
→ Meaning Preservation Checker
→ PASS / REVIEW / FAIL
→ FAIL이면 오류 이유를 전달해 재번역
```

우리 프로젝트에서 Verification은 별도 모듈로 두고, 생성 결과를 다시 검사하는 역할을 맡는다. 기존 계획서도 임베딩 유사도 + Claim 단위 NLI + Omission/Addition 탐지 구조를 기본 방향으로 잡고 있다.

## 2. 검증 항목

### ① 전체 의미 유사도

**방법**: Multilingual Embedding + Cosine Similarity

한국어 원문과 영어 번역문을 각각 벡터로 변환한 뒤 두 벡터의 유사도를 계산한다.

```
한국어 원문 → embedding A
영어 번역문 → embedding B
cosine similarity(A, B)
```

**역할**:

- 번역 전체가 원문의 주제와 의미를 대체로 유지했는지 1차 확인
- 큰 의미 이탈 탐지

**사용 모델**: `gemini-embedding-2`

선정 이유:

- 한국어와 영어를 포함한 100개 이상의 언어 지원
- 의미 기반 유사성 비교에 사용 가능
- 기존 Gemini 기반 프로젝트와 연동이 쉬움
- 기존 계획서의 `text-embedding-004`는 2026년 1월 종료되었으므로 교체 필요

## 3. Embedding만 쓰지 않는 이유

Embedding은 문장 전체의 의미가 얼마나 비슷한지는 잘 보지만, 작은 변화가 핵심 의미를 뒤집는 경우를 놓칠 수 있다.

예:

> The effect was statistically significant.
> The effect was **not** statistically significant.

두 문장은 단어와 구조가 거의 같지만 의미는 반대다.

따라서:

- 전체 의미 비교 = Embedding
- 세부 주장 검증 = NLI

두 가지를 함께 사용한다.

## 4. Claim 단위 검증

원문 전체를 한 번에 검사하지 않고 핵심 주장인 Claim으로 나눈다.

예:

```
원문:
AI는 생산성을 높일 수 있지만, 일부 직종의 고용을 감소시킬 수도 있다.

↓

Claim 1: AI는 생산성을 높일 수 있다.
Claim 2: AI는 일부 직종의 고용을 감소시킬 수 있다.
```

### Claim Extraction

LLM 사용. LLM의 역할은 여기까지다.

```
원문 → LLM → Claim 1, Claim 2, Claim 3 ...
```

LLM에게 의미 보존 여부까지 판단시키지는 않는다.

### Claim Verification

NLI 모델 사용. NLI는 두 문장의 관계를 다음 세 가지로 판정한다.

- Entailment
- Neutral
- Contradiction

### NLI 모델 후보

`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`

선정 이유:

- multilingual NLI 전용 모델
- 한국어 포함, 100개 언어에서 NLI 수행 가능
- entailment / neutral / contradiction을 직접 판정할 수 있음 ([Hugging Face](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7))

## 5. 세부 오류 탐지

### Omission

원문의 중요한 내용이 번역에서 빠졌는지 검사.

```
영어 번역문 → 한국어 원문의 Claim을 지원하는가?
지원하지 못하면: Omission Candidate
```

### Unsupported Addition

번역문에 원문에는 없던 내용이 추가됐는지 검사. 방향이 반대다.

```
한국어 원문 → 영어 번역문에서 나온 Claim을 지원하는가?
지원하지 못하면: Unsupported Addition Candidate
```

### Contradiction

원문과 번역문의 의미가 반대로 바뀐 경우.

예:

> 원문: 유의한 차이가 없었다.
> 번역: A significant difference was observed.

NLI가 contradiction으로 판단하면 중요한 오류로 처리한다.

## 6. 최종 Verification 구조

```
Personalized Translation
        ↓
Meaning Preservation Checker
        ↓
┌────────────────────────────────┐
│ 1. Semantic Similarity         │
│ 2. Claim Coverage              │
│ 3. Contradiction               │
│ 4. Omission                    │
│ 5. Unsupported Addition        │
└────────────────────────────────┘
        ↓
   PASS / REVIEW / FAIL
```

## 7. Meaning Score 설계

처음부터 임의로 "Embedding 30%, NLI 50% ..." 같이 정하지 않는다.

우선 다음 값을 독립적으로 측정한다.

| 기호 | 의미 |
|---|---|
| S | Semantic Similarity |
| C | Claim Coverage — 원문 Claim 중 보존된 비율 |
| A | Addition Support — 번역문의 Claim 중 원문에서 지원되는 비율 |
| R | Contradiction Rate — 전체 Claim 중 모순된 비율 |

즉 `[S, C, A, R]`를 Verification의 기본 지표로 사용한다.

최종 Meaning Score의 weight와 threshold는 나중에 사람이 직접 평가한 번역 데이터와 비교해 정한다.

이 방식의 장점: "왜 이 지표에 40%를 줬나요?"라는 질문에 임의로 정했다고 답하지 않아도 된다.

## 8. Hard Fail 규칙

평균 점수가 높더라도 중요한 내용 하나가 반대로 바뀌면 통과시키면 안 된다.

따라서:

- Critical contradiction 발견 → **FAIL**
- 중요 Claim 누락 → **FAIL 또는 REVIEW**

같은 별도 규칙을 둔다. 즉 최종 점수만 보고 판단하지 않는다.

## 9. Verification 출력 형식

팀원들과 아래 형태로 맞추는 것을 제안.

```json
{
  "semantic_similarity": 0.91,
  "claim_coverage": 0.95,
  "addition_support": 1.0,
  "contradiction_rate": 0.0,
  "critical_errors": [],
  "verdict": "PASS"
}
```

필요하면 세부 Claim 결과도 포함:

```json
{
  "id": "C1",
  "text": "...",
  "status": "entailed",
  "confidence": 0.94
}
```

이렇게 해두면:

- **수영의 Generation 모듈**은 `verdict`, `critical_errors`만 받아서 재번역 여부를 결정할 수 있고,
- **나경의 UI**에서는 "Semantic Similarity 91% · Claim Coverage 95% · No Contradiction · PASS" 형태로 바로 표시할 수 있다.

## 10. 2주차 테스트용 오류 유형

Verification이 제대로 작동하는지 다음 오류를 의도적으로 만든 데이터로 시험한다.

| 유형 | 예시 |
|---|---|
| 부정 | 유의하다 → 유의하지 않다 |
| 숫자 | 20% → 30% |
| 가능성 | 가능하다 → 반드시 발생한다 |
| 누락 | Claim 하나 삭제 |
| 추가 | 원문에 없는 내용 삽입 |
| 비교 | A > B → B > A |
| 인과관계 | A 때문에 B → B 때문에 A |
| 조건 | A일 때만 B → 항상 B |
| 개체명 | 한국 → 일본 |
| 전문 개념 | correlation → causation |

이런 오류는 전체 문장은 비슷하지만 핵심 의미만 바뀌기 때문에, Embedding + NLI를 같이 사용하는 필요성을 검증하기 좋다.

---
