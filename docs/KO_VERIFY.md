# 한→한 문체 변환 의미 검증 (강희 담당)

이 작업은 **새 파일 4개만 추가**했다. 기존 파일은 하나도 고치지 않았다.

| 파일 | 역할 |
|---|---|
| `src/ko_verify.py` | 검증기 본체. 입력 글과 변환 결과를 비교해 PASS / REVIEW / FAIL |
| `scripts/ko_verify_run.py` | 검증 파이프라인. 팀원 결과 파일을 받아 유출 검사 + 검증 → 판정 파일 |
| `scripts/ko_verify_selftest.py` | 검증기 자체 시험(오탐률, 오류 주입 탐지율) |
| `docs/KO_VERIFY.md` | 이 문서 |

- **API·LLM·torch 없음.** Kiwi(이미 쓰는 형태소 분석기)와 규칙만. 한 건 0.1초 안팎. 같은 입력이면 항상 같은 결과.
- 변환은 1회, 검증도 1회. 다시 생성하는 루프 없음.
- 영어 쪽 `src/verify.py`(Gemini 판정)와는 별개이고 그대로 둔다.

## 역할 나눔

```
[팀원] 짝 예시 DB {AI 글 ↔ 사람 글} 만들기 (Gemini)
            │
[강희] ① pairs : 짝 검사 → 뜻이 바뀐 짝·평가 문서로 만든 짝 제거
            │
[팀원] AI 글 입력 → 짝 예시 검색 → 문체 변환 1회 (Gemini) → 변환결과.jsonl
            │
[강희] ② gen   : 유출 검사 + 의미 검증 → verify.jsonl (PASS/REVIEW/FAIL, use)
            │
[팀원] 작성자 분류 → 정답률 (엄격 = 분류 맞힘 AND PASS)
```

## 팀원에게 받을 파일

### 변환 결과 (한 줄 = 변환 1건)

```json
{"id": "nakyung_doc_009_03|ganghee", "cond": "pairs-R", "target": "ganghee",
 "input": "<변환 입력: AI가 쓴 일반 글>",
 "output": "<변환 결과>",
 "examples": [{"id": "doc_007_08", "ai": "<짝의 AI 글>", "styled": "<짝의 사람 글>"}],
 "source": "<입력이 사람 글을 AI 말투로 바꾼 것이면 그 사람 원문>", "source_person": "nakyung", "source_doc": "doc_009"}
```

| 필드 | 필수 | 설명 |
|---|---|---|
| id | ○ | 고유값 |
| target | ○ | ganghee / nakyung / suyoung |
| input, output | ○ | 변환 입력(AI 글), 변환 결과 |
| cond | | 비교 조건 이름. 요약표가 조건별로 나뉨 |
| examples | | 프롬프트에 실제로 넣은 짝. **id는 `doc_007_08` 같은 문서_청크 형식**이어야 평가 문서 유출을 잡음 |
| source, source_person, source_doc | | 입력이 평가 문서를 AI 말투로 바꾼 것일 때. 정답 유출 검사에 씀 |

### 짝 예시 DB

```json
{"id": "doc_007_08", "styled": "<사람이 쓴 문단>", "ai": "<그 문단을 AI 말투로 바꾼 글>"}
```

## 실행

```bash
# ① 짝 DB 검사 → results/ko_verify/pairs_<user>.csv
python scripts/ko_verify_run.py pairs --pairs <짝.jsonl> --user ganghee

# ② 변환 결과 검증 → results/ko_verify/<tag>/verify.jsonl, verify.csv, summary.md
python scripts/ko_verify_run.py gen --gen <변환결과.jsonl> --tag pairs_v1

# REVIEW를 읽고 verify.csv의 '사람판정' 열에 PASS/FAIL을 적은 뒤 반영
python scripts/ko_verify_run.py gen --gen <변환결과.jsonl> --tag pairs_v1 --override <확인한.csv>

# 동작 확인(기존 results/ko_doc/base 결과로) / 검증기 자체 시험
python scripts/ko_verify_run.py demo
python scripts/ko_verify_selftest.py --show
```

## 팀원에게 넘기는 것: `verify.jsonl`

```json
{"id": "...", "cond": "...", "target": "...", "verdict": "PASS", "auto_verdict": "PASS",
 "use": true, "leak": [], "issues": [...], "stats": {"content_recall": 0.8, "style_change": 0.3, "unchanged": false}}
```

- `verdict`: 최종 판정(사람이 확정한 건 그 값), `auto_verdict`: 검증기 판정
- `use`: false면 유출 → 정답률 계산에서 뺀다
- 정답률 계산 쪽에서 쓰는 식: **엄격 정답률 = (분류 맞힘 AND verdict == PASS) 건수 / use == true 건수**
- `stats.unchanged`가 true면 입력을 거의 그대로 낸 것(어절 97% 이상 같음). 많으면 정답률이 입력 판정과 같아진다

## 판정 기준

| 판정 | 유형 | 어떻게 보나 |
|---|---|---|
| FAIL (critical) | 설명·머리말 섞임, 빈 출력 | 첫머리 "다시 쓴 글:", "다음은 … 바꾼", 줄 첫머리 "원문:/예시1:" (입력에도 있으면 제외) |
| | 숫자 불일치 | 아라비아 숫자 비교(3만 = 30,000으로 환산) |
| | 검색 예시 베낌 | 예시와 5어절 이상 같은 구절(입력엔 없는 것), 또는 새로 생긴 문장의 내용어가 예시에서 옴 |
| REVIEW (major) | 내용 누락 / 추가 | 문장의 내용어(명사·동사·형용사 어간)가 상대 글의 이웃 두 문장 안에 50% 미만 |
| | 부정 뒤집힘 | 부정된 서술어(발생하지 않았다 → '발생')가 대응 문장에서 부정 없이 나옴, 또는 반대 |
| | 확신도 변화 | 추측(수 있다·것 같다·보인다…)이 걸린 서술어가 단정으로, 또는 반대 |
| | 범위어 변화 | 모든·항상·결코·전혀·반드시 등이 생기거나 빠짐 |
| | 고유명사 누락, 전체 내용어 재현율 55% 미만 | |
| PASS (minor만) | 형식 변화, 추측 표현 개수만 다름, 약한 범위어(대체로·자주), 수사, 고유명사 추가 | 기록만 함 |

REVIEW는 "틀렸다"가 아니라 "사람이 볼 것"이다.

## 자체 시험 결과 (`scripts/ko_verify_selftest.py`)

오탐 (뜻이 같다고 보는 쌍인데 PASS가 아님)

| 자료 | n | 오탐 | 비고 |
|---|---|---|---|
| 기존 변환 결과(조금 바꾼 긴 글) | 18 | 1 (6%) | 그 1건은 실제 의미 변화(아래) |
| 중립문↔사람 원문(많이 바꾼 문단, 양방향) | 24 | 12 (50%) | 6쌍 중 약 절반은 중립화 모델이 실제로 뜻을 바꿈, 나머지는 반의어·동의어 바꿔 말하기 |

오류 주입 탐지율 (일부러 넣은 오류를 그 유형으로 잡음)

| 오류 | 많이 바꾼 글 | 조금 바꾼 글 |
|---|---|---|
| 부정 뒤집기 | 82% | 100% |
| 숫자 바꾸기 | 90% | 100% |
| 문장 빼기 | 75% | 100% |
| 다른 글 문장 넣기 | 100% | 94% |
| 예시 문장 베끼기 | 100% | 100% |
| 추측↔단정 | 70% | 100% |
| 범위어 넣기 | 100% | 100% |
| 고유명사 바꾸기 | 91% | 100% |
| 설명 머리말 | 100% | 100% |

규칙으로 넣은 오류는 실제 LLM 오류보다 찾기 쉽다. 실제 성능은 짝 예시 결과의 REVIEW·PASS 일부를 사람이 읽어 확인한다.

실제로 잡은 의미 변화(기존 결과, 나경 R): "자리 잡았다" → "자리 잡았다**고 볼 수 있다**", "다르다" → "다르다**고 할 수 있다**".
나경 문체로 바꾸면서 단정이 추측이 됨(공통 규칙 '확신 정도 유지' 위반).

## 짝 예시라서 특히 조심할 것 (팀원과 공유)

1. **정답 유출**: 평가 입력이 사람 문단 P를 AI 말투로 바꾼 것이고 짝 DB에 (AI(P), P)가 있으면, 검색이 그 짝을 찾아 P를 베낀다 → 정답률이 가짜로 오른다. 짝 DB는 학습 문서(`data/eval/split_info.json`의 heldout 제외)로만 만든다. `gen`이 잡은 건은 `use=false`.
2. **짝의 AI 글도 뜻이 바뀔 수 있다**: 기존 중립화 결과에도 "변화시킬 수 있다 → 변화한다"가 있었다. 뜻이 바뀐 짝을 넣으면 모델이 뜻 바꾸기를 배운다 → `pairs`로 먼저 거른다.
3. **예시 베끼기**: 분류기는 학습한 사람 글을 알아보므로 예시 문장을 베끼면 쉽게 맞는다 → 베낌은 FAIL.

## 한계

- 반의어 바꿔 말하기('얻지 못하다'→'잃다'), 동의어로 많이 바꾼 문장은 오탐하기 쉽다. 문체를 많이 바꿀수록 REVIEW가 는다.
- 같은 낱말로 인과·비교 방향만 바꾼 것("A 때문에 B" → "B 때문에 A")은 못 잡는다.
