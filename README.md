# LinguaMe — APPLES DEEP LEARNING Team

한국어 글을 **그 사람의 문체로** 영어로 번역하고, 번역이 원문의 의미를 바꾸지 않았는지 자동으로 검증하는 프로젝트입니다.

## 바로 실행하기

```bash
pip install -r requirements.txt
cp .env.example .env          # .env 안의 GOOGLE_API_KEY를 본인 Gemini API 키로 바꾸기
streamlit run app.py
```

- 처음 버튼을 누를 때 형태소 분석기(Kiwi)와 스타일 프로필을 한 번 준비합니다(몇 초).
- API 키 없이 화면만 보려면 사이드바의 "데모용 가짜 결과"를 켜세요.
- 터미널에서 한 문장만 확인: `python -m src.cli translate --user ganghee --text "이 결과는 우연일 수도 있다."`

## 파이프라인 (조건 C)

```
한국어 입력
 → 본인 한국어 예시 검색(RAG, 임베딩 코사인 Top-4)
 → 문체 지시문(팀원 대비 효과크기 d) + 의미 보존 공통 규칙
 → 번역 (gemini-3.5-flash-lite)
 → 의미 검증 v2: claim 정렬 판정(gemini-3.6-flash 1회) + 숫자·유사도 규칙 검사
 → PASS / REVIEW / FAIL + Meaning Score(0~100)
 → REVIEW·FAIL이면 문제 구절만 고치는 재번역 → 다시 검증 → 가장 좋은 시도 반환
```

설계 근거와 7주차 대비 바뀐 점은 [docs/PIPELINE_V2.md](docs/PIPELINE_V2.md).

## 비교 조건 (app.py)

| 조건 | 내용 |
|---|---|
| A · 일반 번역 | 예시·지시·검증 없이 번역 |
| B · 단순 개인화 | 검색한 본인 한국어 예시만 넣고 번역 |
| C · 우리 팀 방법 | 위 파이프라인 전체 |

## 자주 쓰는 스크립트

| 목적 | 명령 |
|---|---|
| 검증기 평가: 새 평가셋 64건 (긴 문단 포함) | `python scripts/eval_verifier.py --heldout --repeat 2` |
| 검증기 평가: 기존 사람 라벨 32건 | `python scripts/eval_verifier.py` |
| 긴 문단 포함 전체 파이프라인 | `python scripts/run_condition_c.py --user ganghee --input data/measure/heldout_sources.jsonl --out results/v2_heldout` |
| 조건 C 일괄 실행 | `python scripts/run_condition_c.py --user ganghee [--workers 4]` → `results/v2/` |
| 호출 수 확인(API 없음) | `python scripts/measure_pipeline.py --mode offline` |
| 예시 재색인(예시 추가 후) | `python scripts/convert_data_to_examples.py <user>` → `python scripts/build_index.py <user>` |
| 문체 기준 분포 재계산 | `python scripts/build_style_reference.py` |

## 설정 (src/config.py)

- `VERIFY_MODEL` 검증 판정 모델. `VERIFY_FALLBACK_MODEL`(기본 None)을 정하면 판정 모델 한도가 끝났을 때 그 모델로 대체(결과에 `judge_model`로 표시). 유료 API는 None 권장.
- `VERIFY_THINKING_LEVEL` 판정 모델의 생각 길이(`"low"`면 빠름, 지원 안 하는 모델이면 자동 무시).
- `SIMILARITY_GUARD` 원문·번역문 임베딩 유사도 하한. `DEFAULT_MAX_RETRIES` 최대 재번역 횟수(앱·CLI·스크립트 공통, 기본 2).

## 레거시 (7주차)

7주차 결과(`results/week7/`)는 NLI 기반 검증(`src/verify_legacy.py`)으로 만든 것입니다. 다시 돌리려면
`pip install -r requirements-nli.txt` 후 `scripts/run_meaning_evaluation.py` 등 레거시 스크립트를 쓰세요.

## 역할 분담
- **수영**: 문체 프로파일 추출 (`src/style_features.py`) & 개인화 번역 생성 (`src/generate.py`)
- **강희**: 의미 보존 검증 파이프라인 (`src/verify.py`, `src/meaning_score.py`)
- **나경**: 데이터 수집 (`data/`), 평가, 데모 UI (`app.py`)
- **공통**: RAG 인프라 (`src/embeddings.py`, `src/store.py`, `src/retrieve.py`)
