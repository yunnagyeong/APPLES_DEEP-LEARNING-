# Style Profile 명세 초안 (팀원① · 1주차 산출물)

이 문서는 1주차 회의에서 팀 전체와 확정할 "스타일 특징 리스트 + 데이터 형식(StyleProfile 스키마)" 초안입니다. 확정되면 `CLAUDE.md`의 인터페이스 섹션으로 옮기고, 2주차부터 `src/style_features.py` 구현의 기준이 됩니다.

## 설계 원칙

이전 대화에서 짚었던 한계(스타일 수치를 LLM에게 "0~1로 평가해줘"라고 시키면 결국 또 다른 AI의 주관적 판단이라 의미 검증 단계와 같은 약점을 반복함)를 피하기 위해, 아래 항목은 전부 **규칙/통계로 직접 계산 가능한 것만** 골랐습니다. LLM은 형태소 분석이 애매한 경우의 보조 판별 정도로만 쓰고, 핵심 점수 자체는 LLM 판단에 의존하지 않습니다.

한국어 형태소 분석은 **Kiwi**(`kiwipiepy`)를 추천합니다. 설치가 `pip install kiwipiepy` 한 줄로 끝나고, 별도 서버 설치가 필요한 KoNLPy 계열보다 가볍게 시작할 수 있습니다.

## 특징 항목 6가지

### 1. 문장 길이 / 복잡도 — `sentence_length_avg`

**정의**: 평균 문장 길이(어절 수). 문장이 길고 복잡할수록 높은 값.

**계산**: Kiwi로 문장 분리 → 각 문장의 어절 수(공백 기준) 세기 → 전체 평균.

```
sentence_length_avg = sum(len(문장.split()) for 문장 in 문장들) / len(문장들)
```

### 2. 격식성 — `formality_score` (0~1)

**정의**: 격식체 종결어미를 쓰는 문장의 비율.

**계산**: 종결어미가 `-습니다/-합니다/-입니다/-됩니다` 패턴(정규식)에 해당하는 문장 수 ÷ 전체 문장 수.

```
formality_score = 격식체_종결_문장_수 / 전체_문장_수
```

### 3. 단정성 — `directness_score` (0~1)

**정의**: 완곡 표현 없이 단정적으로 끝맺는 문장의 비율. hedging_rate(아래 5번)와 반비례 관계지만, 완곡 표현이 전혀 없어도 어미 자체가 단정형인지("~이다/~하다")는 별도로 볼 가치가 있어 독립 지표로 둡니다.

**계산**: 단정형 종결(`-이다`, `-하다`, `-한다` 등) 문장 수 ÷ 전체 문장 수.

### 4. 논리 명시성 — `logical_connective_rate`

**정의**: 문장 간 논리 관계를 접속 표현으로 얼마나 명시적으로 드러내는가.

**계산**: 접속 표현 사전에 매칭되는 표현의 총 등장 횟수 ÷ 전체 문장 수.

```
접속_표현_사전 = ["그러므로", "따라서", "하지만", "그러나", "왜냐하면",
                "즉", "반면에", "그리고", "결과적으로", "그럼에도"]
```

### 5. Hedging 빈도 — `hedging_rate`

**정의**: 주장을 완곡하게 제한하는 표현의 빈도.

**계산**: hedging 표현 사전 매칭 횟수 ÷ 전체 문장 수.

```
hedging_사전 = ["~것 같다", "~수 있다", "~수도 있다", "~일 수 있다",
              "아마", "대체로", "~인 듯하다", "~라고 볼 수 있다"]
```

### 6. 구조 패턴 — `conclusion_position`

**정의**: 결론을 글의 앞부분에 먼저 제시하는지, 뒷부분에 배치하는지.

**계산**: 결론 표지어(`"결론적으로"`, `"요컨대"`, `"따라서"` 등)가 처음 등장하는 문단이 전체 문단 중 앞쪽 1/3인지 뒤쪽 1/3인지로 `"front"` / `"back"` / `"mixed"` 세 값 중 하나로 분류.

> 전문어 사용 패턴(전문용어 뒤에 설명을 붙이는 습관)은 규칙 기반으로 안정적으로 잡기 까다로워 1주차 초안에서는 제외했습니다. 위 6개로 먼저 파이프라인을 완성한 뒤, 여유가 되면 2차 항목으로 추가하는 걸 제안합니다 — 이 부분은 1주차 회의 안건으로 올려서 팀 의견을 들어보면 좋겠습니다.

## 추가로 검토할 항목

6개로 최소 동작 파이프라인은 충분히 만들 수 있지만, 몇 가지는 넣을지 검토해볼 가치가 있습니다. 항목이 늘어날수록 개인 데이터가 적은 팀원의 경우 항목별 수치가 불안정해질 위험도 같이 커지므로, 전부 넣기보다 아래에서 우선순위 순으로 골라 담는 걸 권장합니다.

### 7. 예시 사용 빈도 — `example_usage_rate`
사실 이건 원래 대화에서 팀이 언급했던 항목("예시 사용 빈도 등")인데 1주차 초안에서 제가 빠뜨렸습니다. "예를 들어", "가령", "예컨대" 같은 표지어의 문장당 등장 빈도로, 접속 표현·hedging과 동일한 사전 매칭 방식이라 구현 비용이 거의 없습니다. 6개 항목에 그대로 추가하는 걸 권장합니다.

### 8. 어휘 다양성 — `lexical_diversity`
(MTLD, Measure of Textual Lexical Diversity. 처음에는 TTR(서로 다른 단어 수 ÷ 전체 단어 수)이었으나, TTR은 글이 길수록 기계적으로 떨어져 코퍼스 길이가 다른 사람끼리 비교할 수 없어서 2026-10-20에 MTLD로 교체했다. 키 이름은 그대로. 값은 0~1이 아니라 대략 수십 단위)
반복적으로 같은 단어를 쓰는지, 어휘가 풍부한지를 잡아냅니다. 별도 사전이 필요 없어 구현이 간단하고, 특히 **7주차 평가 때 쓸 "AI 획일화 정량 측정"(어휘 다양성 비교)과 계산 로직을 그대로 공유**할 수 있어 팀원③의 작업과 코드 중복을 줄여줍니다.

### 9. 평균 문단 길이 — `paragraph_length_avg`
(문단당 평균 문장 수)
`conclusion_position`이 "결론이 어디 있는가"만 보는 반면, 이건 "얼마나 잘게 문단을 쪼개는가"를 보는 항목이라 구조 관련 정보를 하나 더 보강합니다.

**여유가 되면 고려할 것 (구현 비용 중간)**

- **피동 표현 빈도 — `passive_expression_rate`** (`-되다`, `-지다`, `-당하다` 등 피동형 어미 빈도). 사전 매칭만으로 가능하지만, 오탐(피동이 아닌데 형태가 비슷한 경우)이 있어 검증이 좀 더 필요합니다.
- **수식어 밀도 — `modifier_density`** (형용사/부사 비율). Kiwi가 품사 태깅을 같이 해주므로 사전 없이도 계산 가능하지만, 정확도를 위해 실제 데이터로 확인하는 과정이 필요합니다.



## 10. 기능어 선택 프로필 — `function_word_choice` (24번, 2026-10-23)

같은 기능(대조·결론·이유·목적)을 어떤 표현으로 쓰는지의 비율입니다. 값은 Kiwi 형태소 1000개당 등장 횟수이고, 규칙은 (형태, 품사) 매칭입니다(LLM 판단 없음). 그룹은 `style_features.FUNCTION_WORD_GROUPS`:

| 그룹 | 표현 (형태, 품사) | 영어 지시 표현 |
|---|---|---|
| however | 그러나(MAJ) | however |
| but | 하지만·그렇지만(MAJ) | but |
| whereas | 반면(NN*) | whereas / on the other hand |
| therefore | 따라서·그러므로(MAJ) | therefore / thus |
| causal | -므로·-으므로·-어서·-아서(EC), 그래서(MAJ) | since / because |
| purpose_formal | -고자(EC) | in order to |

## 지시문 생성 기준 (표현 층위 / 구성 층위)

지시문은 **원문이 정해주지 않는 선택**에만 개인 지시를 쓰고, 원문에 이미 드러나는 성질은 모든 사람에게 공통인 **보존 규칙**으로 처리합니다(D-006·D-007).

- **효과 크기 d** = (본인 문서별 평균 − 나머지 두 사람의 문서별 평균) / 사람 안 표준편차(pooled within SD). 기준 분포는 `data/style_reference.json`(`scripts/build_style_reference.py`로 재생성, 코퍼스가 바뀌면 재실행). 팀 밖 사용자는 프로필 값을 세 사람 문서 평균과 비교(부호 유지율 없음).
- **지시 조건**: |d| ≥ 0.5(`STYLE_D_THRESHOLD`) **그리고** 문서 재표본 2000회 중 부호 유지 ≥ 90%(`STYLE_SIGN_KEEP_MIN`).
- **지시문은 보존형·선택형 문구만 씁니다**: 문장 길이 d≥+ → "Do not compress or simplify the source's long sentences"(d≤− 줄 "Do not pad or expand …"은 28번에서 조건 간 차이가 없어 D-013에서 삭제); MTLD d≥+ → "Where the source uses distinct terms, keep them distinct…", d≤− → "Use consistent, repeated key terms for clarity"; 접속어 총량은 지시하지 않음(원문에 없던 인과 주장을 만들 수 있음, 공통 규칙이 보존 담당).
- **기능어 선택 지시**(`FUNCTION_WORD_INSTRUCTIONS` 6줄): d가 양수인 그룹 중 d가 큰 상위 2개(`FUNCTION_WORD_MAX_LINES`), however/but은 큰 쪽 하나. 적게 쓰는 방향(d<0)은 지시하지 않음. 문구는 모두 "Where the source <기능>, render it with …" 형태. **적용 조건(D-013)**: 번역 입력(`source_text`)에 그 기능의 한국어 표현이 실제로 있을 때만 줄을 넣음 (`style_features.FUNCTION_WORD_GATE`, Kiwi 규칙. 모델이 "원문에 있을 때만"을 지키지 못해 28번에서 "-며"→therefore, "만들고"→in order to 같은 의미 변경이 나왔기 때문). 대조(however/but)는 그러나·하지만·그렇지만 어느 것이든, causal은 -므로·-으므로·그래서만(-어서·-아서는 시간 순서와 섞여 제외, 지표 계산은 그대로). 조건을 먼저 거른 뒤 상위 2개를 고름. `source_text`를 안 주면 조건 없이 전부 넣음(프로필만 보는 확인용).
- **공통 보존 규칙 3개**(전원 동일, 번역·재번역 프롬프트): 헤징·한정어 보존(`HEDGE_PRESERVATION_RULE`), 문장 경계 유지, 접속·전환 표현 보존(`STRUCTURE_PRESERVATION_RULES`).
- **지시문에서 제외**(프로필 계산은 유지, 설명용): `directness_score`(서술어 종결 유형이지 직설성이 아님), `conclusion_position`, `example_usage_rate`, `paragraph_length_avg`(구성 층위, 재배열·예시 추가를 유도하거나 원문 문단 구분 차이). `formality_score`는 합쇼체 비율 ≥ 0.7일 때만 "formal tone".
- **한계**: 사람당 15~31편이라 d의 구간이 넓음. 한국어 효과 크기·기능어 선택을 영어 지시로 옮기는 대응(그러나→however 등)은 검증되지 않았고, 기능어 그룹 선정에 선택 편향이 있음. 지시문이 번역에 실제로 반영되는지는 `scripts/compare_structure_rules.py`(조건 C 대 D)로 확인 중.
- **단어 리스트의 역할 구분**(B-015, `docs/style_word_lists.md`): 지시문을 실제로 만드는 리스트는 `FUNCTION_WORD_GROUPS` 하나뿐입니다. `LOGICAL_CONNECTIVES`·`HEDGING_EXPRESSIONS`·`EXAMPLE_MARKERS`·`CONCLUSION_MARKERS`는 프로필 설명용 지표 계산에만 쓰이고 지시문에는 쓰이지 않습니다(헤징은 공통 보존 규칙으로 처리). 알려진 문제: `따라서`가 접속어·결론 표지어·`therefore` 그룹에 중복, `수 있다`가 가능·능력 의미까지 세어 헤징 점수가 부풀 수 있음(추측형 `수도 있다`·`일 수 있다`·`-었을 수 있다`로 좁히는 안은 발표 자료용으로 보류), `가령`·`예컨대`·`인 듯하다`·`요컨대`는 코퍼스에서 0회. 기능어 그룹의 정밀도는 `scripts/make_wordlist_check_sheet.py`로 만든 시트를 팀원 2명이 O/X로 점검해 확인합니다.
- **장르 한계**(B-014): 지시 기준 d는 에세이+보고서를 합쳐 계산했습니다. 장르를 통제해도 작성자 차이는 유지되지만(장르 설명 비율 ≤0.08), 다중 비교 보정(p<0.0017, 검정 30개) 기준으로는 however(두 장르 모두)만 두 장르에서 유지되고 therefore·causal·고자·but·directness는 보고서 안에서만 보입니다. 같은 "보고서"라도 강희는 IT, 수영·나경은 인문이라 주제·수업 관습은 통제하지 못했으므로, 보고서 전용 신호가 개인 문체인지 수업 습관인지는 구분할 수 없습니다.
- **작성자 식별 검증**(`scripts/stylometry_check.py`, `scripts/validate_function_word_choice.py`): 기능 형태소 top100 LOO 0.90 / 교차 장르 0.76, 기능어 6그룹만 0.81 / 0.75 (소표본, 선택 편향 있음).

## StyleProfile 스키마 (초안)

팀원②의 검증 모듈, 팀원③의 평가 스크립트가 이 형식을 그대로 참조하게 됩니다. 회의에서 이 스키마 자체(키 이름, 타입)를 먼저 못박아두는 게 이번 주 가장 중요한 결정입니다.

```python
StyleProfile = {
    "sentence_length_avg": float,      # 평균 어절 수
    "formality_score": float,          # 0~1
    "directness_score": float,         # 0~1
    "logical_connective_rate": float,  # 문장당 접속 표현 개수
    "hedging_rate": float,             # 문장당 hedging 표현 개수
    "conclusion_position": str,        # "front" | "back" | "mixed"
    "example_usage_rate": float,       # 문장당 예시 표지어 개수 (추가 권장 항목)
    "lexical_diversity": float,        # MTLD, 값이 클수록 다양 (0~1 아님. 추가 권장 항목 — 7주차 AI 획일화 지표와 로직 공유)
    "paragraph_length_avg": float,     # 문단당 평균 문장 수 (추가 권장 항목)
    "function_word_choice": dict,      # {그룹: 형태소 1000개당 횟수} 6그룹 (24번, 위 10번 항목)
    "source_example_count": int,       # 이 프로파일을 뽑은 데 사용한 원문 개수 (표본 크기 투명성용)
}
```
