"""한→한 문체 변환 경량 의미 검증 (API·LLM·torch 없음, Kiwi + 규칙).

왜 영어용(src/verify.py)과 따로 두나
- 한→영은 언어가 달라 단어를 직접 비교할 수 없어 판정 모델(Gemini) 호출이 필요했다.
  한→한은 같은 언어라 원문·변환문의 내용어(명사·동사·형용사 어간)·숫자·고유명사·부정·추측 표현을
  형태소 단위로 직접 맞대 볼 수 있다. 그래서 모델 판정 없이 규칙만으로 검사한다.
- 한→한 변환은 한 번만 생성하고(재생성 루프 없음) 이 검사도 한 번만 돈다. 결과는 평가에서
  "의미가 유지된 변환만 정답으로 인정"하는 관문으로 쓴다(scripts/ko_verify_run.py → 팀원의 정답률 계산).
- 필요한 것: kiwipiepy(이미 의존성, 모델 파일 약 100MB, CPU). 문단·문서 1편 검사 0.1초 안팎, 최초 로드 약 1초.

검사 항목 → 오류 유형 → 심각도(고정 표) → 판정
  critical(FAIL) : 출력에 설명·머리말이 섞임, 빈 출력, 아라비아 숫자 불일치, 검색 예시 문장을 그대로 베낌
  major(REVIEW)  : 원문 문장 내용 누락, 원문에 없는 문장 추가, 고유명사 누락, 부정 뒤집힘(부정됐던 서술어가
                   부정 없이 나오거나 그 반대),
                   확신도 변화(추측이 걸린 서술어가 단정으로 나오거나 그 반대), 강한 범위어(모든·항상·결코 등) 변화,
                   전체 내용어 재현율이 매우 낮음
  minor(PASS)    : 추측 표현 개수만 다름(서술어 단위 변화 없음), 두 글자 한글 고유명사 누락(Kiwi 오분석 잦음),
                   약한 범위어(대체로·자주 등) 변화, 고유어 수사(세 가지 등) 차이,
                   고유명사 추가(환각→Hallucination처럼 표기만 바꾸는 것도 문체라서. 예시에서 온 이름이면 critical),
                   형식 변화
임계값은 중립문↔사람 원문 12쌍(results/ko_style/neutral.jsonl)과 기존 변환 결과 18건(results/ko_doc/base)을
보며 정했다. 같은 자료로 시험하면 낙관적이므로, 실제 짝 예시 변환 결과는 REVIEW 일부를 사람이 읽어 오탐을 센다.
자체 시험: python scripts/ko_verify_selftest.py (오탐률·오류 주입 탐지율)

한계(규칙이라 못 하는 것): 반의어 바꿔 말하기('얻지 못하다'→'잃다')는 부정 뒤집힘으로 오탐하기 쉽고,
동의어로 많이 바꾼 문장은 누락·추가로 오탐하기 쉽다. 반대로 같은 낱말을 쓰며 인과·비교 방향만 바꾼 것은 못 잡는다.
그래서 REVIEW는 '사람이 볼 것'이지 '틀림'이 아니다. 문체를 많이 바꿀수록 REVIEW가 늘어난다.

반환 dict
  verdict  PASS | REVIEW | FAIL
  issues   [{type, severity, label_ko, note, evidence}]
  stats    content_recall(원문 내용어가 남은 비율), content_precision(변환문 내용어 중 원문에 있던 비율),
           style_change(기능 형태소가 바뀐 비율, 0=그대로), unchanged(거의 그대로 복사했는가),
           sentences(원문→변환문 문장 수)
  format   src/ko_format.check 결과(형식 검사, 의미와 별개)
"""
from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache

# ---------------------------------------------------------------- 심각도 표 (고정 규칙)
SEVERITY = {
    "EMPTY_OUTPUT": "critical",
    "META_TEXT": "critical",
    "NUMBER_MISMATCH": "critical",
    "EXAMPLE_COPY": "critical",
    "OMISSION": "major",
    "ADDITION": "major",
    "LOW_RECALL": "critical",   # 원문 내용 절반 가까이가 사라짐(잘린 재번역 등) → 재번역 (11-19 파트너 제안)
    "TRUNCATED": "critical",
    "EXAMPLE_ENTITY": "major",
    "ENTITY_MISSING": "major",
    "NEGATION_FLIP": "major",
    "MODALITY_SHIFT": "major",
    "SCOPE_CHANGE": "major",
    "MODALITY_MINOR": "minor",
    "SCOPE_MINOR": "minor",
    "NATIVE_NUMBER": "minor",
    "ENTITY_ADDED": "minor",
    "ENTITY_WEAK": "minor",
    "PARAPHRASE": "minor",
    "FORMAT": "minor",
}
LABEL_KO = {
    "EMPTY_OUTPUT": "빈 출력", "META_TEXT": "설명·머리말 섞임", "NUMBER_MISMATCH": "숫자 불일치",
    "EXAMPLE_COPY": "검색 예시 베낌", "OMISSION": "내용 누락", "ADDITION": "원문에 없는 내용 추가",
    "LOW_RECALL": "전체 내용 이탈", "TRUNCATED": "글이 잘림", "EXAMPLE_ENTITY": "예시에 있던 고유명사 생김(오분석일 수 있음)", "ENTITY_MISSING": "고유명사 누락", "ENTITY_ADDED": "고유명사 추가", "ENTITY_WEAK": "두 글자 고유명사 누락", "PARAPHRASE": "많이 바꿔 씀(뜻 확인 권장)",
    "NEGATION_FLIP": "부정 뒤집힘", "MODALITY_SHIFT": "확신도 변화", "SCOPE_CHANGE": "범위 표현 변화",
    "MODALITY_MINOR": "추측 표현 개수 차이", "SCOPE_MINOR": "약한 범위어 변화", "NATIVE_NUMBER": "수사 차이",
    "FORMAT": "형식 변화",
}
RANK = {"none": 0, "minor": 1, "major": 2, "critical": 3}

# ---------------------------------------------------------------- 임계값
SENT_COVERAGE_MIN = 0.50   # 문장 내용어 중 상대 글의 이웃한 두 문장 안에 있는 비율이 이보다 낮으면 누락/추가
                           # (정상 쌍 441문장의 2백분위 0.57 / 빼 본 문장의 75백분위 0.38)
SENT_MIN_LEMMAS = 3        # 내용어가 이보다 적은 짧은 문장은 문장 단위 검사 생략
RECALL_MIN = 0.55          # 글 전체 내용어 재현율 하한 (정상 쌍 최소 0.63)
COPY_NGRAM = 5             # 예시와 이 길이(어절) 이상 같고 원문에는 없는 구절 = 베낌
EXAMPLE_CONTENT_MIN = 0.60  # 추가로 표시된 문장의 내용어가 예시에 이만큼 있으면 예시에서 온 내용
UNCHANGED_RATIO = 0.97     # 어절 열 일치도가 이 이상이면 '거의 그대로'
REWRITE_MIN = 0.25          # 예시 내용어 베낌 판단 전에 '원문 문장을 바꿔 쓴 것'으로 볼 겹침(표본: 바꿔 씀 0.33~0.54, 끼워 넣은 예시 문장 ~0.1)
TRUNC_RATIO = 0.70          # 변환문/원문 글자 수가 이보다 작으면 잘린 것으로 봄(FAIL)
TRUNC_MIN_SOURCE = 200      # 짧은 글은 길이 비가 흔들려 이 길이 이상일 때만 봄
PARA_MIN_SIM = 0.20         # 같은 자리 '누락'·'추가'를 바꿔 쓴 짝으로 볼 최소 겹침(표본: 바꿔 씀 0.2~0.5, 남의 문장 ~0)
ALIGN_MIN = 0.60           # 부정·추측 뒤집힘은 내용어가 이만큼 겹치는 대응 문장끼리만 비교(없으면 판단 안 함)

# 내용어: 명사·수사·외국어·숫자·어근·동사·형용사 어간. 너무 흔한 어간은 뺀다.
CONTENT_TAGS = ("NNG", "NNP", "NR", "SL", "SN", "XR", "VV", "VA")
STOP_STEMS = {"하", "되", "있", "없", "같", "이", "그렇", "이렇", "저렇", "어떻", "대하", "위하", "통하",
              "따르", "의하", "않", "보", "가지", "나", "들", "그러", "이러"}
NEG_MORPHS = {("않", "VX"), ("못", "MAG"), ("안", "MAG"), ("없", "VA"), ("없이", "MAG"), ("아니", "VCN"), ("말", "VX"),
              ("못하", "VX"), ("아니하", "VX"), ("아니", "MAG")}  # 아니하다·아니 가다: 옛 표기(이광수 등)
# 수사 의문 어미: '~아니겠는가'(= 강한 긍정), '어찌 ~ 있으랴'(= 부정). 질문형으로 끝나면 수사 의문으로 본다
# (입력이 AI가 쓴 평서문이므로 변환문에 생긴 질문은 거의 수사 의문이다. 같은 규칙을 양쪽에 똑같이 적용).
_RHET_EF = {"는가", "ᆫ가", "은가", "던가", "나", "냐", "느냐", "니", "랴", "으랴", "리오", "으리오", "리요", "으리요",
            "누", "는고", "ᆫ고", "ᆯ까", "을까", "까", "쏘냐", "소냐", "ᆯ쏘냐"}
_MERELY = {"단순히", "단순한", "단순", "그냥", "단지", "한낱", "비단", "그저"}  # '단순히 X가 아니다' 꼴의 대조
_EMPH_NOUN = {"나위", "틀림", "틀리", "다름", "다르", "여지", "그지"}       # 두말할 나위가 없다, 틀림이 없다 (강조)
_EMPH_VERB = {"피하", "부정", "부인", "거부", "어쩌", "마다하", "의심"}
_LITOTES = {"가볍", "만만하", "만만", "녹록하", "녹록", "예사롭", "심상하", "하찮"}  # 가볍지 않은 = 중요한
# 낱말 속 '없'(보잘것없다·철없다·쓸데없다)과 부사 '~ 없이'(쉴 새 없이·가림 없이·할 것 없이)는 부정이 아니라 한 낱말
_LEX_NEG_NOUN = {"철", "쓸데", "상관", "관계", "형편", "터무니", "어처구니", "아랑곳", "속절", "하염", "가", "거침", "끝",
                 "새", "예외", "틈", "쉼", "아낌", "어김", "빠짐", "남김", "끊임", "가감", "탈", "틀림", "가림", "망설임", "주저",
                 "가차", "여지", "사정", "막힘"}  # 피할 수 없다, 부정하지 못하다 (강조)
_CLASSIC_RHET = {"랴", "으랴", "리오", "으리오", "리요", "으리요", "쏘냐", "소냐", "ᆯ쏘냐"}  # 부정 없이도 수사 의문(=부정)
HEDGE_WORDS = {"아마", "어쩌면", "아마도", "추정", "짐작", "가능성"}
STRONG_SCOPE = {("모든", "MM"), ("모두", "MAG"), ("항상", "MAG"), ("언제나", "MAG"), ("늘", "MAG"), ("결코", "MAG"),
                ("절대", "MAG"), ("절대로", "MAG"), ("전혀", "MAG"), ("반드시", "MAG"), ("오직", "MAG"),
                ("무조건", "MAG"), ("틀림없이", "MAG")}  # '분명히'는 '분명히 하다'(명확히 하다)가 흔해 뺌
# 같은 뜻 범위어는 한 묶음으로 센다('언제나'→'늘'은 바꿔 말하기). '결코·전혀·절대'는 이미 있는 부정을 세게 할 뿐
# 범위를 바꾸지 않으므로(부정 자체가 바뀌면 부정 검사가 잡음) 약한 쪽(minor)으로 센다.
_SCOPE_GROUP = {"모든": "모두", "모두": "모두", "항상": "항상", "언제나": "항상", "늘": "항상", "반드시": "반드시",
                "무조건": "반드시", "틀림없이": "반드시", "오직": "오직", "결코": "결코", "절대": "결코", "절대로": "결코",
                "전혀": "결코"}
_SCOPE_MINOR_GROUPS = {"결코"}
WEAK_SCOPE = {("대체로", "MAG"), ("주로", "MAG"), ("흔히", "MAG"), ("종종", "MAG"), ("자주", "MAG"), ("가끔", "MAG"),
              ("때로", "MAG"), ("거의", "MAG"), ("몇몇", "MM"), ("일부", "NNG"), ("대부분", "NNG"), ("단지", "MAG")}

META_RX = re.compile(  # 첫머리의 머리말·제목, 아무 줄의 '원문:'·'예시1:'(프롬프트 되풀이), 끝맺음 인사
    r"^\s*(#{1,6}\s|\*\*?(원문|변환|다시 쓴)|(다시 쓴 글|변환(된|한)? ?글|결과|출력)\s*[:：])"
    r"|(^|\n)\s*(원문|예시\s?\d*)\s*[:：]"
    r"|^\s*(네|물론|알겠습니다|좋습니다)[,.!]\s"
    r"|다음은[^.\n]{0,40}(다시 쓴|변환한|바꾼|재작성한)"
    r"|(다시 썼|변환했|바꿨)습니다[.!]?\s*$"
)
_KO_UNIT = {"천": 1_000, "만": 10_000, "억": 100_000_000, "조": 1_000_000_000_000}
_NUM_RX = re.compile(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s?([천만억조])?")
_NATIVE = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5, "여섯": 6,
           "일곱": 7, "여덟": 8, "아홉": 9, "열": 10, "스무": 20, "스물": 20}
_NATIVE_RX = re.compile(r"(?<![가-힣])(" + "|".join(sorted(_NATIVE, key=len, reverse=True)) +
                        r")\s?(?:가지|개|명|번|살|시간|달|해|권|편|장|차례|배|단계|군데|마리|주|층|줄|쪽|부분)")
# 한자어 수사를 한글로 쓴 것('100년'→'백 년', '2호선'→'이 호선'). 원문 아라비아 숫자가 빠졌는지 볼 때만 쓴다
# (이것으로 새 불일치를 만들지는 않으므로 '이 호선'의 '이'를 지시어로 잘못 읽어도 FAIL이 생기지 않음).
_SINO_D = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_SINO_RX = re.compile(r"(?<![가-힣])((?:[이삼사오육칠팔구]?천)?(?:[이삼사오육칠팔구]?백)?(?:[이삼사오육칠팔구]?십)?[일이삼사오육칠팔구]?)"
                      r"\s?(?=년|호선|퍼센트|프로|세기|층|개월|주년|월|원|명|회|차|대|위|등|번째|위)")


def _sino_value(w: str) -> int:
    total, cur = 0, 0
    for ch in w:
        if ch in _SINO_D:
            cur = _SINO_D[ch]
        else:
            total += (cur or 1) * {"십": 10, "백": 100, "천": 1000}[ch]
            cur = 0
    return total + cur


_ALIAS_RX = re.compile(r"([가-힣]{2,})\s*\(([A-Za-z][\w .\-]*)\)|([A-Za-z][\w.\-]*)\s*\(([가-힣]{2,})\)")


# ---------------------------------------------------------------- 형태소 분석 (글마다 한 번)
def _kiwi():
    """프로젝트 공용 Kiwi 인스턴스(ko_format·ko_style과 같은 것, 메모리에 하나만)."""
    from .style_features import _get_kiwi
    return _get_kiwi()


@lru_cache(maxsize=512)
def _analyze(text: str) -> dict:
    kiwi = _kiwi()
    sents = [s.text.strip() for s in kiwi.split_into_sents(text.strip()) if s.text.strip()]
    toks = [(t.form, t.tag) for t in kiwi.tokenize(text)]
    sent_toks = [[(t.form, t.tag) for t in kiwi.tokenize(s)] for s in sents]
    sent_lem = [set(_lemmas(st)) for st in sent_toks]
    return {"sents": sents, "toks": toks, "sent_toks": sent_toks, "sent_lem": sent_lem}


@lru_cache(maxsize=8192)
def _hanja_reading(form: str) -> str:
    """한자 단어의 한글 읽기('文學'→'문학'). hanja 패키지가 없으면 그대로 둔다(한자 출력에 누락·추가 오탐이 남음)."""
    try:
        import hanja
    except ImportError:
        return form
    return hanja.translate(form, "substitution")


def hanja_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("hanja") is not None


def _lemmas(toks) -> list[str]:
    """내용어. 한자(SH)는 한글 읽기로 바꿔 넣는다 — 변환문이 '文學'으로 써도 원문 '문학'과 맞대 볼 수 있게."""
    out = []
    for f, t in toks:
        if t == "SH":
            r = _hanja_reading(f)
            if len(r) >= 2:
                out.append(r)
        elif t.startswith(CONTENT_TAGS) and f not in STOP_STEMS:
            out.append(f)
    return out


def _present(lemma: str, pool: set[str]) -> bool:
    """정확히 같거나, 두 글자 이상끼리 한쪽이 다른 쪽을 포함(대물림/물림, 적응/적응력)."""
    if lemma in pool:
        return True
    return len(lemma) >= 2 and any(len(p) >= 2 and (lemma in p or p in lemma) for p in pool)


def _coverage(lemmas: list[str], pool: set[str]) -> float:
    s = set(lemmas)
    return sum(_present(l, pool) for l in s) / len(s) if s else 1.0


def _multiset_recall(a: list[str], b: list[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values()) / max(1, sum(ca.values()))


# ---------------------------------------------------------------- 개별 검사
def _issue(kind: str, note: str, evidence: str = "", where: str = "") -> dict:
    """where: evidence가 어느 글의 문장·구절인지("source"=입력, "output"=변환문, ""=근거 문장 없음). 화면 강조용."""
    return {"type": kind, "severity": SEVERITY[kind], "label_ko": LABEL_KO[kind], "note": note,
            "evidence": evidence[:600], "where": where if evidence else ""}


def _numbers(text: str) -> tuple[Counter, Counter]:
    arabic = Counter()
    for m in _NUM_RX.finditer(text):
        v = float(m.group(1).replace(",", "")) * _KO_UNIT.get(m.group(2) or "", 1)
        arabic[f"{v:g}"] += 1
    native = Counter(f"{_NATIVE[m.group(1)]:g}" for m in _NATIVE_RX.finditer(text))
    for m in _SINO_RX.finditer(text):
        if m.group(1):
            native[f"{_sino_value(m.group(1)):g}"] += 1
    return arabic, native


def check_numbers(src: str, out: str) -> list[dict]:
    (sa, sn), (oa, on) = _numbers(src), _numbers(out)
    s_all, o_all = sa + sn, oa + on  # '3가지'↔'세 가지'처럼 표기만 바뀐 것은 같은 수로 본다
    issues = []
    missing = sorted(set(sa) - set(o_all))
    extra = sorted(set(oa) - set(s_all))
    if missing or extra:
        note = "; ".join(x for x in (f"원문 숫자 {', '.join(missing)} 없음" if missing else "",
                                     f"원문에 없는 숫자 {', '.join(extra)}" if extra else "") if x)
        issues.append(_issue("NUMBER_MISMATCH", note))
    n_miss, n_extra = set(sn) - set(o_all), set(on) - set(s_all)
    if n_miss or n_extra:
        issues.append(_issue("NATIVE_NUMBER", f"수사 차이: 원문만 {sorted(n_miss)} / 변환문만 {sorted(n_extra)}"))
    return issues


def _entities(text: str, toks) -> set[str]:
    return {f for f, t in toks if (t == "NNP" and len(f) >= 2) or (t == "SL" and len(f) >= 2)}


def _aliases(*texts: str) -> dict[str, set[str]]:
    """'제미니(Gemini)'처럼 괄호로 병기한 이름은 서로 같은 것으로 본다."""
    al: dict[str, set[str]] = {}
    for t in texts:
        for m in _ALIAS_RX.finditer(t):
            a, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            al.setdefault(a.lower(), set()).add(b.lower())
            al.setdefault(b.lower(), set()).add(a.lower())
    return al


def check_entities(src: str, out: str, s_toks, o_toks, example_text: str = "") -> list[dict]:
    al = _aliases(src, out)
    lo_src, lo_out = src.lower().replace(" ", ""), out.lower().replace(" ", "")  # '황화 수소' = '황화수소'

    def found(e: str, hay: str) -> bool:
        return e.lower().replace(" ", "") in hay or any(a.replace(" ", "") in hay for a in al.get(e.lower(), ()))

    missing = sorted(e for e in _entities(src, s_toks) if not found(e, lo_out))
    added = sorted(e for e in _entities(out, o_toks) if not found(e, lo_src))
    issues = []
    # 두 글자 한글 NNP는 Kiwi가 일반명사(송신 등)를 잘못 붙이는 일이 잦아 경고(minor)로만 둔다
    no_latin_out = not re.search(r"[A-Za-z]", out)
    # 두 글자 한글 NNP, 그리고 변환문이 로마자를 전혀 안 쓸 때의 영어 낱말('도구적 Rationality'→'도구적 합리성',
    # 'adult의 눈'→'어른의 눈')은 번역·음역됐을 수 있어 경고(minor)로만 둔다
    strong = [e for e in missing if not re.fullmatch(r"[가-힣]{2}", e)
              and not (no_latin_out and re.fullmatch(r"[A-Za-z][A-Za-z\-]*", e))]
    weak = [e for e in missing if e not in strong]
    if strong:
        issues.append(_issue("ENTITY_MISSING", f"원문 고유명사가 변환문에 없음: {', '.join(strong)}"))
    if weak:
        issues.append(_issue("ENTITY_WEAK", f"두 글자 고유명사·영어 낱말(오분석·번역 가능)이 변환문에 없음: {', '.join(weak)}"))
    weak_ko = [e for e in weak if re.fullmatch(r"[가-힣]{2}", e)]
    if weak_ko and added and not strong:
        # 두 글자 이름이 사라지고 다른 이름이 생겼으면 오분석이 아니라 바꿔치기로 본다(영어 낱말→한글 음역은 제외)
        issues = [_issue("ENTITY_MISSING", f"고유명사가 바뀐 것으로 보임: {', '.join(weak_ko)} → {', '.join(added)}")]
    if added:
        from_ex = [e for e in added if example_text and e.lower() in example_text.lower()]
        note = f"원문에 없는 고유명사: {', '.join(added)}" + (f" (그중 검색 예시에 있던 것: {', '.join(from_ex)})" if from_ex else "")
        if not from_ex:
            issues.append(_issue("ENTITY_ADDED", note))
        else:
            # Kiwi가 옛 표기·한자어를 고유명사로 잘못 붙이는 일이 있어('別離[별리]', '대상이로소이다'의 '로소',
            # '에어 싸인'=에워싸인) 확실한 이름(세 글자 이상, 어절 첫머리, 한자 읽기 괄호가 아님)일 때만 베낌(FAIL)
            sure = [e for e in from_ex if _sure_name(e, out)]
            issues.append(_issue("EXAMPLE_COPY" if sure else "EXAMPLE_ENTITY", note))
    return issues


def _sure_name(e: str, text: str) -> bool:
    """예시에서 온 고유명사가 오분석이 아니라 확실한 이름인가."""
    if len(e) < 3:
        return False
    for m in re.finditer(re.escape(e), text):
        before = text[m.start() - 1] if m.start() else " "
        if before in "[(（［":  # 한자 뒤 읽기 괄호 '別離[별리]'
            continue
        if before.isspace() or not ("가" <= before <= "힣"):  # 어절 첫머리
            return True
    return False


def _ending(toks):
    """문장 끝 어미(형태, 품사, 위치)와 물음표 여부."""
    q = any(f in ("?", "？") for f, t in toks[-3:] if t.startswith("SF"))
    for j in range(len(toks) - 1, -1, -1):
        f, t = toks[j]
        if not t.startswith(("SF", "SS", "SE", "SP", "SW", "SO")):
            return f, t, j, q
    return "", "", -1, q


def _neg_raw(toks, i) -> bool:
    """부정 표지 자체인가(문맥 보정 전).
    - '변함없다·끊임없다·틀림없다'처럼 명사형(ETN) 뒤 '없'은 굳은 말이라 뺀다.
    - '가지 아니하였다'는 Kiwi가 아니/IC + 하/VX로 나누기도 해서 그 짝도 부정으로 본다.
    - '말다'는 '-지 말다'(금지)만 부정. '전락하고 만다'(-고 말다 = 끝내 ~하다)는 아니다."""
    x = toks[i]
    if x == ("아니", "IC"):
        return i + 1 < len(toks) and toks[i + 1] == ("하", "VX")
    if x == ("말", "VX"):
        return i >= 1 and toks[i - 1] == ("지", "EC")
    if x[0] == "모르" and i >= 1 and toks[i - 1] == ("줄", "NNB"):
        return True  # '~할 줄 모르다' = '~하지 못하다'
    if x not in NEG_MORPHS:
        return False
    if x[0] in ("없", "없이") and i:
        k = i - 1
        while k > 0 and toks[k][1] in ("JKS", "JX", "JKO"):
            k -= 1  # '수 없다'와 '수가 없다'·'바가 없다'를 같게 본다(조사 건너뜀)
        prev = toks[k]
        if prev[1] == "ETN" or (prev[1] == "NNB" and prev[0] in ("수", "바", "나위", "도리", "것")) or (prev[1].startswith("NN") and prev[0] in _LEX_NEG_NOUN):
            return False  # 변함없다·보잘것없다·철없는·쉴 새 없이·할 수(가) 없다·다를 바 없다: 낱말 속 '없'
            # ('한 적 없다'·'~는 법이 없다'·'~할 리 없다'는 진짜 부정이라 남김)
    return True


def _rhetorical(toks) -> str:
    """수사 의문인가. 'neg' = 부정 수사 의문('~아니겠는가', '~되지 않으랴' → 강한 긍정),
    'pos' = 긍정 수사 의문('어찌 ~ 있으랴' → 부정), '' = 아님."""
    f, t, j, q = _ending(toks)
    is_q = q or (t in ("EF", "EC") and f in _RHET_EF) or (f == "소" and j >= 1 and toks[j - 1][0] == "겠") \
        or (any(x == ("어찌", "MAG") for x in toks) and f in ("오", "소", "요", "리", "리라", "고", "ᆯ꼬"))
    # '어찌 ~ 그칠 것이오'는 수사 의문, '어찌 변화하든 ~하다'(양보)는 아님
    tail_mal = any(toks[k][0] == "말" and toks[k][1].startswith("NN") and k >= 1 and toks[k - 1][0] in ("단", "ᆫ단", "란")
                   and k + 1 < len(toks) and toks[k + 1][1] == "VCP" for k in range(max(0, j - 4), j))
    is_q = is_q or (tail_mal and f in ("ᆸ니까", "ᆫ가", "ᆫ가요", "냐", "니", "가"))  # '~할 수 있단 말입니까'
    if not is_q:
        return ""
    if any(_neg_raw(toks, i) for i in range(max(0, j - 6), len(toks))):
        return "neg"  # 질문 끝 가까이의 부정만('~뿐만 아니라 ~ 어찌 모를 수 있으랴'는 긍정 수사 의문)
    if f in _CLASSIC_RHET or tail_mal or any(x[0] in ("어찌", "어떻게", "누가", "어디") for x in toks):
        return "pos"
    return ""


def _neg_idx(toks) -> list[int]:
    """실제 부정으로 셀 위치. 부정 표지라도 뜻이 긍정이거나 대조인 것은 뺀다:
    - 이중부정 '~지 않을 수 없다', '아닐 수 없다', '~지 않으면 안 된다' → 강한 긍정
    - 대조 'A가 아니라 B', '뿐만 아니라', '~이 아니요', '~에 그치지 않고' → 'A를 넘어 B'와 같은 뜻
    - 부정 수사 의문 '~아니겠는가', '~되지 않으랴' → 강한 긍정"""
    idx = [i for i in range(len(toks)) if _neg_raw(toks, i)]
    drop = set()
    if idx and _rhetorical(toks) == "neg" and _ending(toks)[2] - idx[-1] <= 6:
        drop.add(idx[-1])  # 문장 끝 질문에 걸린 부정만 뺀다('맡기지 않겠다는 선언이 아닌가'의 '않'은 남김,
        # '상처받지 않기 위해 ~ 어찌하리오'처럼 끝에서 먼 부정도 남김)
    merely = [k for k, x in enumerate(toks) if x[0] in _MERELY]
    for i in idx:
        nxt = next((t for t in toks[i + 1:i + 4] if t[1] in ("EF", "EC")), None)
        if nxt and nxt[0] in _CLASSIC_RHET:
            drop.add(i)  # 문장 가운데 수사 의문: '갇혀 살아가지 않으랴마는', '~하는 일이 아니더랴'
        if any(0 < i - k <= 20 for k in merely) and (toks[i][0] in ("아니", "안", "않", "아니하") or (toks[i][0] == "없" and toks[i - 1][0] == "리")
                 or (toks[i][0] == "없" and i >= 2 and toks[i - 1][1].startswith("J") and toks[i - 2][0] == "리")):
            drop.add(i)  # '단순히 X가 아니다'·'단순한 X를 의미하지 않는다' = X를 넘어선다는 대조
            # (뒤따르는 20토큰 안의 '아니·않'만. '그저 ~ 이해하지 못하고'의 '못'은 남긴다)
        if toks[i] == ("아니", "VCN") and i + 1 < len(toks) and toks[i + 1][0] in ("면", "으면") \
                and (i == 0 or toks[i - 1][1] in ("SP", "EC")):
            drop.add(i)  # '순종이요, 아니면 무감각이다'의 '아니면' = '또는'
        if toks[i][0] == "없" and any(toks[k][0] in _EMPH_NOUN and (toks[k][1].startswith("NN") or (
                k + 1 < len(toks) and toks[k + 1][1] == "ETN")) for k in range(max(0, i - 3), i)):
            drop.add(i)  # 두말할 나위가 없다, 틀림이 없다, 다름이 없다, 의심의 여지가 없다 (명사 꼴 + 없다만, 강조)
        if any(toks[k][0] in _EMPH_VERB for k in range(max(0, i - 4), i)) and toks[i][0] in ("없", "못하", "않"):
            drop.add(i)  # 피할 수 없는, 부정하지 못하리라, 부인할 수 없다 (강조)
        if toks[i] == ("말", "VX") and i + 1 < len(toks) and toks[i + 1] == ("고", "EC"):
            drop.add(i)  # ~지 말고 B = ~하기보다 B (대조)
        if toks[i][0] in ("않", "아니하") and i >= 2 and toks[i - 1] == ("지", "EC"):
            if toks[i - 2][0] == "지나":
                drop.add(i)  # ~에 지나지 않다 = ~일 뿐이다 (부정이 아님)
            elif toks[i - 2][0] in _LITOTES:
                drop.add(i)  # 결코 가볍지 않은 = 중요한 (완곡 긍정)
        if toks[i] == ("아니", "VCN") and i >= 2 and toks[i - 1] == ("ᆷ", "ETN") and toks[i - 2][0] == "다르":
            drop.add(i)  # ~에 다름아니다 = 바로 ~이다
        for j in range(i + 1, min(len(toks) - 1, i + 5)):  # 이중부정: 부정 뒤 '수' + '없'
            if toks[j] == ("수", "NNB") and toks[j + 1][0] == "없":
                drop |= {i, j + 1}
                break
        if toks[i][0] in ("않", "아니하", "아니") and i + 2 < len(toks) and toks[i + 1][1] == "EC" \
                and toks[i + 1][0] in ("으면", "면") and toks[i + 2] == ("안", "MAG"):  # ~지 않으면 안 되다
            drop |= {i, i + 2}
        if toks[i] == ("아니", "VCN") and i + 1 < len(toks) and (
                (toks[i + 1][1] == "EC" and toks[i + 1][0] in ("라", "고", "요"))
                or (toks[i + 1] == ("ᆫ", "ETM") and i + 2 < len(toks) and toks[i + 2][1] != "NNB")):
            drop.add(i)  # A가 아니라 B / A가 아닌 B
        if toks[i][0] in ("않", "아니하") and i >= 2 and toks[i - 1] == ("지", "EC") and toks[i - 2][0] in ("그치", "머무르", "머물"):
            # ~에 그치지 않고/않는다/아니할 터이다 = ~를 넘어선다(대조). '그쳤다↔그치지 않았다'의 진짜 뒤집힘은 놓칠 수 있음
            drop.add(i)
    return [i for i in idx if i not in drop]


def _neg_count(toks) -> int:
    return 1 if _rhetorical(toks) == "pos" else len(_neg_idx(toks))


def _has_neg(toks) -> bool:
    return _neg_count(toks) > 0


def _neg_predicates(toks) -> list[str]:
    """부정이 걸린 서술어(내용어)들. '발생하지 않았다'→발생, '효과가 없다'→효과, '못 간다'→가."""
    out = []
    for i in _neg_idx(toks):
        rng = range(i + 1, min(len(toks), i + 4)) if toks[i][1] == "MAG" else range(i - 1, max(-1, i - 8), -1)
        for j in rng:
            f, t = toks[j]
            if t.startswith(CONTENT_TAGS) and f not in STOP_STEMS:
                out.append(f)
                break
    return out


def _positional_map(source: str, output: str, a: dict, b: dict) -> dict[int, int]:
    """문단 수가 같고 문단마다 문장 수가 같으면 '입력 k번째 문장 ↔ 출력 k번째 문장' 대응표(전역 문장 번호).
    문단 단위로 생성하면 대부분 이렇게 1:1로 나온다. 단어를 많이 바꿔 겹침으로는 짝을 못 찾을 때 이 짝을 쓴다."""
    sp = [p for p in re.split(r"\n\s*\n", source.strip()) if p.strip()]
    op = [p for p in re.split(r"\n\s*\n", output.strip()) if p.strip()]
    if len(sp) != len(op):
        return {}
    ss = [_analyze(p)["sents"] for p in sp]
    os_ = [_analyze(p)["sents"] for p in op]
    if any(len(x) != len(y) for x, y in zip(ss, os_)):
        return {}
    flat_s = [x for p in ss for x in p]
    flat_o = [x for p in os_ for x in p]
    if flat_s != a["sents"] or flat_o != b["sents"]:  # 문장 나누기가 글 전체와 문단별로 다르면 쓰지 않음
        return {}
    return {i: i for i in range(len(flat_s))}


def _para_positions(text: str, an: dict):
    """문장마다 (문단 번호, 문단 안 상대 위치 0~1). 문단별 문장 나누기가 글 전체와 다르면 None."""
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    out = []
    for pi, p in enumerate(paras):
        ss = _analyze(p)["sents"]
        out += [(pi, (k + 0.5) / len(ss)) for k in range(len(ss))]
    return (out, len(paras)) if [x for p in paras for x in _analyze(p)["sents"]] == an["sents"] else (None, len(paras))


def _bigram_sim(x: str, y: str) -> float:
    """한글 글자 2-gram 겹침(작은 쪽 기준). 바꿔 쓴 문장은 어근 일부를 공유한다(반군성/반란성, 영역/구역)."""
    gx = {x[i:i + 2] for i in range(len(x) - 1) if re.fullmatch(r"[가-힣]{2}", x[i:i + 2])}
    gy = {y[i:i + 2] for i in range(len(y) - 1) if re.fullmatch(r"[가-힣]{2}", y[i:i + 2])}
    return len(gx & gy) / min(len(gx), len(gy)) if gx and gy else 0.0


def _paraphrase_pairs(source: str, output: str, a: dict, b: dict, omit: set[int], add: set[int]) -> dict[int, int]:
    """'누락' 입력 문장과 '추가' 출력 문장 중 같은 자리에서 바꿔 쓴 짝. 문단 수가 같을 때만, 같은 문단 안에서
    문단 속 상대 위치가 가까운 것끼리(문장을 합치거나 나눠 문장 수가 달라도 됨) 하나씩 짝짓는다."""
    pa, na = _para_positions(source, a)
    pb, nb = _para_positions(output, b)
    if pa is None or pb is None or na != nb:
        return {}
    cand = []
    for k in omit:
        for j in add:
            if pa[k][0] == pb[j][0]:
                d = abs(pa[k][1] - pb[j][1])
                n = max(sum(1 for x in pa if x[0] == pa[k][0]), sum(1 for x in pb if x[0] == pb[j][0]))
                if d <= 1 / n + 0.15 and (_coverage(a["sent_lem"][k], b["sent_lem"][j]) >= PARA_MIN_SIM
                                          or _bigram_sim(a["sents"][k], b["sents"][j]) >= PARA_MIN_SIM):
                    cand.append((d, k, j))  # 남의 글 문장으로 바꿔치기한 것은 겹침이 거의 0이라 짝이 안 됨
    pairs, used = {}, set()
    for d, k, j in sorted(cand):
        if k not in pairs and j not in used:
            pairs[k] = j
            used.add(j)
    return pairs


def _aligned(lem: set[str], other: dict):
    """lem과 가장 닮은 상대 문장(내용어 덮는 비율)과 그 비율."""
    best, idx = 0.0, -1
    for i, ol in enumerate(other["sent_lem"]):
        c = _coverage(lem, ol)
        if c > best:
            best, idx = c, i
    return idx, best


# 부정+반대말 = 같은 뜻: '같다'↔'다르지 않다', '넘어'↔'그치지 않고'. 대응 문장에 반대말이 부정 없이 있으면 뒤집힘 아님.
_ANTONYM = {"다르": "같", "같": "다르", "그치": "넘", "머무르": "넘", "머물": "넘", "어렵": "쉽", "쉽": "어렵",
            "적": "많", "많": "적", "작": "크", "크": "작", "잃": "얻", "얻": "잃"}


def _flips(a: dict, b: dict, preds_of, count, pos: dict[int, int] | None = None) -> list[tuple[str, str]]:
    """a 문장에서 표지(부정·추측)가 걸린 서술어 p마다, b의 대응 문장에서
    (p에 같은 표지가 걸려 있거나) (표지 개수가 a 문장 이상이면) 정상, 아니면 뒤집힘.
    대응 문장 = a 문장과 내용어가 ALIGN_MIN 이상 겹치는 b 문장(문장이 나뉜 경우 앞뒤 문장 포함).
    대응 문장이 없으면 판단하지 않는다(서술어만으로 찾으면 같은 낱말이 든 엉뚱한 문장과 비교하게 됨). 개수 조건 덕에 '적응하는 방식이 아니다'↔'적응하지 않는다'처럼
    표지가 다른 낱말에 붙은 바꿔 말하기는 통과하고, 원래 부정이 있던 문장에 부정이 하나 더 생긴 것은 잡는다."""
    flips = []
    for k, (s, st, sl) in enumerate(zip(a["sents"], a["sent_toks"], a["sent_lem"])):
        need = count(st)
        preds = preds_of(st)
        if not preds:
            continue
        # 대응 문장이 분명하면(내용어 ALIGN_MIN 이상 겹침) 그 문장과 비교한다. 서술어만으로 찾으면 같은 낱말이 든
        # 엉뚱한 문장과 비교하게 된다('받아들이지 않으며'↔'인정하지 않으며'). 문장이 둘로 나뉜 경우만 앞뒤 문장을 더 본다.
        i, cov = _aligned(sl, b)
        window = None
        if i >= 0 and cov >= ALIGN_MIN:
            window = [b["sent_toks"][i]]
            if len(b["sent_lem"][i]) < 0.6 * len(sl):
                window = b["sent_toks"][max(0, i - 1):i + 2]
        merged = False
        if window is None and i >= 0:
            # 두 문장을 하나로 합친 경우: 이웃 문장과 합쳐야 ALIGN_MIN을 넘으면 두 문장을 한 덩어리로 비교
            best = max(((_coverage(sl, set(b["sent_lem"][i]) | set(b["sent_lem"][j])), j)
                        for j in (i - 1, i + 1) if 0 <= j < len(b["sents"])), default=(0.0, -1))
            if best[0] >= ALIGN_MIN and best[0] - cov >= 0.15:
                window = [b["sent_toks"][j] for j in sorted((i, best[1]))]
                merged = True
        if window is None and pos and k in pos:
            window = [b["sent_toks"][pos[k]]]  # 많이 바꿔 썼어도 문단·문장 위치가 1:1이면 같은 자리 문장과 비교
        if window is None:
            continue  # 대응 문장이 없으면 뒤집힘은 판단하지 않는다 → 내용 누락·추가 검사가 맡는다
        for p in preds:
            hits = window
            ant = _ANTONYM.get(p)
            if ant and any(t[0] == ant for bt in window for t in bt) and count(st) > 0 and preds_of is _neg_predicates:
                continue  # '같다'→'다르지 않다' 같은 반대말 바꿔 말하기
            if i >= 0 and len(window) == 1 and not _present(p, set(b["sent_lem"][i])):
                # 서술어가 대응 문장에 없고 바로 앞뒤 문장에 있으면 두 문장이 하나로 합쳐진 것 → 그 문장과 비교
                near = [j for j in (i - 1, i + 1) if 0 <= j < len(b["sents"]) and _present(p, set(b["sent_lem"][j]))]
                if near:
                    hits = [b["sent_toks"][near[0]]]
            if merged:  # 합친 두 문장의 표지를 모두 더해도 모자라면 뒤집힘
                if not any(_present(p, set(preds_of(bt))) for bt in hits) and sum(count(bt) for bt in hits) < need:
                    flips.append((p, s))
                    break
                continue
            if hits and not any(_present(p, set(preds_of(bt))) or count(bt) >= need for bt in hits) \
                    and sum(count(bt) for bt in hits) < need * max(1, len(hits)):
                flips.append((p, s))
                break  # 한 문장에서 하나만
    return flips


def _neg_flips(a: dict, b: dict, pos=None) -> list[tuple[str, str]]:
    return _flips(a, b, _neg_predicates, _neg_count, pos)


def _hedge_positions(toks) -> list[int]:
    """추측 표현이 시작하는 토큰 위치. '알 수 있다'(결론 제시)는 추측이 아니라 뺀다.
    '~ㄹ 수 없다'는 불가능(추측 아님)이라 빼고, 긍정 수사 의문('어찌 ~ 있으랴')의 '수 있'과
    부정된 '수 있'('정의될 수 있는 것이 아니라')도 셈하지 않는다. 비유 '~하듯'(연결어미)은 추측이 아니다.
    '~한 모양이다'(= ~인 것 같다)는 추측으로 센다. '~인 셈이다'는 '결국 ~인 것이다'라는 정리 말투라 세지 않는다."""
    pos = []
    rhet = _rhetorical(toks)
    rhet_pos = rhet == "pos"  # '어찌 ~ 있으랴'의 '수 있'만 뺀다(부정 수사 의문 '~아니겠소'의 추측은 그대로)
    raw_neg = [k for k in range(len(toks)) if _neg_raw(toks, k)]
    if rhet == "neg" and raw_neg:
        raw_neg = raw_neg[:-1]  # '~할 수 있지 않으리요'의 끝 부정은 수사 의문이라 '수 있'을 부정하지 않음
    for i, (f, t) in enumerate(toks):
        nxt = toks[i + 1][0] if i + 1 < len(toks) else ""
        if nxt in ("도", "는", "가") and i + 2 < len(toks):  # '할 수도 있다', '할 수는 있다'
            nxt = toks[i + 2][0]
        prv = toks[i - 2][0] if i >= 2 else ""
        if (f == "모양" and t.startswith("NN") and i and toks[i - 1][1] == "ETM"
                and i + 1 < len(toks) and toks[i + 1][1] == "VCP"):
            pos.append(i)
            continue
        negated = any(i < k < i + 7 for k in raw_neg)  # '~ㄹ 수 있는 것이 아니라'
        past = any(toks[k][1] == "EP" and toks[k][0] in ("었", "았", "였") for k in range(i + 2, min(len(toks), i + 4)))
        negated = negated or past  # '들을 수 있었다면'·'할 수 있었다'는 과거의 능력이지 추측이 아님
        j_has = i + (3 if i + 1 < len(toks) and toks[i + 1][0] in ("도", "는", "가") else 2)  # '있' 다음 토큰 위치
        attributive = j_has + 1 < len(toks) and toks[j_has][1] == "ETM" and toks[j_has + 1][1] != "NNB" \
            and toks[j_has][0] != "단"  # '뒷받침할 수 있는 (참된) 환경' = 능력·여건이지 글쓴이의 추측이 아님
        formula = i >= 3 and toks[i - 2][0] == "하" and toks[i - 3][1] == "EC" and toks[i - 3][0] in ("라", "고")
        # '~라 할 수 있다'(='~라 하겠다')는 결론을 맺는 굳은 말투라 추측으로 세지 않는다('~다고 볼 수 있다'는 셈)
        literal_see = f == "보이" and any(toks[k][0] == "눈" for k in range(max(0, i - 3), i))  # 눈에 보이다
        aux_show = f == "보이" and (
            (i >= 2 and toks[i - 1][1] == "EC" and toks[i - 2][1] == "VV")  # 드러내 보이다(동사+어 보이다)
            or (i >= 1 and toks[i - 1][1] == "JKO")  # '경향을 보이다'·'길을 보여 주다' = 드러내다
            or (i + 2 < len(toks) and toks[i + 1][1] == "EC" and toks[i + 2] == ("주", "VX")))  # 보여 주다
        ability = j_has < len(toks) and (toks[j_has][0] in ("어야", "도록", "게") and toks[j_has][1] == "EC"
                                         or toks[j_has] == ("기", "ETN"))  # 할 수 있어야/있도록/있기 위해 = 능력
        if (f == "수" and t == "NNB" and nxt == "있" and prv != "알" and not rhet_pos and not negated and not attributive
                and not formula and not ability) \
                or (f == "것" and t == "NNB" and nxt == "같" and not (
                    i + 2 < len(toks) and toks[i + 2][1] == "EC" and toks[i + 2][0] in ("으면", "면"))) \
                or (f in ("듯", "듯하") and t != "EC") or (f == "보이" and t.startswith("VV") and not literal_see
                                                         and not aux_show) or f in HEDGE_WORDS \
                or (t == "EC" and f.endswith("지도")):
            pos.append(i)
    return pos


def _hedge_count(toks) -> int:
    return len(_hedge_positions(toks))


def _hedged_predicates(toks) -> list[str]:
    """추측이 걸린 서술어: '변화시킬 수 있다'→변화, '늦어질 것 같다'→늦, '어쩌면 오른다'→오르."""
    out = []
    for i in _hedge_positions(toks):
        f = toks[i][0]
        rng = range(i + 1, min(len(toks), i + 4)) if f in HEDGE_WORDS and toks[i][1] == "MAG" else range(i - 1, max(-1, i - 8), -1)
        for j in rng:
            g, t = toks[j]
            if t.startswith(CONTENT_TAGS) and g not in STOP_STEMS:
                out.append(g)
                break
    return out


def _hedge_flips(a: dict, b: dict, pos=None) -> list[tuple[str, str]]:
    """a에서 추측이 걸린 서술어가 b에서 추측 없이 나오면 단정으로 바뀐 것."""
    return _flips(a, b, _hedged_predicates, lambda st: len(_hedge_positions(st)), pos)


def _sent_with(sents, sent_toks, pred) -> str:
    return next((s for s, st in zip(sents, sent_toks) if pred(st)), "")


def check_polarity(a: dict, b: dict, pos: dict[int, int] | None = None) -> list[dict]:
    """부정(서술어 단위)·추측(개수)·범위어(개수). 근거로 해당 표현이 든 문장을 붙인다."""
    issues = []
    pos = pos or {}
    inv = {v: k for k, v in pos.items()}
    for (p, sent), what, side in [(x, "원문에서 부정→변환문에서 긍정", "source") for x in _neg_flips(a, b, pos)] + \
                                 [(x, "원문에서 긍정→변환문에서 부정", "output") for x in _neg_flips(b, a, inv)]:
        issues.append(_issue("NEGATION_FLIP", f"'{p}': {what}", sent, side))
    flips = [(x, "원문 추측→변환문 단정", "source") for x in _hedge_flips(a, b, pos)] + \
            [(x, "원문 단정→변환문 추측", "output") for x in _hedge_flips(b, a, inv)]
    for (p, sent), what, side in flips:
        issues.append(_issue("MODALITY_SHIFT", f"'{p}': {what}", sent, side))
    hs, ho = _hedge_count(a["toks"]), _hedge_count(b["toks"])
    if hs != ho and not flips:
        issues.append(_issue("MODALITY_MINOR", f"추측 표현(수 있다·것 같다·보인다 등) {hs}→{ho}개 (서술어 단위 변화는 없음)"))
    def group(x, strong):
        if not strong:
            return x[0]
        return _SCOPE_GROUP.get(x[0], x[0])

    def where(doc, other, words):
        """화면 강조·검토용: 그 범위어가 든 문장 중, 대응 문장(상대 글)에는 그 범위어가 없는 문장을 먼저 고른다."""
        first = ""
        for sent, st, sl in zip(doc["sents"], doc["sent_toks"], doc["sent_lem"]):
            if any(t[0] in words for t in st):
                first = first or sent
                i, cov = _aligned(sl, other)
                if i < 0 or cov < ALIGN_MIN or not any(t[0] in words for t in other["sent_toks"][i]):
                    return sent
        return first

    for table, strong in ((STRONG_SCOPE, True), (WEAK_SCOPE, False)):
        def words_in(doc):  # '항상성'(恒常性)의 '항상'처럼 뒤에 붙은 말이 있으면 범위어가 아님
            tk = doc["toks"]
            return [x for k, x in enumerate(tk) if x in table and not (k + 1 < len(tk) and tk[k + 1][0] == "성")]
        cs = Counter(group(x, strong) for x in words_in(a))
        co = Counter(group(x, strong) for x in words_in(b))
        gone, new = cs - co, co - cs
        for keys, side, doc, other, verb in ((gone, "source", a, b, "빠짐"), (new, "output", b, a, "생김")):
            for g in keys:
                words = {f for f, _ in table if group((f, ""), strong) == g}
                kind = "SCOPE_MINOR" if (not strong or g in _SCOPE_MINOR_GROUPS) else "SCOPE_CHANGE"
                forms = sorted({t[0] for t in doc["toks"] if t[0] in words})
                issues.append(_issue(kind, f"범위어 {verb}: {', '.join(forms) or g}", where(doc, other, words), side))
    return issues


def _eojeol_ngrams(text: str, n: int) -> set[tuple]:
    w = [re.sub(r"[^\w가-힣]", "", x) for x in text.split()]
    w = [x for x in w if x]
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _copied_span(out_text: str, copied: set[tuple]) -> str:
    """베낀 5어절 묶음들을 이어 변환문의 실제 구절(가장 긴 것)로 돌려준다. 재번역 지시·화면 강조에 그대로 쓰도록."""
    raw = out_text.split()
    norm = [re.sub(r"[^\w가-힣]", "", x) for x in raw]
    hit = [False] * len(raw)
    for i in range(len(raw) - COPY_NGRAM + 1):
        if tuple(norm[i:i + COPY_NGRAM]) in copied:
            for j in range(i, i + COPY_NGRAM):
                hit[j] = True
    best, cur = [], []
    for w, h in zip(raw, hit):
        cur = cur + [w] if h else []
        if len(cur) > len(best):
            best = cur
    return " ".join(best)


def check_content(a: dict, b: dict, example_text: str = "", pos: dict[int, int] | None = None,
                  source: str = "", output: str = "") -> tuple[list[dict], dict]:
    """문장 단위 누락·추가 + 글 전체 재현율 + 예시 베낌.
    pos(문단·문장 위치 1:1 대응)가 있을 때 같은 자리의 입력 문장이 '누락', 출력 문장이 '추가'로 함께 걸리면
    빠진 게 아니라 한 문장을 크게 바꿔 쓴 것이므로 '많이 바꿔 씀'(minor) 하나로 합친다.
    (new18h_para 표본 12쌍을 읽어 보니 8쌍은 뜻이 같은 바꿔 말하기, 4쌍은 정도·정밀도 차이, 명백한 오류 0쌍)"""
    issues = []
    la, lb = _lemmas(a["toks"]), _lemmas(b["toks"])
    pa, pb = set(la), set(lb)
    def local(lem: set[str], other: dict) -> float:
        # 상대 글의 이웃한 두 문장(i, i+1) 중 가장 잘 덮는 곳. 문장 합치기·나누기는 허용, 멀리 흩어진 낱말은 인정 안 함
        ls = other["sent_lem"] or [set()]
        return max(_coverage(lem, ls[i] | (ls[i + 1] if i + 1 < len(ls) else set())) for i in range(len(ls)))

    pos = pos or {}
    ex_pool = set(_lemmas(_analyze(example_text)["toks"])) if example_text else set()
    omit = {k for k, lem in enumerate(a["sent_lem"]) if len(lem) >= SENT_MIN_LEMMAS and local(lem, b) < SENT_COVERAGE_MIN}
    add = {k for k, lem in enumerate(b["sent_lem"]) if len(lem) >= SENT_MIN_LEMMAS and local(lem, a) < SENT_COVERAGE_MIN}
    copied_sent = set()
    def rewrites_source(k: int) -> bool:
        # 원문의 어떤 문장을 바꿔 쓴 것인가(내용어·두 글자 겹침 REWRITE_MIN 이상). 고유어로 많이 바꾸는 문체
        # (이광수: 경험하다→겪다, 타인→남)는 새 낱말이 예시에 있는 게 당연하므로 베낌으로 보지 않는다.
        lem = b["sent_lem"][k]
        for j in range(len(a["sents"])):
            shared = sum(1 for x in lem if _present(x, a["sent_lem"][j]))
            if (_coverage(lem, a["sent_lem"][j]) >= REWRITE_MIN and shared >= 3) \
                    or _bigram_sim(a["sents"][j], b["sents"][k]) >= REWRITE_MIN:
                return True  # 짧은 예시 문장이 낱말 두어 개 겹친다고 바꿔 쓴 것으로 보지 않게 '겹친 낱말 3개 이상'
        return False

    for k in add:  # 예시 베낌이 먼저: 원문 어느 문장도 바꿔 쓴 게 아닌데 내용이 예시에서 왔으면 베낌
        new = sorted(l for l in b["sent_lem"][k] if not _present(l, pa))
        if ex_pool and new and _coverage(new, ex_pool) >= EXAMPLE_CONTENT_MIN and not rewrites_source(k):
            copied_sent.add(k)
            issues.append(_issue("EXAMPLE_COPY", f"원문에 없는 문장인데 내용어가 검색 예시에서 옴 ({', '.join(new[:8])})",
                                 b["sents"][k], "output"))
    pairs = {k: pos[k] for k in omit if k in pos and pos[k] in add and pos[k] not in copied_sent}
    if source and output:  # 문장 수가 달라도 같은 문단의 비슷한 자리면 짝(문장 합치기·나누기)
        rest = _paraphrase_pairs(source, output, a, b, omit - set(pairs), add - set(pairs.values()) - copied_sent)
        pairs.update(rest)
    para = set(pairs)
    for k in sorted(para):
        issues.append(_issue("PARAPHRASE", f"같은 자리 문장을 단어를 많이 바꿔 다시 씀 — 입력: “{a['sents'][k][:80]}”",
                             b["sents"][pairs[k]], "output"))
    paired_out = set(pairs.values())
    for k in sorted(omit - para):
        lem = a["sent_lem"][k]
        miss = sorted(l for l in lem if not _present(l, pb))
        note = f"빠진 말: {', '.join(miss[:8])}" if miss else "낱말은 다른 곳에 흩어져 있으나 이 문장에 대응하는 곳이 없음"
        issues.append(_issue("OMISSION", f"원문 문장에 대응하는 변환문 문장이 없음 ({note})", a["sents"][k], "source"))
    for k in sorted(add - paired_out - copied_sent):
        new = sorted(l for l in b["sent_lem"][k] if not _present(l, pa))
        issues.append(_issue("ADDITION", f"원문에 대응하는 곳이 없는 문장 (새 말: {', '.join(new[:8]) or '없음'})",
                             b["sents"][k], "output"))
    recall, precision = _multiset_recall(la, lb), _multiset_recall(lb, la)
    if recall < RECALL_MIN:
        issues.append(_issue("LOW_RECALL", f"원문 내용어 중 {recall:.0%}만 남음 (하한 {RECALL_MIN:.0%})"))
    if example_text:
        copied = (_eojeol_ngrams(" ".join(b["sents"]), COPY_NGRAM) & _eojeol_ngrams(example_text, COPY_NGRAM)) \
            - _eojeol_ngrams(" ".join(a["sents"]), COPY_NGRAM)
        if copied and not any(i["type"] == "EXAMPLE_COPY" for i in issues):
            span = _copied_span(" ".join(b["sents"]), copied)
            issues.append(_issue("EXAMPLE_COPY", f"검색 예시와 {COPY_NGRAM}어절 이상 같은 구절 {len(copied)}곳 (원문에는 없음)", span, "output"))
    return issues, {"content_recall": round(recall, 3), "content_precision": round(precision, 3)}


def check_meta(out: str, src: str = "") -> list[dict]:
    if not out.strip():
        return [_issue("EMPTY_OUTPUT", "변환문이 비어 있음")]
    m = META_RX.search(out)
    if m and m.group(0).strip() not in src:  # 입력에도 있던 표현('결과:' 등)은 머리말이 아님
        return [_issue("META_TEXT", "변환문에 설명·머리말·원문 표시가 섞임", m.group(0).strip(), "output")]
    return []


def style_change(src: str, out: str) -> dict:
    """문체가 얼마나 바뀌었나(의미 판정과 무관한 참고값). 기능 형태소(조사·어미·접속부사 등) 기준."""
    fa = Counter(x for x in _analyze(src)["toks"] if x[1].startswith(("J", "E", "XS", "VX", "MAJ", "NNB")))
    fb = Counter(x for x in _analyze(out)["toks"] if x[1].startswith(("J", "E", "XS", "VX", "MAJ", "NNB")))
    change = 1 - sum((fa & fb).values()) / max(1, max(sum(fa.values()), sum(fb.values())))
    ratio = SequenceMatcher(None, src.split(), out.split(), autojunk=False).ratio()
    return {"style_change": round(change, 3), "eojeol_match": round(ratio, 3), "unchanged": ratio >= UNCHANGED_RATIO}


# ---------------------------------------------------------------- 진입점
def _example_text(examples) -> str:
    """examples: 문자열 목록 또는 {"text"|"styled"|"ai": ...} 목록. 짝 예시는 양쪽 모두 베낌 검사에 쓴다."""
    if not examples:
        return ""
    parts = []
    for e in examples:
        if isinstance(e, str):
            parts.append(e)
        else:
            parts += [str(e[k]) for k in ("text", "styled", "ai") if e.get(k)]
    return "\n".join(parts)


def decide(issues: list[dict]) -> str:
    worst = max((RANK[i["severity"]] for i in issues), default=0)
    return "FAIL" if worst >= 3 else "REVIEW" if worst == 2 else "PASS"


def verify(source: str, output: str, examples=None, check_format: bool = True) -> dict:
    """source(변환 입력 = AI가 쓴 일반 글) 대비 output(문체 변환 결과)의 의미 유지 검사. 한 번만 실행."""
    issues = check_meta(output, source)
    if issues and issues[0]["type"] == "EMPTY_OUTPUT":
        return {"verdict": "FAIL", "issues": issues, "stats": {}, "format": None}
    ratio = len(output.strip()) / max(1, len(source.strip()))
    if len(source.strip()) >= TRUNC_MIN_SOURCE and ratio < TRUNC_RATIO:
        # 실제 변환 240개의 길이 비는 0.99~1.75. 0.7 미만은 잘린 출력(재번역이 마지막 문단만 돌려준 경우 등)
        issues.append(_issue("TRUNCATED", f"변환문 길이가 원문의 {ratio:.0%}뿐 (하한 {TRUNC_RATIO:.0%})"))
    ex = _example_text(examples)
    a, b = _analyze(source), _analyze(output)
    issues += check_numbers(source, output)
    issues += check_entities(source, output, a["toks"], b["toks"], ex)
    pos = _positional_map(source, output, a, b)
    issues += check_polarity(a, b, pos)
    c_issues, c_stats = check_content(a, b, ex, pos, source, output)
    issues += c_issues
    fmt = None
    if check_format:
        from .ko_format import check as fcheck
        fmt = fcheck(source, output)
        if not fmt["pass"]:
            issues.append(_issue("FORMAT", "; ".join(fmt["reasons"])))
    stats = {**c_stats, **style_change(source, output), "sentences": f"{len(a['sents'])}→{len(b['sents'])}"}
    return {"verdict": decide(issues), "issues": issues, "stats": stats, "format": fmt}


def summarize(result: dict) -> str:
    """사람이 읽을 한 줄 요약."""
    if not result["issues"]:
        return f"{result['verdict']}"
    c = Counter((i["severity"], i["label_ko"]) for i in result["issues"])
    order = sorted(c, key=lambda k: -RANK[k[0]])
    return f"{result['verdict']} | " + " / ".join(f"[{s}] {l}" + (f" ×{c[(s, l)]}" if c[(s, l)] > 1 else "") for s, l in order)


# ---------------------------------------------------------------- 재번역·사람 확인용 (src/ko_pipeline.py가 씀)
# 정책: FAIL(critical)만 기계가 고친다. REVIEW(major)는 오탐이 섞여 있어 자동으로 고치면 멀쩡한 문체를 망칠 수
# 있으므로 고치지 않고, 문제 문장을 표시해 사람에게 넘긴다.
FIX_HINTS = {
    "NUMBER_MISMATCH": "숫자는 원문과 똑같이 쓰세요. 빠진 숫자는 되살리고, 원문에 없는 숫자는 지우세요.",
    "EXAMPLE_COPY": "이 부분은 원문에 없는 내용입니다(참고 예시에서 온 것). 지우거나 원문의 해당 내용으로 바꾸세요.",
    "META_TEXT": "설명·머리말·원문 표시는 지우고 다시 쓴 글만 남기세요.",
    "EMPTY_OUTPUT": "다시 쓴 글 전체를 출력하세요.",
    "TRUNCATED": "글 일부만 출력됐습니다. 원문 전체(모든 문단)를 빠짐없이 다시 쓰세요.",
    "LOW_RECALL": "원문 내용이 많이 빠졌습니다. 원문 전체(모든 문단)를 빠짐없이 다시 쓰세요.",
}

_PREAMBLE_LINE = re.compile(
    r"^\s*((네|물론|알겠습니다|좋습니다)[,.!]?\s*)?(다음은|아래는|요청하신)[^\n]{0,60}(다시 쓴|변환한|바꾼|재작성한|옮긴)[^\n]{0,30}$"
    r"|^\s*[^\n]{0,40}(다시 쓴|변환(된|한)?|바꾼)[^\n]{0,20}[:：]\s*$"
    r"|^\s*#{1,6}\s[^\n]*$")
_CLOSING_LINE = re.compile(r"^\s*[^\n]{0,40}(다시 썼|변환했|바꿨|재작성했)습니다[.!]?\s*$")
_LABEL_PREFIX = re.compile(r"^\s*(\*\*)?(다시 쓴 글|변환(된|한)? ?글|결과|출력)(\*\*)?\s*[:：]\s*")


def retry_issues(result: dict) -> list[dict]:
    """재번역으로 고칠 문제 = critical만."""
    return [i for i in result.get("issues", []) if i["severity"] == "critical"]


def strip_meta(source: str, output: str) -> str:
    """머리말·제목·끝맺음 인사·'다시 쓴 글:' 꼬리표를 규칙으로 지운다(API 호출 없음).
    입력 글에도 있는 줄은 머리말이 아니므로 남긴다. 지운 뒤에도 META_TEXT가 남으면 재번역으로 넘긴다."""
    lines = output.strip().split("\n")
    src = source.strip()
    while lines and (not lines[0].strip() or _PREAMBLE_LINE.match(lines[0])) and (not lines[0].strip() or lines[0].strip() not in src):
        lines.pop(0)
    if lines and not _LABEL_PREFIX.match(src):
        lines[0] = _LABEL_PREFIX.sub("", lines[0])
    while lines and (not lines[-1].strip() or _CLOSING_LINE.match(lines[-1])) and (not lines[-1].strip() or lines[-1].strip() not in src):
        lines.pop()
    return "\n".join(lines).strip()


def fix_instructions(result: dict) -> str:
    """critical 문제를 재번역 모델에 줄 한국어 지시로. 근거 구절이 있으면 그대로 인용한다."""
    out = []
    for i in retry_issues(result):
        line = f"- [{i['label_ko']}] {i['note']}"
        if i.get("evidence"):
            line += f' / 해당 부분: "{i["evidence"]}"'
        hint = FIX_HINTS.get(i["type"], "원문과 뜻이 같게 고치세요.")
        if i["type"] == "EXAMPLE_COPY" and "같은 구절" in i["note"]:
            # 맺음말 같은 구절은 '지우거나 바꾸라'로는 안 지워지는 일이 있어(회억해 보는 것입니다) 낱말을 바꾸라고 분명히 함
            hint = ("따옴표 안 구절은 참고 예시의 문장을 그대로 옮긴 것입니다. 이 구절의 낱말을 그대로 쓰지 말고, "
                    "그 문장은 원문의 해당 문장 뜻대로 다른 말로 끝맺으세요(한자를 덧붙여도 같은 구절입니다).")
        out.append(line + f" → {hint}")
    return "\n".join(out)


def review_points(result: dict) -> list[dict]:
    """사람이 특히 확인할 곳(major 문제). 같은 문장은 하나로 묶는다.
    반환: [{"sentence", "where"("source"|"output"|""), "labels"[...], "notes"[...]}] — 근거 문장이 있는 것부터."""
    pts: dict[tuple, dict] = {}
    for i in result.get("issues", []):
        if i["severity"] != "major":
            continue
        key = (i.get("where", ""), i.get("evidence", "")) if i.get("evidence") else ("", i["type"])
        p = pts.setdefault(key, {"sentence": i.get("evidence", ""), "where": i.get("where", ""), "labels": [], "notes": []})
        if i["label_ko"] not in p["labels"]:
            p["labels"].append(i["label_ko"])
        p["notes"].append(i["note"])
    return sorted(pts.values(), key=lambda p: (not p["sentence"], p["where"] != "output"))


def mark_segments(text: str, sentences: list[str]) -> list[tuple[str, bool]]:
    """text를 [(조각, 표시 여부)]로 나눈다. sentences에 든 문장(그대로 나오는 것)만 표시. 화면 강조용."""
    spans = []
    for s in {s for s in sentences if s and s in text}:
        start = text.find(s)
        while start != -1:
            spans.append((start, start + len(s)))
            start = text.find(s, start + len(s))
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    segs, pos = [], 0
    for a, b in merged:
        if pos < a:
            segs.append((text[pos:a], False))
        segs.append((text[a:b], True))
        pos = b
    if pos < len(text):
        segs.append((text[pos:], False))
    return segs
