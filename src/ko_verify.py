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
    "LOW_RECALL": "major",
    "ENTITY_MISSING": "major",
    "NEGATION_FLIP": "major",
    "MODALITY_SHIFT": "major",
    "SCOPE_CHANGE": "major",
    "MODALITY_MINOR": "minor",
    "SCOPE_MINOR": "minor",
    "NATIVE_NUMBER": "minor",
    "ENTITY_ADDED": "minor",
    "ENTITY_WEAK": "minor",
    "FORMAT": "minor",
}
LABEL_KO = {
    "EMPTY_OUTPUT": "빈 출력", "META_TEXT": "설명·머리말 섞임", "NUMBER_MISMATCH": "숫자 불일치",
    "EXAMPLE_COPY": "검색 예시 베낌", "OMISSION": "내용 누락", "ADDITION": "원문에 없는 내용 추가",
    "LOW_RECALL": "전체 내용 이탈", "ENTITY_MISSING": "고유명사 누락", "ENTITY_ADDED": "고유명사 추가", "ENTITY_WEAK": "두 글자 고유명사 누락",
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
ALIGN_MIN = 0.70           # 부정·추측이 걸린 서술어가 상대 글에 없을 때, 이만큼 닮은 문장(과 앞뒤 문장)끼리 비교

# 내용어: 명사·수사·외국어·숫자·어근·동사·형용사 어간. 너무 흔한 어간은 뺀다.
CONTENT_TAGS = ("NNG", "NNP", "NR", "SL", "SN", "XR", "VV", "VA")
STOP_STEMS = {"하", "되", "있", "없", "같", "이", "그렇", "이렇", "저렇", "어떻", "대하", "위하", "통하",
              "따르", "의하", "않", "보", "가지", "나", "들", "그러", "이러"}
NEG_MORPHS = {("않", "VX"), ("못", "MAG"), ("안", "MAG"), ("없", "VA"), ("없이", "MAG"), ("아니", "VCN"), ("말", "VX"),
              ("못하", "VX")}
HEDGE_WORDS = {"아마", "어쩌면", "아마도", "추정", "짐작", "가능성"}
STRONG_SCOPE = {("모든", "MM"), ("모두", "MAG"), ("항상", "MAG"), ("언제나", "MAG"), ("늘", "MAG"), ("결코", "MAG"),
                ("절대", "MAG"), ("절대로", "MAG"), ("전혀", "MAG"), ("반드시", "MAG"), ("오직", "MAG"),
                ("무조건", "MAG"), ("틀림없이", "MAG")}  # '분명히'는 '분명히 하다'(명확히 하다)가 흔해 뺌
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


def _lemmas(toks) -> list[str]:
    return [f for f, t in toks if t.startswith(CONTENT_TAGS) and f not in STOP_STEMS]


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
            "evidence": evidence[:200], "where": where if evidence else ""}


def _numbers(text: str) -> tuple[Counter, Counter]:
    arabic = Counter()
    for m in _NUM_RX.finditer(text):
        v = float(m.group(1).replace(",", "")) * _KO_UNIT.get(m.group(2) or "", 1)
        arabic[f"{v:g}"] += 1
    native = Counter(f"{_NATIVE[m.group(1)]:g}" for m in _NATIVE_RX.finditer(text))
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
    lo_src, lo_out = src.lower(), out.lower()

    def found(e: str, hay: str) -> bool:
        return e.lower() in hay or any(a in hay for a in al.get(e.lower(), ()))

    missing = sorted(e for e in _entities(src, s_toks) if not found(e, lo_out))
    added = sorted(e for e in _entities(out, o_toks) if not found(e, lo_src))
    issues = []
    # 두 글자 한글 NNP는 Kiwi가 일반명사(송신 등)를 잘못 붙이는 일이 잦아 경고(minor)로만 둔다
    strong = [e for e in missing if not re.fullmatch(r"[가-힣]{2}", e)]
    weak = [e for e in missing if e not in strong]
    if strong:
        issues.append(_issue("ENTITY_MISSING", f"원문 고유명사가 변환문에 없음: {', '.join(strong)}"))
    if weak:
        issues.append(_issue("ENTITY_WEAK", f"두 글자 고유명사(오분석 가능)가 변환문에 없음: {', '.join(weak)}"))
    if weak and added and not strong:
        # 두 글자 이름이 사라지고 다른 이름이 생겼으면 오분석이 아니라 바꿔치기로 본다
        issues = [_issue("ENTITY_MISSING", f"고유명사가 바뀐 것으로 보임: {', '.join(weak)} → {', '.join(added)}")]
    if added:
        from_ex = [e for e in added if example_text and e.lower() in example_text.lower()]
        note = f"원문에 없는 고유명사: {', '.join(added)}" + (f" (그중 검색 예시에 있던 것: {', '.join(from_ex)})" if from_ex else "")
        issues.append(_issue("EXAMPLE_COPY" if from_ex else "ENTITY_ADDED", note))
    return issues


def _is_neg(toks, i) -> bool:
    """부정 표지인가. '변함없다·끊임없다·틀림없다'처럼 명사형(ETN) 뒤 '없'은 굳은 말이라 뺀다."""
    return toks[i] in NEG_MORPHS and not (toks[i][0] == "없" and i and toks[i - 1][1] == "ETN")


def _has_neg(toks) -> bool:
    return any(_is_neg(toks, i) for i in range(len(toks)))


def _neg_predicates(toks) -> list[str]:
    """부정이 걸린 서술어(내용어)들. '발생하지 않았다'→발생, '효과가 없다'→효과, '못 간다'→가."""
    out = []
    for i, x in enumerate(toks):
        if not _is_neg(toks, i):
            continue
        rng = range(i + 1, min(len(toks), i + 4)) if x[1] == "MAG" else range(i - 1, max(-1, i - 8), -1)
        for j in rng:
            f, t = toks[j]
            if t.startswith(CONTENT_TAGS) and f not in STOP_STEMS:
                out.append(f)
                break
    return out


def _aligned(lem: set[str], other: dict):
    """lem과 가장 닮은 상대 문장(내용어 덮는 비율)과 그 비율."""
    best, idx = 0.0, -1
    for i, ol in enumerate(other["sent_lem"]):
        c = _coverage(lem, ol)
        if c > best:
            best, idx = c, i
    return idx, best


def _flips(a: dict, b: dict, preds_of, count) -> list[tuple[str, str]]:
    """a 문장에서 표지(부정·추측)가 걸린 서술어 p마다, b의 대응 문장에서
    (p에 같은 표지가 걸려 있거나) (표지 개수가 a 문장 이상이면) 정상, 아니면 뒤집힘.
    대응 문장 = b에서 p가 든 문장들. p가 b에 없으면(바꿔 말함) a 문장과 ALIGN_MIN 이상 닮은 b 문장과
    그 앞뒤 문장(나누기·합치기 대비). 개수 조건 덕에 '적응하는 방식이 아니다'↔'적응하지 않는다'처럼
    표지가 다른 낱말에 붙은 바꿔 말하기는 통과하고, 원래 부정이 있던 문장에 부정이 하나 더 생긴 것은 잡는다."""
    flips = []
    for s, st, sl in zip(a["sents"], a["sent_toks"], a["sent_lem"]):
        need = count(st)
        for p in preds_of(st):
            hits = [bt for bt, bl in zip(b["sent_toks"], b["sent_lem"]) if _present(p, bl)]
            if not hits:
                i, cov = _aligned(sl, b)
                hits = b["sent_toks"][max(0, i - 1):i + 2] if i >= 0 and cov >= ALIGN_MIN else []
            if hits and not any(_present(p, set(preds_of(bt))) or count(bt) >= need for bt in hits) \
                    and sum(count(bt) for bt in hits) < need * max(1, len(hits)):
                flips.append((p, s))
                break  # 한 문장에서 하나만
    return flips


def _neg_flips(a: dict, b: dict) -> list[tuple[str, str]]:
    return _flips(a, b, _neg_predicates, lambda st: sum(_is_neg(st, i) for i in range(len(st))))


def _hedge_positions(toks) -> list[int]:
    """추측 표현이 시작하는 토큰 위치. '알 수 있다'(결론 제시)는 추측이 아니라 뺀다."""
    pos = []
    for i, (f, t) in enumerate(toks):
        nxt = toks[i + 1][0] if i + 1 < len(toks) else ""
        if nxt in ("도", "는", "가") and i + 2 < len(toks):  # '할 수도 있다', '할 수는 있다'
            nxt = toks[i + 2][0]
        prv = toks[i - 2][0] if i >= 2 else ""
        if (f == "수" and t == "NNB" and nxt in ("있", "없") and prv != "알") or (f == "것" and t == "NNB" and nxt == "같") \
                or f in ("듯", "듯하") or (f == "보이" and t.startswith("VV")) or f in HEDGE_WORDS \
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


def _hedge_flips(a: dict, b: dict) -> list[tuple[str, str]]:
    """a에서 추측이 걸린 서술어가 b에서 추측 없이 나오면 단정으로 바뀐 것."""
    return _flips(a, b, _hedged_predicates, lambda st: len(_hedge_positions(st)))


def _sent_with(sents, sent_toks, pred) -> str:
    return next((s for s, st in zip(sents, sent_toks) if pred(st)), "")


def check_polarity(a: dict, b: dict) -> list[dict]:
    """부정(서술어 단위)·추측(개수)·범위어(개수). 근거로 해당 표현이 든 문장을 붙인다."""
    issues = []
    for (p, sent), what, side in [(x, "원문에서 부정→변환문에서 긍정", "source") for x in _neg_flips(a, b)] + \
                                 [(x, "원문에서 긍정→변환문에서 부정", "output") for x in _neg_flips(b, a)]:
        issues.append(_issue("NEGATION_FLIP", f"'{p}': {what}", sent, side))
    flips = [(x, "원문 추측→변환문 단정", "source") for x in _hedge_flips(a, b)] + \
            [(x, "원문 단정→변환문 추측", "output") for x in _hedge_flips(b, a)]
    for (p, sent), what, side in flips:
        issues.append(_issue("MODALITY_SHIFT", f"'{p}': {what}", sent, side))
    hs, ho = _hedge_count(a["toks"]), _hedge_count(b["toks"])
    if hs != ho and not flips:
        issues.append(_issue("MODALITY_MINOR", f"추측 표현(수 있다·것 같다·보인다 등) {hs}→{ho}개 (서술어 단위 변화는 없음)"))
    for table, kind in ((STRONG_SCOPE, "SCOPE_CHANGE"), (WEAK_SCOPE, "SCOPE_MINOR")):
        cs = Counter(x for x in a["toks"] if x in table)
        co = Counter(x for x in b["toks"] if x in table)
        gone, new = cs - co, co - cs
        if gone or new:
            parts = ([f"빠짐: {', '.join(f for f, _ in gone)}"] if gone else []) + \
                    ([f"생김: {', '.join(f for f, _ in new)}"] if new else [])
            issues.append(_issue(kind, "범위어 " + " / ".join(parts)))
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


def check_content(a: dict, b: dict, example_text: str = "") -> tuple[list[dict], dict]:
    """문장 단위 누락·추가 + 글 전체 재현율 + 예시 베낌."""
    issues = []
    la, lb = _lemmas(a["toks"]), _lemmas(b["toks"])
    pa, pb = set(la), set(lb)
    def local(lem: set[str], other: dict) -> float:
        # 상대 글의 이웃한 두 문장(i, i+1) 중 가장 잘 덮는 곳. 문장 합치기·나누기는 허용, 멀리 흩어진 낱말은 인정 안 함
        ls = other["sent_lem"] or [set()]
        return max(_coverage(lem, ls[i] | (ls[i + 1] if i + 1 < len(ls) else set())) for i in range(len(ls)))

    for s, lem in zip(a["sents"], a["sent_lem"]):
        if len(lem) >= SENT_MIN_LEMMAS and local(lem, b) < SENT_COVERAGE_MIN:
            miss = sorted(l for l in lem if not _present(l, pb))
            note = f"빠진 말: {', '.join(miss[:8])}" if miss else "낱말은 다른 곳에 흩어져 있으나 이 문장에 대응하는 곳이 없음"
            issues.append(_issue("OMISSION", f"원문 문장에 대응하는 변환문 문장이 없음 ({note})", s, "source"))
    ex_pool = set(_lemmas(_analyze(example_text)["toks"])) if example_text else set()
    for s, lem in zip(b["sents"], b["sent_lem"]):
        if len(lem) >= SENT_MIN_LEMMAS and local(lem, a) < SENT_COVERAGE_MIN:
            new = sorted(l for l in lem if not _present(l, pa))
            if ex_pool and new and _coverage(new, ex_pool) >= EXAMPLE_CONTENT_MIN:
                issues.append(_issue("EXAMPLE_COPY", f"원문에 없는 문장인데 내용어가 검색 예시에서 옴 ({', '.join(new[:8])})", s, "output"))
            else:
                issues.append(_issue("ADDITION", f"원문에 대응하는 곳이 없는 문장 (새 말: {', '.join(new[:8]) or '없음'})", s, "output"))
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
    ex = _example_text(examples)
    a, b = _analyze(source), _analyze(output)
    issues += check_numbers(source, output)
    issues += check_entities(source, output, a["toks"], b["toks"], ex)
    issues += check_polarity(a, b)
    c_issues, c_stats = check_content(a, b, ex)
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
        out.append(line + f" → {FIX_HINTS.get(i['type'], '원문과 뜻이 같게 고치세요.')}")
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
