"""한→한 결과의 AI 획일화·스타일 지표 (데모 표시용, API 없음). 한→영 대시보드(app.py의 cliche_rate·
lexical_diversity·avg_sentence_length·length_match)의 한국어판.

- AI식 쉼표(KatFishNet, Park 외 ACL 2025: LLM이 사람보다 쉼표를 많이 쓰고 특히 연결어미 뒤에 붙임):
  쉼표 있는 문장 비율, 연결어미(EC) 뒤 쉼표 비율. 우리 데이터 확인(D-108, 소표본): AI 입력 20편 대 팀 원문 67편
  AUC 0.84·0.91. 예전 AI 표현 사전(data/ko_ai_markers.json)은 같은 확인에서 AUC 0.56이라 화면에서 뺌(함수는 남김).
- '-습니다' 비율: 종결어미 중 합쇼체 비율(말투, AI 표현과 따로).
- 어휘 다양도: MATTR(내용어 50개 창의 평균 TTR, 0~1). 절대값은 AI 글이 사람 글보다 높게 나와(D-109: AI 0.90, 팀 0.76)
  "높을수록 좋다"로 읽으면 안 되므로, 화면에는 목표 인물 실제 글과의 일치도(문장 길이 일치도와 같은 식)로 보여 줌.
  예전 MTLD(내용어)는 비선형이라 차이가 과장돼(AI 260 대 팀 60) 표에서 뺌(함수는 남김).
- 평균 문장 길이(어절), 문장 길이 일치도(목표 인물 실제 글 기준, 1에 가까울수록 비슷).
목표 인물 실제 글 = data/raw/corpus_<사람>_raw.jsonl 전체(데모 기준, 평가 문서 제외 없음).
"""
from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def load_markers() -> dict:
    p = config.DATA_DIR / "ko_ai_markers.json"
    if not p.exists():
        return {"reviewed": False, "markers": [], "missing": True}
    return json.loads(p.read_text(encoding="utf-8"))


def _kiwi():
    from .ko_style import _kiwi as k
    return k()


def _units(text: str) -> Counter:
    toks = [(t.form, t.tag) for t in _kiwi().tokenize(text) if not t.tag.startswith("S")]
    u = [f"{f}/{g}" for f, g in toks]
    return Counter(u + [u[i] + " " + u[i + 1] for i in range(len(u) - 1)])


def find_markers(text: str, kind: str = "ai_marker") -> Counter:
    """{표시용 표현: 횟수}"""
    c = _units(text)
    return Counter({m["label"]: c[m["unit"]] for m in load_markers()["markers"] if m["kind"] == kind and c[m["unit"]]})


def marker_rate(text: str) -> float:
    n = len(text.split())
    return 100.0 * sum(find_markers(text).values()) / n if n else 0.0


def comma_sentence_rate(text: str) -> float:
    """쉼표가 하나 이상 있는 문장의 비율(KatFishNet 쉼표 지표)."""
    sents = [s.text for s in _kiwi().split_into_sents(text.strip()) if s.text.strip()] if text.strip() else []
    return sum("," in s for s in sents) / len(sents) if sents else 0.0


def comma_after_ec_rate(text: str) -> float:
    """연결어미(EC) 바로 뒤에 쉼표가 오는 비율(KatFishNet: LLM이 사람보다 훨씬 높음)."""
    toks = _kiwi().tokenize(text)
    ec = [i for i, t in enumerate(toks) if t.tag == "EC"]
    return sum(1 for i in ec if i + 1 < len(toks) and toks[i + 1].form == ",") / len(ec) if ec else 0.0


def hapsyo_rate(text: str) -> float:
    ef = [t.form for t in _kiwi().tokenize(text) if t.tag == "EF"]
    return sum("니다" in f for f in ef) / len(ef) if ef else 0.0


def avg_sentence_length(text: str) -> float:
    sents = [s.text for s in _kiwi().split_into_sents(text.strip()) if s.text.strip()] if text.strip() else []
    return sum(len(s.split()) for s in sents) / len(sents) if sents else 0.0


def lexical_diversity(text: str) -> float:
    from .style_features import lexical_diversity as mtld, split_sentences
    return mtld(split_sentences(text))


def mattr(text: str, window: int = 50) -> float | None:
    """MATTR: 내용어를 window개씩 밀며 잰 TTR의 평균(0~1). 길이에 거의 영향을 받지 않음. 내용어가 window개 미만이면 None."""
    from .style_features import CONTENT_TAGS
    c = [t.form for t in _kiwi().tokenize(text) if t.tag in CONTENT_TAGS]
    if len(c) < window:
        return None
    return sum(len(set(c[i:i + window])) for i in range(len(c) - window + 1)) / (window * (len(c) - window + 1))


def ratio_match(value: float | None, ref: float | None) -> float | None:
    """목표 인물 값과의 일치도 = 1 − |값 − 목표| / 목표 (0~1, 1이면 같음). length_match와 같은 식."""
    if value is None or not ref:
        return None
    return max(0.0, 1 - abs(value - ref) / ref)


def length_match(text: str, ref_len: float) -> float:
    if not ref_len:
        return 0.0
    return max(0.0, 1 - abs(avg_sentence_length(text) - ref_len) / ref_len)


@lru_cache(maxsize=8)
def person_docs(user: str) -> tuple[str, ...]:
    """목표 인물 실제 글(글 한 편씩). essay·report·blog, 없으면 전체."""
    p = config.RAW_DIR / f"corpus_{user}_raw.jsonl"
    if not p.exists():
        return ()
    docs = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return tuple([d["text"] for d in docs if d.get("type") in ("essay", "report", "blog")] or [d["text"] for d in docs])


@lru_cache(maxsize=8)
def person_text(user: str) -> str:
    return "\n\n".join(person_docs(user))


BAND_METRICS = {"comma_ec": comma_after_ec_rate, "mattr": mattr, "sent_len": avg_sentence_length}


@lru_cache(maxsize=8)
def person_bands(user: str) -> dict:
    """목표 인물 글 한 편씩의 값 분포 → {지표: (초록 하한, 초록 상한, 노랑 하한, 노랑 상한, 글 수)}. 색 범위 기준.
    참고용이라 넉넉하게: 초록 = 그 사람 글의 10~90%, 노랑 = 사분위 범위 1.5배 이상치 경계 안, 그 밖은 빨강.
    본인 글을 한 편씩 빼고 넣어 보면 대부분 초록·빨강 1편 안팎(D-110). 경계값을 손으로 정하지 않음."""
    from statistics import quantiles
    out = {}
    for k, f in BAND_METRICS.items():
        vals = sorted(v for v in (f(t) for t in person_docs(user)) if v is not None)
        if len(vals) >= 4:
            dec = quantiles(vals, n=10)
            q1, med, q3 = quantiles(vals, n=4)
            lo, hi = min(q1 - 1.5 * (q3 - q1), dec[0]), max(q3 + 1.5 * (q3 - q1), dec[-1])  # 노랑 경계가 초록을 포함하게
            out[k] = (dec[0], dec[-1], lo, hi, len(vals), med)
    return out


def band(value: float | None, b: tuple | None) -> str | None:
    """'green' = 그 사람 글의 10~90% 안, 'yellow' = 이상치 경계 안, 'red' = 경계 밖. 기준이 없으면 None."""
    if value is None or not b:
        return None
    g1, g2, y1, y2 = b[:4]
    return "green" if g1 <= value <= g2 else "yellow" if y1 <= value <= y2 else "red"


def spread_match(value: float | None, b: tuple | None) -> float | None:
    """색과 같은 잣대의 일치도(D-112): 그 사람 글의 중앙값과 같으면 1, 이상치 경계(노랑 끝)에 닿거나 넘으면 0, 사이는 거리에 비례.
    사람·지표마다 원래 흔들리는 폭이 반영됨(어휘 다양도처럼 폭이 좁으면 조금만 달라도 크게 떨어짐)."""
    if value is None or not b:
        return None
    y1, y2, med = b[2], b[3], b[5]
    edge = (y2 - med) if value >= med else (med - y1)
    return max(0.0, 1 - abs(value - med) / edge) if edge > 0 else (1.0 if value == med else 0.0)


@lru_cache(maxsize=8)
def person_stats(user: str) -> dict:
    t = person_text(user)
    if not t:
        return {}
    return {"comma_sent": comma_sentence_rate(t), "comma_ec": comma_after_ec_rate(t), "hapsyo": hapsyo_rate(t), "mattr": mattr(t), "sent_len": avg_sentence_length(t)}


def table(source: str, converted: str, user: str, person_label: str) -> list[dict]:
    """[{행 이름, 지표...}] 입력 / 변환문 / 참고: 목표 인물 실제 글."""
    ref = person_stats(user)
    ref_len = ref.get("sent_len", 0.0)
    bands = person_bands(user)
    rows = []
    for name, t in (("입력 (AI 글)", source), ("변환문", converted)):
        rows.append({"구분": name, "연결어미 뒤 쉼표 비율": comma_after_ec_rate(t), "쉼표 있는 문장 비율": comma_sentence_rate(t),
                     "'-습니다' 비율": hapsyo_rate(t),
                     "어휘 다양도 (MATTR)": (m := mattr(t)), "어휘 다양도 일치도": spread_match(m, bands.get("mattr")),
                     "평균 문장 길이 (어절)": (sl := avg_sentence_length(t)),
                     "문장 길이 일치도": spread_match(sl, bands.get("sent_len"))})
    if ref:
        rows.append({"구분": f"참고 · {person_label} 실제 글", "연결어미 뒤 쉼표 비율": ref["comma_ec"], "쉼표 있는 문장 비율": ref["comma_sent"],
                     "'-습니다' 비율": ref["hapsyo"],
                     "어휘 다양도 (MATTR)": ref["mattr"], "어휘 다양도 일치도": spread_match(ref["mattr"], bands.get("mattr")),
                     "평균 문장 길이 (어절)": ref_len, "문장 길이 일치도": spread_match(ref_len, bands.get("sent_len"))})
    return rows
