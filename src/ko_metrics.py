"""한→한 결과의 AI 획일화·스타일 지표 (데모 표시용, API 없음). 한→영 대시보드(app.py의 cliche_rate·
lexical_diversity·avg_sentence_length·length_match)의 한국어판.

- AI 표현 비율: data/ko_ai_markers.json(scripts/build_ko_ai_markers.py가 짝 데이터에서 뽑은 후보, 사람 검토 전)의
  표현이 100어절당 몇 번 나오나. 손으로 고른 사전이 아니라 "같은 내용의 AI 글 ↔ 사람 글"에서 AI 쪽에 많았던 것.
- '-습니다' 비율: 종결어미 중 합쇼체 비율(말투, AI 표현과 따로).
- 어휘 다양도: MTLD(내용어, style_features.lexical_diversity와 같은 계산).
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


def hapsyo_rate(text: str) -> float:
    ef = [t.form for t in _kiwi().tokenize(text) if t.tag == "EF"]
    return sum("니다" in f for f in ef) / len(ef) if ef else 0.0


def avg_sentence_length(text: str) -> float:
    sents = [s.text for s in _kiwi().split_into_sents(text.strip()) if s.text.strip()] if text.strip() else []
    return sum(len(s.split()) for s in sents) / len(sents) if sents else 0.0


def lexical_diversity(text: str) -> float:
    from .style_features import lexical_diversity as mtld, split_sentences
    return mtld(split_sentences(text))


def length_match(text: str, ref_len: float) -> float:
    if not ref_len:
        return 0.0
    return max(0.0, 1 - abs(avg_sentence_length(text) - ref_len) / ref_len)


@lru_cache(maxsize=8)
def person_text(user: str) -> str:
    p = config.RAW_DIR / f"corpus_{user}_raw.jsonl"
    if not p.exists():
        return ""
    docs = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    keep = [d["text"] for d in docs if d.get("type") in ("essay", "report", "blog")] or [d["text"] for d in docs]
    return "\n\n".join(keep)


@lru_cache(maxsize=8)
def person_stats(user: str) -> dict:
    t = person_text(user)
    if not t:
        return {}
    return {"marker_rate": marker_rate(t), "hapsyo": hapsyo_rate(t), "mtld": lexical_diversity(t), "sent_len": avg_sentence_length(t)}


def table(source: str, converted: str, user: str, person_label: str) -> list[dict]:
    """[{행 이름, 지표...}] 입력 / 변환문 / 참고: 목표 인물 실제 글."""
    ref = person_stats(user)
    ref_len = ref.get("sent_len", 0.0)
    rows = []
    for name, t in (("입력 (AI 글)", source), ("변환문", converted)):
        rows.append({"구분": name, "AI 표현 (100어절당)": marker_rate(t), "'-습니다' 비율": hapsyo_rate(t),
                     "어휘 다양도 (MTLD)": lexical_diversity(t), "평균 문장 길이 (어절)": avg_sentence_length(t),
                     "문장 길이 일치도": length_match(t, ref_len) if ref else None})
    if ref:
        rows.append({"구분": f"참고 · {person_label} 실제 글", "AI 표현 (100어절당)": ref["marker_rate"], "'-습니다' 비율": ref["hapsyo"],
                     "어휘 다양도 (MTLD)": ref["mtld"], "평균 문장 길이 (어절)": ref_len, "문장 길이 일치도": 1.0})
    return rows
