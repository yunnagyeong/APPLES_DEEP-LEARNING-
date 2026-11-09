"""한→한 변환의 형식 유지 검사 (D-056·D-057·D-063). 규칙으로만 계산하고 모델 판단은 쓰지 않는다.

원문(= 중립문, 변환의 입력)과 변환문을 비교해 형식이 달라졌는지 표시한다. 변환 출력은 바꾸지 않는 사후 측정이다.
임계값은 초안이다(선택용 문단에서 조정, 확인용 문단에서는 고정).

  문단 수(빈 줄 기준)      : 다르면 표시
  목록 줄 수               : 다르면 표시 (-, •, ·, *, 숫자., 숫자), ①… 로 시작하는 줄)
  명사형 종결 비율          : 차이 0.20 이상이면 표시 (문장이 -음/-함/-기/명사로 끝나는 비율)
  문장 수(Kiwi)            : 2문장 이상 줄거나 1문장 이상 늘면 표시 (수영 줄이 이웃 문장 잇기를 허용하므로 값은 항상 기록)
  글자 수 비율             : 0.8~1.25 밖이면 표시
"""
from __future__ import annotations

import re

from .ko_style import _kiwi

LIST_RX = re.compile(r"^\s*(?:[-•·*▪○◦■□▶▷※]|\d+[.)]|[①-⑳]|\(\d+\))\s*\S")
NOMINAL_DIFF, SENT_DROP, SENT_RISE, RATIO_LO, RATIO_HI = 0.20, 2, 1, 0.8, 1.25


def _sentences(text: str) -> list[str]:
    kiwi = _kiwi()
    if kiwi is None:
        return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]
    return [s.text.strip() for s in kiwi.split_into_sents(text.strip()) if s.text.strip()]


def _is_nominal_end(sentence: str) -> bool:
    """문장이 서술어 종결(EF)이 아니라 명사형·명사로 끝나는가."""
    kiwi = _kiwi()
    if kiwi is None:
        return bool(re.search(r"(?:음|함|됨|임|기|것)[.\s]*$", sentence.strip()))
    toks = [t for t in kiwi.tokenize(sentence) if not t.tag.startswith("S")]  # 문장부호·기호 제외
    return bool(toks) and (toks[-1].tag in ("ETN", "NNG", "NNP", "NNB", "NR", "SL", "SN"))


def stats(text: str) -> dict:
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    lines = [l for l in text.splitlines() if l.strip()]
    sents = _sentences(text)
    nominal = sum(_is_nominal_end(s) for s in sents)
    return {
        "paragraphs": len(paras),
        "list_lines": sum(bool(LIST_RX.match(l)) for l in lines),
        "sentences": len(sents),
        "nominal_rate": nominal / max(1, len(sents)),
        "chars": len(text.strip()),
    }


def check(source: str, output: str) -> dict:
    """{"pass": bool, "reasons": [사유...], "src": stats, "out": stats}. 원문=변환 입력(중립문)."""
    a, b = stats(source), stats(output)
    why = []
    if a["paragraphs"] != b["paragraphs"]:
        why.append(f"문단 {a['paragraphs']}→{b['paragraphs']}")
    if a["list_lines"] != b["list_lines"]:
        why.append(f"목록 줄 {a['list_lines']}→{b['list_lines']}")
    if abs(a["nominal_rate"] - b["nominal_rate"]) >= NOMINAL_DIFF:
        why.append(f"명사형 종결 {a['nominal_rate']:.2f}→{b['nominal_rate']:.2f}")
    d = b["sentences"] - a["sentences"]
    if d <= -SENT_DROP or d >= SENT_RISE:
        why.append(f"문장 {a['sentences']}→{b['sentences']}")
    ratio = b["chars"] / max(1, a["chars"])
    if not RATIO_LO <= ratio <= RATIO_HI:
        why.append(f"글자 비 {ratio:.2f}")
    return {"pass": not why, "reasons": why, "src": a, "out": b}
