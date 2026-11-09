"""한국어→한국어 문체 변환 (D-043·D-044·D-050). 중립적인 한국어를 한 사람의 문체로 다시 쓴다.

scripts/ko_style_transfer.py(평가 실험)와 app.py(데모)가 같은 코드를 쓰도록 모은 모듈입니다.
- 문체 지시(P): 목표 인물별 한국어 지시 초안. 강희·나경 줄은 입력에 해당 기능(이유·대조·결론 표현)이
  실제로 있을 때만 넣고(Kiwi 규칙, 모델 판단 아님, D-013), 수영 줄은 항상 넣는다.
- 본인 글 예시(R): 목표 인물의 코퍼스에서 입력과 비슷한 글 k개(config.DEFAULT_TOP_K)를 검색해 붙인다.
  평가 실험은 학습용 코퍼스(LINGUA_CORPUS_SPLIT=train)와 exclude_doc_ids로 평가 문서를 빼고,
  데모는 전체 코퍼스를 그대로 쓴다.
- 생성만 부르고 검증은 하지 않는다. 변환 결과의 의미 확인은 사람이 한다.
- 문체 판정(style_judgment): 기능 형태소 top100 Burrows' Delta로 세 사람 중 어느 쪽에 가까운지.
  글이 짧으면 불안정해 MIN_CHARS_FOR_STYLE_SCORE(1,000자, D-048) 이상일 때만 보여 주는 용도.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache

from . import config

PEOPLE = ["ganghee", "nakyung", "suyoung"]
MODES = ("P", "R", "P+R")
MIN_CHARS_FOR_STYLE_SCORE = 1000
SHORT_INPUT_NOTICE = "짧은 글은 문체 차이가 작게 나타납니다."

COMMON_RULES = [
    "원문에 없는 내용을 더하거나, 있는 내용을 빼지 않는다.",
    "원문의 확신 정도(추측·단정·완곡)를 그대로 유지한다.",
    "문장 수는 자유롭게 바꿔도 된다.",
]
# 목표 인물별 한국어 문체 지시 초안. 기능 이름이 있으면 입력에 그 기능이 있을 때만 넣는다(None은 항상).
# D-049의 어휘·명사화 지시 후보는 실험 결과를 본 뒤에 추가한다(지금은 넣지 않음).
P_LINES = {
    "ganghee": [("reason", "이유를 나타낼 때는 '-므로'나 '그래서'로 잇는다.")],
    "nakyung": [("contrast", "대조를 나타낼 때는 '그러나'를 쓴다."), ("conclusion", "결론을 나타낼 때는 '따라서'를 쓴다.")],
    "suyoung": [(None, "문장을 짧게 끊지 말고 길게 이어 쓴다.")],
}
# 입력에 기능이 있는지 판단(Kiwi 규칙, kiwipiepy가 없으면 정규식 근사). 모델이 아니라 규칙이 정한다(D-013).
GATE_KIWI = {
    "reason": [("므로", None), ("으므로", None), ("그래서", "MAJ"), ("때문", None), ("니까", None), ("으니까", None), ("왜냐하면", "MAJ")],
    "contrast": [("그러나", None), ("하지만", None), ("그렇지만", None), ("반면", None), ("지만", None)],
    "conclusion": [("따라서", "MAJ"), ("그러므로", "MAJ"), ("결론적", None), ("요컨대", None), ("결국", None)],
}
GATE_REGEX = {
    "reason": r"므로|그래서|때문|니까|왜냐하면",
    "contrast": r"그러나|하지만|그렇지만|반면|지만",
    "conclusion": r"(?<!에 )따라서|그러므로|결론적|요컨대|결국",
}


def _kiwi():
    """공용 Kiwi 인스턴스. kiwipiepy가 없으면 None."""
    try:
        from .style_features import _get_kiwi
        return _get_kiwi()
    except ImportError:
        return None


def detect_functions(text: str):
    """입력에 있는 기능 집합과 판단 방식('kiwi'|'regex')."""
    kiwi = _kiwi()
    if kiwi is not None:
        toks = [(t.form, t.tag) for t in kiwi.tokenize(text)]
        found = {g for g, rules in GATE_KIWI.items()
                 if any(f == form and (tag is None or tg.startswith(tag)) for form, tg in toks for f, tag in rules)}
        return found, "kiwi"
    return {g for g, rx in GATE_REGEX.items() if re.search(rx, text)}, "regex"


def p_lines_for(target: str, text: str):
    """(목표 인물에게 적용할 한국어 지시 줄 목록, 기능 판단 방식)."""
    present, mode = detect_functions(text)
    return [line for fn, line in P_LINES[target] if fn is None or fn in present], mode


def build_prompt(text: str, lines: list[str], examples: list[dict] | None) -> str:
    parts = ["다음 글을 한 필자의 문체로 다시 써 주세요. 뜻과 내용은 그대로 두세요.", "", "규칙:"]
    parts += [f"{i}. {r}" for i, r in enumerate(COMMON_RULES, 1)]
    if lines:
        parts += ["", "문체 지시:"] + [f"- {l}" for l in lines]
    if examples:
        parts += ["", "아래는 이 필자가 쓴 다른 글입니다. 말투와 표현 방식만 참고하고, 내용은 가져오지 마세요."]
        parts += [f"예시{i}: {e['text']}" for i, e in enumerate(examples, 1)]
    parts += ["", "원문:", text, "", "다시 쓴 글만 출력하세요. 설명이나 제목은 쓰지 마세요."]
    return "\n".join(parts)


def generate(prompt: str, label: str = "문체 변환") -> str:
    """생성 모델(config.GENERATION_MODEL)에 prompt를 보내 텍스트를 받는다. 검증 호출 없음."""
    from .gen_backend import generate_text
    from .retry import call_with_retry
    return call_with_retry(lambda: generate_text(prompt), label=label).strip()


def retrieve_examples(text: str, user: str, exclude_doc_ids: set[str] | None = None) -> list[dict]:
    """목표 인물 본인 글에서 입력과 비슷한 예시 k개: [{id, score, type, text}]."""
    from .retrieve import retrieve_similar
    ex = retrieve_similar(text, user, k=config.DEFAULT_TOP_K, exclude_doc_ids=exclude_doc_ids or None)
    return [{"id": x["id"], "score": round(float(x["score"]), 4), "type": x.get("type"), "text": x["text"]} for x in ex]


def convert_style(text: str, user: str, mode: str = "P+R", *, examples: list[dict] | None = None,
                  exclude_doc_ids: set[str] | None = None, dummy_output: str | None = None) -> dict:
    """text를 user의 문체로 바꾼다.

    mode: "P"(문체 지시만) / "R"(본인 글 예시만) / "P+R". 공통 규칙은 항상 들어간다.
    examples: 미리 검색해 둔 예시(평가 실험이 같은 검색 결과를 쓰려고 넘김). None이면 R일 때 여기서 검색한다.
    exclude_doc_ids: 검색에서 뺄 문서 id(평가 실험용). 데모는 None = 전체 코퍼스.
    dummy_output: 주면 검색·모델 호출 없이 이 문자열을 변환문으로 쓴다(데모 가짜 결과).
    반환: {"converted", "lines"(적용된 지시), "examples"(사용한 예시), "mode", "gate_mode", "prompt"}
    """
    if mode not in MODES:
        raise ValueError(f"mode는 {MODES} 중 하나여야 합니다: {mode!r}")
    if user not in P_LINES:
        raise ValueError(f"알 수 없는 사용자: {user!r}")
    all_lines, gate_mode = p_lines_for(user, text)  # R만 쓸 때도 판단 방식은 기록
    lines = all_lines if "P" in mode else []
    used: list[dict] = []
    if "R" in mode:
        if examples is not None:
            used = [dict(e) for e in examples]
        elif dummy_output is not None:
            used = [{"id": "dummy", "score": 0.0, "type": "예시", "text": "(데모용 가짜 예시)"}]
        else:
            used = retrieve_examples(text, user, exclude_doc_ids)
    prompt = build_prompt(text, lines, used or None)
    converted = dummy_output if dummy_output is not None else generate(prompt)
    return {"converted": converted, "lines": lines, "examples": used, "mode": mode, "gate_mode": gate_mode, "prompt": prompt}


# ───────── 문체 판정 (Burrows' Delta, 기능 형태소 top100) ─────────
# 지시에 쓴 기능어 표현(그러나·따라서·므로 등)을 뺀 버전을 기본으로 쓴다(순환 평가 방지, D-044).
EXCL_FORMS = {"그러나", "하지만", "그렇지만", "반면", "따라서", "그러므로", "므로", "으므로", "그래서"}
FUNC = ("J", "E", "XS", "VX", "VCP", "VCN", "NNB", "MAJ", "XP")


def func_tokens(text: str, exclude: bool) -> list[str]:
    toks = [(t.form, t.tag) for t in _kiwi().tokenize(text) if t.tag.startswith(FUNC)]
    return [f"{f}/{tg}" for f, tg in toks if not (exclude and f in EXCL_FORMS)]


def load_reference_docs(excluded_docs: dict[str, set[str]] | None = None) -> list[dict]:
    """data/raw의 essay·report 문서 [{author, toks}]. excluded_docs={사용자: {빼는 doc id}}로 평가 문서를 뺄 수 있다."""
    excluded_docs = excluded_docs or {}
    docs = []
    for who in PEOPLE:
        path = config.RAW_DIR / f"corpus_{who}_raw.jsonl"
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("type") in ("essay", "report") and o["id"] not in excluded_docs.get(who, set()):
                docs.append({"author": who, "toks": func_tokens(o["text"], False)})
    return docs


def build_ref(docs: list[dict], exclude: bool) -> dict:
    import numpy as np
    toks = [[x for x in d["toks"] if not (exclude and x.split("/")[0] in EXCL_FORMS)] for d in docs]
    vocab = [w for w, _ in Counter(x for t in toks for x in t).most_common(100)]
    X = np.array([[Counter(t)[w] / max(1, len(t)) for w in vocab] for t in toks])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    y = np.array([d["author"] for d in docs])
    return {"vocab": vocab, "mu": mu, "sd": sd, "cents": {a: Z[y == a].mean(0) for a in sorted(set(y))}}


def zvec(ref: dict, text: str, exclude: bool):
    import numpy as np
    t = func_tokens(text, exclude)
    c = Counter(t)
    return (np.array([c[w] / max(1, len(t)) for w in ref["vocab"]]) - ref["mu"]) / ref["sd"], len(t)


@lru_cache(maxsize=1)
def _demo_reference(exclude: bool) -> dict:
    return build_ref(load_reference_docs(), exclude)


def style_judgment(text: str, exclude_group_forms: bool = True, ref: dict | None = None) -> dict:
    """text가 세 사람 중 누구의 기능 형태소 분포에 가까운지. {"distances": {사용자: Delta}, "nearest", "tokens"}.

    ref를 안 주면 전체 코퍼스(평가 문서 제외 없음)로 만든 기준을 쓴다(데모용, 첫 호출에 수십 초).
    Delta는 짧은 글에서 불안정하므로 호출하는 쪽에서 MIN_CHARS_FOR_STYLE_SCORE 이상일 때만 보여 준다.
    """
    import numpy as np
    ref = ref or _demo_reference(exclude_group_forms)
    z, n = zvec(ref, text, exclude_group_forms)
    d = {p: float(np.abs(z - ref["cents"][p]).mean()) for p in PEOPLE}
    return {"distances": d, "nearest": min(d, key=d.get), "tokens": n}
