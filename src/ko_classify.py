"""한→한 문체 변환 평가용 작성자 분류기 두 개 (D-047·D-052·D-053).

둘 다 같은 특징(기능 형태소, 지시 그룹 표현 뺌 = 순환 방지)을 쓰고, 실행 전에 설정을 고정한다.
  - SVM  : 학습용 문서의 문단(150~450자)으로 학습한 선형 SVM (기능 형태소 1~2gram, tf-idf, 클래스 균형)
  - Delta: 학습용 문서(글 한 편 단위)로 만든 사람별 중심 + 기능 형태소 top100 Burrows' Delta
"가장 가까운 사람"만 돌려주며 확률이 아니다. 평가 문서(나경 split)는 학습에서 제외한다. API 호출 없음.
kiwipiepy, scikit-learn, numpy 필요(sklearn은 이 모듈을 쓸 때만).
"""
from __future__ import annotations

import json
from collections import Counter

from src import config
from src.ko_style import PEOPLE, build_ref, func_tokens, zvec

MIN_C, MAX_C = 150, 450  # 학습·시험 문단 길이(자)
SVM_NGRAM, SVM_MIN_DF = (1, 2), 2


def _split_info() -> dict[str, set[str]]:
    info = json.loads((config.ROOT_DIR / "data/eval/split_info.json").read_text(encoding="utf-8"))
    return {u["user"]: set(u["heldout_docs"]) for u in info["users"]}


def _pair_docs() -> dict[str, set[str]]:
    """짝 데이터(scripts/ko_pairs.py select)에 쓴 문서. 파일이 없으면 빈 값(기존 동작 그대로)."""
    p = config.ROOT_DIR / "data/ko_pairs/selected.json"
    if not p.exists():
        return {}
    return {u: set(v) for u, v in json.loads(p.read_text(encoding="utf-8"))["pair_docs"].items()}


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if MIN_C <= len(p.strip()) <= MAX_C]


def load_corpus(heldout: dict[str, set[str]] | None = None) -> dict:
    """{"train_docs": [{author, toks}], "train_paras": [(author, text)], "eval_paras": [(author, text)]}.
    train = essay·report 중 평가 문서가 아닌 것. eval_paras = 평가 문서 문단(시험·상한용)."""
    heldout = heldout or _split_info()
    pair_docs = _pair_docs()
    out = {"train_docs": [], "train_paras": [], "eval_paras": []}
    for who in PEOPLE:
        for line in (config.RAW_DIR / f"corpus_{who}_raw.jsonl").open(encoding="utf-8"):
            if not line.strip():
                continue
            o = json.loads(line)
            if o["id"] in heldout.get(who, set()):
                out["eval_paras"] += [(who, p) for p in _paragraphs(o["text"])]
            elif o["id"] in pair_docs.get(who, set()):
                continue  # 짝 예시에 쓴 문서는 분류기 학습에서 뺌(D-081 문제 1). 모든 변형에 같은 분류기
            elif o.get("type") in ("essay", "report"):
                out["train_docs"].append({"author": who, "toks": func_tokens(o["text"], False)})
                out["train_paras"] += [(who, p) for p in _paragraphs(o["text"])]
    return out


class Classifiers:
    """SVM과 Delta를 한 번에 학습하고 같은 글을 두 방식으로 분류한다."""

    def __init__(self, corpus: dict | None = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import LinearSVC
        self.corpus = corpus or load_corpus()
        paras = self.corpus["train_paras"]
        self.svm = make_pipeline(
            TfidfVectorizer(token_pattern=r"\S+", ngram_range=SVM_NGRAM, sublinear_tf=True, min_df=SVM_MIN_DF),
            LinearSVC(class_weight="balanced"),
        ).fit([self._feat(t) for _, t in paras], [a for a, _ in paras])
        self.ref = build_ref(self.corpus["train_docs"], True)

    @staticmethod
    def _feat(text: str) -> str:
        return " ".join(func_tokens(text, True))

    def predict(self, texts: list[str]) -> dict[str, list[str]]:
        import numpy as np
        svm = [str(x) for x in self.svm.predict([self._feat(t) for t in texts])]
        delta = []
        for t in texts:
            z, _ = zvec(self.ref, t, True)
            d = {p: float(np.abs(z - self.ref["cents"][p]).mean()) for p in PEOPLE}
            delta.append(min(d, key=d.get))
        return {"SVM": svm, "Delta": delta}

    def scores(self, texts: list[str]) -> dict[str, list[dict[str, float]]]:
        """사람별 점수(클수록 그 사람 쪽). SVM = decision_function, Delta = -평균 거리. 배정 지표용(D-072)."""
        import numpy as np
        dec = self.svm.decision_function([self._feat(t) for t in texts])
        classes = [str(c) for c in self.svm.classes_]
        svm = [{c: float(row[i]) for i, c in enumerate(classes)} for row in dec]
        delta = []
        for t in texts:
            z, _ = zvec(self.ref, t, True)
            delta.append({p: -float(np.abs(z - self.ref["cents"][p]).mean()) for p in PEOPLE})
        return {"SVM": svm, "Delta": delta}

    def sizes(self) -> str:
        c = Counter(a for a, _ in self.corpus["train_paras"])
        return (f"SVM 학습 문단 {sum(c.values())}개 ({', '.join(f'{k} {v}' for k, v in c.items())}), "
                f"Delta 학습 문서 {len(self.corpus['train_docs'])}편, 시험용 평가 문서 문단 {len(self.corpus['eval_paras'])}개")


def confusion(true: list[str], pred: list[str]) -> list[list[int]]:
    c = Counter(zip(true, pred))
    return [[c[(t, p)] for p in PEOPLE] for t in PEOPLE]


def accuracy(true: list[str], pred: list[str]) -> float:
    return sum(a == b for a, b in zip(true, pred)) / max(1, len(true))
