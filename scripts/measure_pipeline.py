"""파이프라인 호출 수·구간 시간 측정 (투두 7·9번 기준값 / 7번 적용 후 검증용).

src/는 건드리지 않고, 실행할 때만 함수를 감싸서(monkeypatch) 호출 횟수와 시간을 잰다.

사용법 (프로젝트 루트에서):
  python scripts/measure_pipeline.py --mode offline
      API·NLI 모델 없이 가짜 함수로 호출 횟수만 센다 (재번역 0/1/2회 시나리오).
  python scripts/measure_pipeline.py --mode live --tag before7 [--sample 8] [--ids 1,5,9]
      실제 API·NLI로 error_cases.csv 케이스의 구간별 시간을 잰다 (.env의 API 키 필요).
      결과는 data/measure/latency_<tag>_<시각>.csv 로 저장된다.
      7번 적용 전후를 같은 --ids/--sample로 돌려 비교할 것 (태그만 바꿈).
"""

import argparse
import csv
import functools
import math
import os
import sys
import time
import types
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_USER = "ganghee"
ERROR_CASES = ROOT / "error_cases.csv"
OUT_DIR = ROOT / "data" / "measure"
WORST_CASE_CALLS = 10  # max_retries=2 기준 최악 호출 수 (검증 경량화 후. 7번 전 18, 7번 후 16)
PROMPTS = []  # 오프라인 모드에서 생성 모델에 보낸 프롬프트 기록


# --------------------------------------------------------------------------- 기록기
class Rec:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = defaultdict(int)
        self.t = defaultdict(float)
        self.retries = 0
        self.top_score = None

    def add(self, key, dt):
        self.n[key] += 1
        self.t[key] += dt


REC = Rec()


def timed(key, fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            REC.add(key, time.perf_counter() - t0)

    return wrapper


def _wrap_call_with_retry(orig):
    """생성/재생성 호출을 label로 구분해 시간·횟수를 재고, 재시도(503 등) 횟수도 센다."""

    def wrapper(fn, *, label, **kw):
        if label.startswith("translate_with_style"):
            key = "gen"
        elif label.startswith("retranslation"):
            key = "regen"
        else:
            key = "other_api"

        def counted():
            try:
                return fn()
            except Exception:
                REC.retries += 1
                raise

        t0 = time.perf_counter()
        try:
            return orig(counted, label=label, **kw)
        finally:
            REC.add(key, time.perf_counter() - t0)

    return wrapper


def _wrap_retrieve(orig):
    @functools.wraps(orig)
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            res = orig(*a, **kw)
            if REC.top_score is None and res:
                REC.top_score = res[0]["score"]
            return res
        finally:
            REC.add("rag", time.perf_counter() - t0)

    return wrapper


def install_hooks():
    from src import generate, pipeline, retrieve, verify

    for mod in (generate, pipeline):
        mod.retrieve_similar = _wrap_retrieve(mod.retrieve_similar)
        mod.call_with_retry = _wrap_call_with_retry(mod.call_with_retry)
    retrieve.load_indexed_examples = timed("rag_load", retrieve.load_indexed_examples)
    retrieve.embed_text = timed("rag_embed", retrieve.embed_text)
    verify.embed_text = timed("sim_embed", verify.embed_text)
    verify.embed_texts = timed("sim_embed", verify.embed_texts)
    verify.extract_claims = timed("claims", verify.extract_claims)
    verify.check_semantic_nuance = timed("claims", verify.check_semantic_nuance)
    verify.analyze_meaning = timed("claims", verify.analyze_meaning)  # 묶음 호출(claim+뉘앙스 1회)
    verify.nli_check_batch = timed("nli", verify.nli_check_batch)
    verify.call_with_retry = _wrap_call_with_retry(verify.call_with_retry)
    verify.verify_translation = timed("verify", verify.verify_translation)


# --------------------------------------------------------------------------- 오프라인
def _install_ml_stubs():
    """torch/transformers를 가짜로 대체해 verify.py를 모델 로딩 없이 import한다."""
    torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    torch.no_grad = lambda: _NoGrad()
    transformers = types.ModuleType("transformers")

    class _Dummy:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

    transformers.AutoTokenizer = _Dummy
    transformers.AutoModelForSequenceClassification = _Dummy
    sys.modules["torch"] = torch
    sys.modules["transformers"] = transformers


def _install_offline_fakes(dim, fake_state):
    import numpy as np

    from src import generate, pipeline, retrieve, verify

    def fake_embed(text, task_type="RETRIEVAL_DOCUMENT"):
        rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
        return rng.standard_normal(dim).tolist()

    retrieve.embed_text = fake_embed
    verify.embed_text = fake_embed
    verify.embed_texts = lambda texts, task_type="RETRIEVAL_DOCUMENT": [fake_embed(t) for t in texts]

    class _Models:
        def generate_content(self, model, contents):
            PROMPTS.append(contents)
            return SimpleNamespace(text="Fake translation.")

    fake_client = SimpleNamespace(models=_Models())
    generate._ensure_client = lambda: fake_client
    pipeline._ensure_client = lambda: fake_client
    no_err = {"has_meaning_error": False, "error_types": [], "explanation": ""}
    verify.extract_claims = lambda text, language="ko": ["claim"]
    verify.check_semantic_nuance = lambda s, t: dict(no_err)
    verify.analyze_meaning = lambda s, t, source_claims=None: {
        "source_claims": source_claims or ["claim"], "translation_claims": ["claim"], "semantic_nuance": dict(no_err)}
    verify.nli_check_batch = lambda pairs: [
        {"label": fake_state["label"], "confidence": 1.0, "scores": {}} for _ in pairs]


def run_offline(user):
    os.environ.setdefault("GOOGLE_API_KEY", "offline-dummy")
    _install_ml_stubs()
    from src.store import load_indexed_examples

    _, vectors = load_indexed_examples(user)
    fake_state = {"label": "entailment"}
    _install_offline_fakes(len(vectors[0]), fake_state)
    install_hooks()

    from src import pipeline, verify

    profile = {
        "sentence_length_avg": 8.0, "formality_score": 0.5, "directness_score": 0.5,
        "logical_connective_rate": 0.5, "hedging_rate": 0.5, "conclusion_position": "mixed",
        "example_usage_rate": 0.2, "lexical_diversity": 60.0, "paragraph_length_avg": 3.0,
        "source_example_count": 0,
    }
    base_verify = verify.verify_translation

    rows = []
    for fail_rounds in (0, 1, 2):  # 앞의 N번 검증은 FAIL, 그 뒤는 PASS → 재번역 N회
        REC.reset()
        PROMPTS.clear()
        counter = {"round": 0}

        def controlled(*a, _counter=counter, _fr=fail_rounds, **kw):
            _counter["round"] += 1
            fake_state["label"] = "contradiction" if _counter["round"] <= _fr else "entailment"
            return base_verify(*a, **kw)

        verify.verify_translation = controlled
        result = pipeline.translate_with_verification("측정용 문장입니다.", profile, user, max_retries=2)
        n = REC.n
        api_total = n["gen"] + n["regen"] + n["claims"] + n["rag_embed"] + n["sim_embed"]  # claims = 검증용 3.6-flash 호출
        rows.append([
            fail_rounds, result["attempts"], result["verdict"], n["rag"], n["rag_load"],
            n["rag_embed"], n["gen"] + n["regen"], n["claims"], n["sim_embed"], n["nli"], api_total,
            zlib.crc32("\x00".join(PROMPTS).encode("utf-8")),
        ])
    verify.verify_translation = base_verify

    head = ["재번역", "시도", "verdict", "검색", "디스크로드", "검색임베딩", "번역생성",
            "claim추출", "유사도임베딩", "NLI(로컬)", "API합계", "프롬프트crc"]
    print("\n=== 호출 수 (max_retries=2, 가짜 함수로 센 값) ===")
    print("  ".join(f"{h:>8}" for h in head))
    for r in rows:
        print("  ".join(f"{str(v):>8}" for v in r))
    print("\n기대값 — 7번 전 API합계 6/12/18 · 7번 후 6/11/16 (뉘앙스 점검 추가 전 기준)")
    print("        경량화 후(VERIFY_SINGLE_CALL=True) API합계 4/7/10, claim추출(=3.6-flash) 1/2/3")
    print("API합계 = 번역생성 + claim추출 + 검색임베딩 + 유사도임베딩 (NLI는 로컬이라 제외)")
    print("프롬프트crc = 생성 모델에 보낸 프롬프트 전체의 해시. 7번 전후로 같으면 번역 입력이 동일")


# --------------------------------------------------------------------------- 실측
def load_cases(ids, sample):
    with open(ERROR_CASES, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if ids:
        wanted = {i.strip() for i in ids.split(",")}
        return [r for r in rows if r["번호"] in wanted]
    bad = [r for r in rows if r["문제유형"].strip() != "없음"]
    ok = [r for r in rows if r["문제유형"].strip() == "없음"]
    n_ok = round(sample * 3 / 8)
    n_bad = sample - n_ok
    pick_bad = bad[:: max(1, len(bad) // max(1, n_bad))][:n_bad]
    pick_ok = ok[:: max(1, len(ok) // max(1, n_ok))][:n_ok]
    return pick_bad + pick_ok


def run_live(user, tag, sample, ids):
    install_hooks()
    from src import pipeline

    cases = load_cases(ids, sample)
    print(f"[measure] {len(cases)}개 케이스, 최대 약 {len(cases) * WORST_CASE_CALLS}회 API 호출 "
          f"(재번역이 없으면 훨씬 적음). user={user}, tag={tag}")
    profile = pipeline._load_or_compute_style_profile(user)

    out = []
    for r in cases:
        REC.reset()
        t0 = time.perf_counter()
        err, result = "", None
        try:
            result = pipeline.translate_with_verification(r["원문"], profile, user)
        except Exception as e:  # 한 건이 죽어도 나머지는 계속 측정
            err = f"{type(e).__name__}: {e}"
        total = time.perf_counter() - t0
        t, n = REC.t, REC.n
        row = {
            "번호": r["번호"], "사람라벨": r["문제유형"][:20],
            "재번역": (result["attempts"] - 1) if result else "",
            "verdict": result["verdict"] if result else "ERROR",
            "total_s": total,
            "rag_s": t["rag"], "rag_load_s": t["rag_load"], "rag_embed_s": t["rag_embed"],
            "gen_s": t["gen"], "regen_s": t["regen"],
            "verify_s": t["verify"], "claims_s": t["claims"],
            "sim_embed_s": t["sim_embed"], "nli_s": t["nli"],
            "other_s": total - t["rag"] - t["gen"] - t["regen"] - t["verify"],
            "n_rag": n["rag"], "n_gen": n["gen"] + n["regen"], "n_claims": n["claims"],
            "n_sim_embed": n["sim_embed"], "n_nli": n["nli"],
            "api_total": n["gen"] + n["regen"] + n["claims"] + n["rag_embed"] + n["sim_embed"],
            "api_retries": REC.retries,
            "top_score": "" if REC.top_score is None else round(REC.top_score, 3),
            "error": err,
        }
        out.append(row)
        flag = "  ⚠ API 재시도 포함(시간 왜곡)" if REC.retries else ""
        print(f"[{row['번호']:>2}] 재번역={row['재번역']} {row['verdict']:<6} "
              f"{total:6.1f}s (검색 {row['rag_s']:.2f} / 생성 {row['gen_s'] + row['regen_s']:.1f} / "
              f"검증 {row['verify_s']:.1f}) API {row['api_total']}회{flag}{(' ERR ' + err) if err else ''}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"latency_{tag}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print("\n=== 재번역 횟수별 평균 (n이 작아 경향만 참고) ===")
    groups = defaultdict(list)
    for row in out:
        if row["verdict"] != "ERROR" and row["api_retries"] == 0:
            groups[row["재번역"]].append(row)
    for k in sorted(groups):
        g = groups[k]
        mean = lambda key: sum(x[key] for x in g) / len(g)  # noqa: E731
        print(f"재번역 {k}회 (n={len(g)}): 전체 {mean('total_s'):.1f}s · 검색 {mean('rag_s'):.2f}s · "
              f"생성 {mean('gen_s') + mean('regen_s'):.1f}s · claim {mean('claims_s'):.1f}s · "
              f"NLI {mean('nli_s'):.1f}s · API {mean('api_total'):.1f}회")
    skipped = len(out) - sum(len(g) for g in groups.values())
    if skipped:
        print(f"(API 재시도가 있었거나 에러난 {skipped}건은 평균에서 제외)")
    print(f"\n저장: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mode", choices=["offline", "live"], required=True)
    ap.add_argument("--user", default=DEFAULT_USER)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--ids", default="", help="error_cases.csv 번호를 쉼표로 (예: 1,5,9). 주면 --sample 무시")
    args = ap.parse_args()
    if args.mode == "offline":
        run_offline(args.user)
    else:
        run_live(args.user, args.tag, args.sample, args.ids)


if __name__ == "__main__":
    main()
