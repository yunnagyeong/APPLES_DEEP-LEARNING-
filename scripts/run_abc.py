"""A/B/C 비교 실험: 평가 원문 30개를 세 조건으로 번역하고, 셋 모두 같은 검증기(v2)로 채점.

사용법 (프로젝트 루트):
    python scripts/run_abc.py --limit 2     # 먼저 원문 2개로 시험 (API 약 20회)
    python scripts/run_abc.py               # 전체 30개 × A/B/C
    python scripts/run_abc.py --only R      # R(프로필만, 검색 없음) 추가 실행
    python scripts/score_abc.py             # 점수 계산 → results/abc/summary.csv

- 학습용 코퍼스(평가 글을 뺀 것)로 자동 전환합니다. 앱·데모는 영향 없음.
- 중간에 끊겨도 다시 실행하면 끝난 건 건너뛰고 이어서 합니다.
- A·B는 번역 후 검증만 1번(재번역 없음), C는 파이프라인 그대로(검증 + 최대 2회 재번역).
  C가 이기는 게 당연한 '자기채점'이므로, 보고할 때는 A·B·C 모두 같은 검증기로 잰 결과와
  검증기와 무관한 지표(score_abc.py의 상투어·어휘 다양도·지시 반영)를 같이 보세요.
"""
import os

os.environ["LINGUA_CORPUS_SPLIT"] = "train"  # src를 import하기 전에 설정해야 적용됨

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.baselines import translate_in_context, translate_zero_shot  # noqa: E402
from src.pipeline import _load_or_compute_style_profile, translate_with_verification  # noqa: E402
from src.verify import verify_translation  # noqa: E402

SOURCES = ROOT / "data" / "eval" / "eval_sources.jsonl"
OUT_DIR = ROOT / "results" / "abc"
RESULTS = OUT_DIR / "results.jsonl"
_lock = threading.Lock()


def _issues_str(issues):
    return "; ".join(f"{i['severity']}:{i['type']}" for i in issues or [])


def run_one(src: dict, cond: str, profiles: dict, retries: int) -> dict:
    user, text = src["person"], src["text"]
    t0 = time.perf_counter()
    rec = {"sample_id": src["id"], "person": user, "condition": cond, "source": text}
    try:
        if cond in ("C", "R"):
            # R = C에서 비슷한 글 검색(RAG)만 뺀 조건(D-035): examples=[]를 주면 검색을 건너뛰고 프로필 지시문만 씀
            r = translate_with_verification(korean_text=text, style_profile=profiles[user],
                                            user=user, max_retries=retries,
                                            examples=[] if cond == "R" else None)
            rec.update(output=r["translation"], verdict=r["verdict"], meaning_score=r.get("meaning_score"),
                       issues=_issues_str(r.get("issues")), attempts=r.get("attempts"),
                       judge_model=r.get("judge_model"))
        else:
            out = translate_zero_shot(text) if cond == "A" else translate_in_context(text, user)
            v = verify_translation(text, out)
            rec.update(output=out, verdict=v["verdict"], meaning_score=v.get("meaning_score"),
                       issues=_issues_str(v.get("issues")), attempts=1, judge_model=v.get("judge_model"))
        rec["error"] = ""
    except Exception as e:  # 한 건 실패해도 나머지는 계속
        rec.update(output="", verdict="ERROR", error=f"{type(e).__name__}: {e}")
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


def load_done() -> set:
    if not RESULTS.exists():
        return set()
    done = set()
    for line in RESULTS.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("output") and not r.get("error"):
                done.add((r["sample_id"], r["condition"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="사람마다 원문 몇 개만 (시험용)")
    ap.add_argument("--only", nargs="*", choices=["A", "B", "C", "R"], default=["A", "B", "C"],
                    help="R은 기본에 포함 안 됨: --only R 로 따로 실행")
    ap.add_argument("--retries", type=int, default=config.DEFAULT_MAX_RETRIES)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--person", nargs="*", choices=["ganghee", "nakyung", "suyoung"], default=None,
                    help="이 사람의 원문만 (예: --person suyoung)")
    ap.add_argument("--redo", action="store_true",
                    help="이미 끝난 건도 다시 실행(결과는 results.jsonl 뒤에 덧붙고, score_abc.py는 가장 나중 것을 씀)")
    args = ap.parse_args()

    sources = [json.loads(l) for l in SOURCES.open(encoding="utf-8") if l.strip()]
    if args.person:
        sources = [s for s in sources if s["person"] in args.person]
    if args.limit:
        count, kept = {}, []
        for s in sources:
            count[s["person"]] = count.get(s["person"], 0) + 1
            if count[s["person"]] <= args.limit:
                kept.append(s)
        sources = kept
    users = sorted({s["person"] for s in sources})
    print(f"[준비] 학습용 코퍼스 사용: {[config.examples_path(u).name for u in users]}")
    profiles = {u: _load_or_compute_style_profile(u) for u in users}  # 학습용으로 문체 프로필 계산

    done = set() if args.redo else load_done()
    jobs = [(s, c) for s in sources for c in args.only if (s["id"], c) not in done]
    print(f"[준비] 할 일 {len(jobs)}건 (이미 끝난 {len(sources) * len(args.only) - len(jobs)}건 건너뜀)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool, RESULTS.open("a", encoding="utf-8") as f:
        futs = [pool.submit(run_one, s, c, profiles, args.retries) for s, c in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            with _lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            print(f"[{n}/{len(jobs)}] {rec['sample_id']} {rec['condition']} {rec['verdict']} "
                  f"score={rec.get('meaning_score')} ({rec['seconds']}초)" + (f" | {rec['error']}" if rec["error"] else ""))

    print(f"\n완료 → {RESULTS.relative_to(ROOT)}\n다음: python scripts/score_abc.py")


if __name__ == "__main__":
    main()
