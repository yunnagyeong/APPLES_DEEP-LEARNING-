"""의미 검증 v2(src/verify.py)를 사람 라벨(error_cases.csv 32건)과 비교합니다.

사용법 (프로젝트 루트):
  python scripts/eval_verifier.py                 # 남은 사례를 동시에 4건씩 평가하고 요약 출력 (32건 약 1~2분)
  python scripts/eval_verifier.py --workers 8     # 더 많이 동시에
  python scripts/eval_verifier.py --limit 10      # 이번 실행에서 최대 10건만
  python scripts/eval_verifier.py --report        # API 호출 없이 저장된 결과로 요약만
  python scripts/eval_verifier.py --model gemini-3.5-flash-lite --out data/measure/verifier_v2_lite.jsonl

- 사례당 판정 모델 1회 + 임베딩 1회. 결과는 끝나는 대로 한 줄씩 저장되어, 중간에 끊겨도 다시 실행하면 이어서 함.
- 같은 32건으로 레거시(7주차) 검증의 결과(data/measure/meaning_evaluation.jsonl + nuance_evaluation.jsonl)와
  나란히 비교합니다.
- 주의: v2 판정 기준(유형·심각도·점수 가중치)을 이 32건을 보고 설계했으므로 여기 숫자는 낙관적입니다.
  새 문장으로 확인하려면 --cases에 같은 형식(번호·원문·번역·심각도 열)의 다른 CSV를 주세요.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEV_ORDER = ["정상", "하", "중", "상"]


def load_jsonl(path):
    if not path.exists():
        return {}
    return {str(json.loads(l)["id"]): json.loads(l) for l in open(path, encoding="utf-8") if l.strip()}


def auc(normal, errors):
    """오류 사례 점수가 정상 사례보다 낮은 쌍의 비율(동점 0.5)."""
    if not normal or not errors:
        return None
    return sum((e < n) + 0.5 * (e == n) for n in normal for e in errors) / (len(normal) * len(errors))


def report(cases, done):
    rows = [(c, done[c["번호"]]) for c in cases if c["번호"] in done]
    if not rows:
        print("저장된 결과가 없습니다.")
        return
    sev = lambda c: c["심각도"] or "정상"  # noqa: E731
    errs = [(c, r) for c, r in rows if sev(c) != "정상"]
    oks = [(c, r) for c, r in rows if sev(c) == "정상"]

    print(f"\n=== v2 검증 결과 ({len(rows)}/{len(cases)}건, 오류 {len(errs)} · 정상 {len(oks)}) ===")
    print(f"{'#':>3} {'사람':4} {'판정':6} {'점수':>5}  문제")
    for c, r in rows:
        probs = ", ".join(f"{i['severity'][:3]}:{i['type']}" for i in r["issues"]) or "-"
        print(f"{c['번호']:>3} {sev(c):4} {r['verdict']:6} {r['meaning_score']:5.0f}  {probs}")

    flagged = lambda r: r["verdict"] != "PASS"  # noqa: E731
    any_issue = lambda r: bool(r["issues"])  # noqa: E731
    print("\n[v2] 오류 탐지(REVIEW·FAIL):", f"{sum(flagged(r) for _, r in errs)}/{len(errs)}",
          "| 정상 오탐:", f"{sum(flagged(r) for _, r in oks)}/{len(oks)}")
    print("[v2] minor까지 포함한 지적:", f"{sum(any_issue(r) for _, r in errs)}/{len(errs)}",
          "| 정상에 지적:", f"{sum(any_issue(r) for _, r in oks)}/{len(oks)}")
    for s in SEV_ORDER:
        sc = [r["meaning_score"] for c, r in rows if sev(c) == s]
        if sc:
            vd = {v: sum(1 for c, r in rows if sev(c) == s and r["verdict"] == v) for v in ("PASS", "REVIEW", "FAIL")}
            print(f"  {s:3}: n={len(sc):2d} 평균 점수 {sum(sc)/len(sc):5.1f}  판정 {vd}")
    a = auc([r["meaning_score"] for _, r in oks], [r["meaning_score"] for _, r in errs])
    if a is not None:
        print(f"[v2] 점수 분리도(AUC, 오류<정상): {a:.3f}")

    # 레거시 비교
    legacy = load_jsonl(ROOT / "data/measure/meaning_evaluation.jsonl")
    nuance = load_jsonl(ROOT / "data/measure/nuance_evaluation.jsonl")
    ids = [c["번호"] for c, _ in rows if c["번호"] in legacy and c["번호"] in nuance]
    if ids:
        by = {c["번호"]: c for c, _ in rows}
        le = [i for i in ids if sev(by[i]) != "정상"]
        lo = [i for i in ids if sev(by[i]) == "정상"]
        lf = lambda i: legacy[i]["verification"]["verdict"] != "PASS" or nuance[i]["nuance"]["has_meaning_error"]  # noqa: E731
        print(f"[레거시 NLI+뉘앙스, 같은 {len(ids)}건] 오류 탐지 {sum(map(lf, le))}/{len(le)} | 정상 오탐 {sum(map(lf, lo))}/{len(lo)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "error_cases.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/measure/verifier_v2_eval.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4, help="동시에 평가할 사례 수")
    ap.add_argument("--model", default=None, help="판정 모델 바꾸기(기본 config.VERIFY_MODEL)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    cases = list(csv.DictReader(open(a.cases, encoding="utf-8-sig")))
    out = Path(a.out)
    done = load_jsonl(out)
    if not a.report:
        from src import config
        from src.verify import verify_translation
        if a.model:
            config.VERIFY_MODEL = a.model
        todo = [c for c in cases if c["번호"] not in done][: a.limit]
        print(f"평가 {len(todo)}건 (이미 {len(done)}건 완료)")
        out.parent.mkdir(parents=True, exist_ok=True)
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        lock = threading.Lock()

        def _one(c):
            return c, verify_translation(c["원문"], c["번역"])

        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
            futures = [pool.submit(_one, c) for c in todo]
            for fut in as_completed(futures):
                try:
                    c, v = fut.result()
                except Exception as e:
                    print(f"실패(다시 실행하면 이 사례부터 이어서 함): {e}")
                    continue
                with lock:
                    _save(out, c, v, done)

    report(cases, done)


def _save(out, c, v, done):
    rec = {"id": c["번호"], "human_severity": c["심각도"], "human_error": c["문제유형"],
           "verdict": v["verdict"], "meaning_score": v["meaning_score"], "issues": v["issues"],
           "source_claims": v["source_claims"], "additions": v["additions"],
           "semantic_similarity": v["semantic_similarity"], "judge_model": v["judge_model"]}
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    done[c["번호"]] = rec
    print(f"#{c['번호']:>2} 사람={c['심각도'] or '정상':2} → {v['verdict']:6} {v['meaning_score']:5.0f} [{v['judge_model']}]")


if __name__ == "__main__":
    main()
