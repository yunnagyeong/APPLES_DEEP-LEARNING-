"""12·16번 본 실행: D-029의 두 부분을 한 스크립트로 돌린다.

[sentences] error_cases 32문장 = 의미 보존 비교 (조건 A·B + v2 검증, C는 scripts/run_condition_c.py)
  - A 일반 번역(zero-shot), B 단순 개인화(RAG 예시만): src.baselines
  - 두 조건 모두 v2 검증(src.verify.verify_translation)으로 판정·Meaning Score를 기록
  - C는 `python scripts/run_condition_c.py --user <u> --out results/v2` (같은 32문장 team_evaluation.jsonl)
  - 입력은 코퍼스 밖 문장이라 홀드아웃 불필요. 해석 주의(D-029): 32문장의 B·C 차이는 개인화가 아니라
    주로 검증·재번역 효과다.

[paragraphs] 각자 글 문단 = 개인화 비교 (생성만, 검증 호출 없음)
  - 입력: 본인 홀드아웃 문서의 문단(data/measure/holdout_docs.json), 지시 대상 2/3 + 대조군 1/3
    (compare_structure_rules.py --holdout과 같은 문단 선택 함수 → 같은 문단)
  - B  RAG만       : src.baselines.translate_in_context (프로필 지시 없음)
  - C  프로필+RAG  : generate.translate_with_style
  - R  프로필만(RAG 없음): translate_with_style(examples=[]) → 12번 "RAG 없음" 조건
  - RAG는 본인 홀드아웃 문서를 제외하고 검색
  - 측정(규칙 기반): 문장 수 비율, 영어 접속 표현 수·목록, 기능어 지시 표현 수. 사람이 "원문 대응(사람 표시)" 열에
    영어 접속 표현마다 O/X를 목록 순서대로 적는다(28번 방식). AI는 번역을 평가하지 않는다.

사용법 (프로젝트 루트, 맥):
  .venv/bin/python scripts/run_12_16.py paragraphs --user suyoung --paras 6 --n 2
  .venv/bin/python scripts/run_12_16.py sentences  --user ganghee
  .venv/bin/python scripts/run_12_16.py paragraphs --user suyoung --dry   # API 없이 문단·프롬프트 확인
실행 시점 git 커밋 해시가 결과 CSV와 콘솔에 기록된다(D-007 ①).
호출 수: paragraphs = 문단당 임베딩 1 + 생성 3×n. sentences = 문장당 생성 2 + 판정 2 + 임베딩 3(B 검색 1 + 유사도).
"""
import argparse
import csv
import json
import statistics as st
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_structure_rules import (  # noqa: E402
    EN_CONN, FW_MARK, GROUP_EN, en_sentences, pick_paragraphs, user_groups,
)

OUT_DIR = ROOT / "data" / "measure"


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "scripts", "data/style_reference.json",
                                         "data/measure/holdout_docs.json"], cwd=ROOT, text=True).strip()
        return h + (" (+미커밋 변경 있음)" if dirty else "")
    except Exception:
        return "unknown"


def install_fake_client():
    """--dry: 생성 호출을 가짜로 바꾸고 프롬프트를 모은다."""
    from src import baselines, generate
    prompts = []
    fake = types.SimpleNamespace(models=types.SimpleNamespace(
        generate_content=lambda model, contents: (prompts.append(contents), types.SimpleNamespace(text="A. B."))[1]))
    generate._ensure_client = lambda: fake
    baselines._ensure_client = lambda: fake
    return prompts


# ---------------------------------------------------------------- paragraphs
def run_paragraphs(a):
    from src import config
    from src import generate as gen
    from src.generate import _style_profile_to_instructions, FUNCTION_WORD_INSTRUCTIONS, translate_with_style
    from src.baselines import translate_in_context
    from src.retrieve import retrieve_similar

    holdout = json.load(open(OUT_DIR / "holdout_docs.json", encoding="utf-8"))["holdout"]
    excl = set(holdout[a.user])
    profile = json.load(open(config.style_profile_cache_path(a.user), encoding="utf-8"))
    groups = user_groups(_style_profile_to_instructions(profile, a.user), gen)
    paras = pick_paragraphs([a.user], a.paras, groups, {a.user: excl})
    print(f"실행 커밋: {git_hash()}")
    print(f"홀드아웃 문서 {sorted(excl)}, 문단 {len(paras)}개, 지시 그룹 {groups}, 역할 {[p['role'] for p in paras]}")

    prompts = install_fake_client() if a.dry else None
    applied = []
    for p in paras:
        ins = _style_profile_to_instructions(profile, a.user, source_text=p["ko"])
        applied.append(sum(1 for line in FUNCTION_WORD_INSTRUCTIONS.values() if line in ins))
    print("문단별 적용 기능어 줄 수:", list(zip([p["role"] for p in paras], applied)))

    conds = ("B RAG만", "C 프로필+RAG", "R 프로필만(RAG 없음)")
    rows = []
    for i, p in enumerate(paras, 1):
        ex = [] if a.dry else retrieve_similar(p["ko"], a.user, k=config.DEFAULT_TOP_K, exclude_doc_ids=excl)
        for label in conds:
            for k in range(1, a.n + 1):
                if label.startswith("B"):
                    en = translate_in_context(p["ko"], a.user, examples=ex).strip()
                elif label.startswith("C"):
                    en = translate_with_style(p["ko"], profile, a.user, examples=ex).strip()
                else:
                    en = translate_with_style(p["ko"], profile, a.user, examples=[]).strip()
                ens = len(en_sentences(en))
                found = [m.group(0).lower() for m in EN_CONN.finditer(en)]
                gmark = sum(len(GROUP_EN[g].findall(en)) for g in groups)
                rows.append({
                    "문단": i, "출처문서": p["doc"], "조건": label, "샘플": k,
                    "한국어 문장 수": p["ko_n"], "영어 문장 수": ens, "문장 수 비율": round(ens / p["ko_n"], 2),
                    "영어 접속 표현 수": len(found), "영어 접속 표현 목록": ", ".join(found),
                    "한국어 접속·연결어 목록": ", ".join(p["ko_found"]),
                    "원문 대응(사람 표시)": "",
                    "기능어 지시 표현 수": len(FW_MARK.findall(en)),
                    "문단 역할": p["role"], "원문 지시그룹 수": p["ko_target"], "영어 지시그룹 표지 수": gmark,
                    "적용 기능어 줄 수": applied[i - 1] if not label.startswith("B") else 0,
                    "실행 커밋": git_hash(), "한국어 원문": p["ko"], "번역": en,
                })
        if not a.dry:
            time.sleep(1)

    if a.dry:
        import difflib
        n = a.n
        print("\n=== B→C 프롬프트 차이 ===")
        for l in difflib.unified_diff(prompts[0].splitlines(), prompts[n].splitlines(), "B", "C", lineterm="", n=0):
            print(l[:160])
        print("\n=== C→R(RAG 없음) 프롬프트 차이 ===")
        for l in difflib.unified_diff(prompts[n].splitlines(), prompts[2 * n].splitlines(), "C", "R", lineterm="", n=0):
            print(l[:160])
        return

    out = OUT_DIR / f"personalization_{a.user}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n조건별 평균 (문단 단위로 먼저 평균)")
    for label in conds:
        rs = [r for r in rows if r["조건"] == label]
        by = lambda key: [st.mean(r[key] for r in rs if r["문단"] == i) for i in range(1, len(paras) + 1)]
        ratios = by("문장 수 비율")
        print(f"  {label:18s} 문장 수 비율 {st.mean(ratios):.2f} (|비율-1| {st.mean(abs(x - 1) for x in ratios):.2f})"
              f"  영어 접속 표현 {st.mean(by('영어 접속 표현 수')):.1f}  기능어 지시 표현 {st.mean(by('기능어 지시 표현 수')):.1f}")
    if groups:
        print(f"\n지시 그룹 {groups}: 지시 대상 문단 영어 표지 평균 / 대조군 문단 평균")
        for label in conds:
            rs = [r for r in rows if r["조건"] == label]

            def avg(role):
                xs = [st.mean(r["영어 지시그룹 표지 수"] for r in rs if r["문단"] == i)
                      for i in range(1, len(paras) + 1) if paras[i - 1]["role"] == role]
                return st.mean(xs) if xs else float("nan")
            print(f"  {label:18s} {avg('지시 대상'):.2f} / {avg('대조군'):.2f}")
    print(f"\n저장: {out}")


# ----------------------------------------------------------------- sentences
def run_sentences(a):
    from src import config
    from src.baselines import translate_in_context, translate_zero_shot
    from src.retrieve import retrieve_similar
    from src.verify import verify_translation

    cases = [json.loads(l) for l in open(ROOT / a.input, encoding="utf-8-sig") if l.strip()]
    print(f"실행 커밋: {git_hash()}, user={a.user}, 문장 {len(cases)}개")
    if a.limit:
        cases = cases[:a.limit]
    rows = []
    out = OUT_DIR / f"baselines_ab_{a.user}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    done = {}
    if a.resume:
        out = ROOT / a.resume
        for r in csv.DictReader(open(out, encoding="utf-8-sig")):
            done[(str(r["id"]), r["condition"])] = r
        print(f"이어하기: 성공한 {sum(1 for r in done.values() if r['verdict'] != 'ERROR')}건 유지, 나머지 다시 실행")
    seen = set()
    quota_fail = 0  # 연속 429 횟수. 3번 연속이면 쿼터 소진으로 보고 멈춤(남은 문장은 --resume으로 이어서)

    def save():
        """처리한 행 + 아직 처리 안 한 이전 행(성공·실패 모두)을 함께 저장 → 중간에 끊겨도 이전 결과가 사라지지 않음."""
        rest = [r for k, r in done.items() if k not in seen]
        allrows = rows + rest
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0]))
            w.writeheader()
            w.writerows(allrows)

    for n, c in enumerate(cases, 1):
        src = c["source"]
        if a.resume and all(done.get((str(c["id"]), l), {}).get("verdict", "ERROR") != "ERROR" for l in ("A", "B")):
            for l in ("A", "B"):
                rows.append(done[(str(c["id"]), l)])
                seen.add((str(c["id"]), l))
            continue
        ex = retrieve_similar(src, a.user, k=config.DEFAULT_TOP_K)
        for label, fn in (("A", lambda: translate_zero_shot(src)), ("B", lambda: translate_in_context(src, a.user, examples=ex))):
            if done.get((str(c["id"]), label), {}).get("verdict", "ERROR") != "ERROR":
                rows.append(done[(str(c["id"]), label)])
                seen.add((str(c["id"]), label))
                continue
            t0 = time.perf_counter()
            row = {"id": c["id"], "condition": label, "source": src, "translation": "", "verdict": "ERROR",
                   "meaning_score": "", "meaning_score_method": "", "issues": "", "judge_model": "",
                   "seconds": "", "error": "", "실행 커밋": git_hash()}
            try:
                en = fn().strip()
                v = verify_translation(src, en)
                row.update(translation=en, verdict=v["verdict"], meaning_score=v.get("meaning_score"),
                           meaning_score_method=v.get("score_method"),
                           issues="; ".join(f"{i['severity']}:{i['type']}" for i in v.get("issues", [])),
                           judge_model=v.get("judge_model"))
            except Exception as exc:  # 한 문장이 실패해도 끝난 결과는 보존
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["seconds"] = round(time.perf_counter() - t0, 1)
            rows.append(row)
            seen.add((str(c["id"]), label))
            print(f"[{label}] {n}/{len(cases)} #{c['id']} {row['verdict']} score={row['meaning_score']} {row['seconds']}초"
                  + (f" | {row['error']}" if row["error"] else ""))
            save()
            quota_fail = quota_fail + 1 if "RESOURCE_EXHAUSTED" in row["error"] or "429" in row["error"] else 0
            if quota_fail >= 3:
                print("\n쿼터 소진(429 연속 3회)으로 중단합니다. 쿼터가 복구되면 --resume으로 이어서 실행하세요.")
                break
        if quota_fail >= 3:
            break
    save()
    for label in ("A", "B"):
        rs = [r for r in rows if r["condition"] == label]
        sc = [float(r["meaning_score"]) for r in rs if r["meaning_score"] not in ("", None)]
        vc = {v: sum(1 for r in rs if r["verdict"] == v) for v in ("PASS", "REVIEW", "FAIL", "ERROR")}
        print(f"  {label}: n={len(rs)} 평균 Meaning Score {st.mean(sc) if sc else float('nan'):.1f} {vc}")
    print(f"저장: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("paragraphs")
    p.add_argument("--user", required=True, choices=["ganghee", "nakyung", "suyoung"])
    p.add_argument("--paras", type=int, default=6)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--dry", action="store_true")
    s = sub.add_parser("sentences")
    s.add_argument("--user", default="ganghee", choices=["ganghee", "nakyung", "suyoung"])
    s.add_argument("--input", default="data/measure/team_evaluation.jsonl")
    s.add_argument("--limit", type=int, default=0, help="앞에서 N문장만(시험용)")
    s.add_argument("--resume", default="", help="이전 결과 CSV 경로. 이미 성공(ERROR 아님)한 (id, 조건)은 건너뛰고 같은 파일에 이어서 저장")
    a = ap.parse_args()
    run_paragraphs(a) if a.mode == "paragraphs" else run_sentences(a)


if __name__ == "__main__":
    main()
