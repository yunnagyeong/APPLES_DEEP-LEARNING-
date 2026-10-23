"""23번·D-006 변경(상대 기준 지시문 + 문장 경계·접속 표현 보존 규칙)의 효과를 문장 수 비율로 비교한다.

조건 3개를 같은 입력·같은 RAG 예시로 번역한다(번역 생성만 호출, claim 추출·NLI 없음):
  A  전      : git <base>(기본 HEAD)의 src/generate.py (절대 임계값 지시문, 공통 규칙 없음)
  B  d만     : 작업 폴더의 generate.py, 문장 경계·접속 표현 규칙만 뺌 (지시문 변화만 반영)
  C  d+규칙  : 작업 폴더 그대로
측정(규칙 기반, API·LLM 판단 없음):
  문장 수 비율 = 영어 문장 수 / 한국어 문장 수 (1에서 멀면 문장 경계가 바뀐 것)
  접속 표현 수: 한국어 원문의 LOGICAL_CONNECTIVES 개수 vs 영어 번역의 접속 단어 개수(참고값)
입력: --from 사용자의 raw 문단. 번역 대상 사용자(--user)의 RAG 풀에 없는 글이어야 같은 문장이
예시로 되돌아오는 일이 없으므로, 기본은 다른 사람의 문서에서 고른다.

사용법: python scripts/compare_structure_rules.py --user suyoung --from ganghee --paras 6 --n 2
        python scripts/compare_structure_rules.py --dry   # API 없이 문단 선택·프롬프트 확인
결과: data/measure/structure_compare_<user>_<시각>.csv
호출 수: 문단당 임베딩 1 + 생성 3×n
"""
import argparse, csv, json, re, statistics as st, sys, time, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_prompt_change import load_base_generate  # noqa: E402

EN_CONN = re.compile(
    r"\b(however|therefore|thus|hence|moreover|furthermore|in addition|as a result|consequently|"
    r"in contrast|on the other hand|whereas|nevertheless|but|because|so that|that is|in other words)\b",
    re.IGNORECASE,
)
_ABBR = re.compile(r"\b(e\.g|i\.e|etc|vs|et al|Fig|No|Dr|Mr|Ms)\.$")


def en_sentences(text: str) -> list[str]:
    """영어 문장 분리(규칙 기반). 약어 뒤 마침표는 경계로 보지 않는다."""
    parts, buf = [], ""
    for tok in re.split(r"(?<=[.!?])\s+", text.strip()):
        buf = f"{buf} {tok}".strip()
        if _ABBR.search(buf) or not re.search(r"[.!?][\"')\]]*$", buf):
            continue
        parts.append(buf)
        buf = ""
    if buf:
        parts.append(buf)
    return parts


def pick_paragraphs(src_user: str, n: int) -> list[dict]:
    """4~6문장·150~450자·접속어 1개 이상인 문단(덩어리)을 고른다.

    수영·나경 raw는 문서 전체가 한 문단이라(문단 구분 없음, D-003) 문단이 7문장 이상이면
    연속한 5문장씩 잘라 쓴다. 마지막 조각이 4문장 미만이면 버린다.
    """
    from src import style_features as sf
    out = []
    for line in open(ROOT / f"data/raw/corpus_{src_user}_raw.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d.get("type") not in ("essay", "report"):
            continue
        for para in sf.split_paragraphs(d["text"]):
            sents = sf.split_sentences(para)
            chunks = [sents] if len(sents) <= 6 else [sents[i:i + 5] for i in range(0, len(sents), 5)]
            for ch in chunks:
                text = " ".join(ch)
                nconn = sum(text.count(c) for c in sf.LOGICAL_CONNECTIVES)
                if 4 <= len(ch) <= 6 and 150 <= len(text) <= 450 and nconn >= 1:
                    out.append({"doc": d["id"], "ko": text, "ko_n": len(ch), "ko_conn": nconn})
    if not out:
        sys.exit(f"{src_user}에서 조건에 맞는 문단을 찾지 못했습니다.")
    step = max(1, len(out) // n)
    return out[::step][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="suyoung")
    ap.add_argument("--from", dest="src", default="ganghee")
    ap.add_argument("--paras", type=int, default=6)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--base", default="HEAD")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    from src import config, generate as new
    from src.retrieve import retrieve_similar

    old = load_base_generate(a.base)
    profile = json.load(open(config.style_profile_cache_path(a.user), encoding="utf-8"))
    paras = pick_paragraphs(a.src, a.paras)
    print(f"문단 {len(paras)}개 선택 (출처 {a.src}, 문장 수 {[p['ko_n'] for p in paras]})")
    rules = new.STRUCTURE_PRESERVATION_RULES

    prompts = []
    if a.dry:
        fake = types.SimpleNamespace(models=types.SimpleNamespace(
            generate_content=lambda model, contents: (prompts.append(contents), types.SimpleNamespace(text="A. B."))[1]))
        old._ensure_client = lambda: fake
        new._ensure_client = lambda: fake

    rows = []
    for i, p in enumerate(paras, 1):
        ex = [] if a.dry else retrieve_similar(p["ko"], a.user, k=config.DEFAULT_TOP_K)
        for label, mod, r in (("A 전", old, None), ("B d만", new, ()), ("C d+규칙", new, rules)):
            if r is not None:
                new.STRUCTURE_PRESERVATION_RULES = r
            for k in range(1, a.n + 1):
                en = mod.translate_with_style(p["ko"], profile, a.user, examples=ex).strip()
                ens = len(en_sentences(en))
                rows.append({"문단": i, "출처문서": p["doc"], "조건": label, "샘플": k,
                             "한국어 문장 수": p["ko_n"], "영어 문장 수": ens,
                             "문장 수 비율": round(ens / p["ko_n"], 2),
                             "한국어 접속어 수": p["ko_conn"], "영어 접속 표현 수": len(EN_CONN.findall(en)),
                             "번역": en})
            new.STRUCTURE_PRESERVATION_RULES = rules
        if not a.dry:
            time.sleep(1)

    if a.dry:
        import difflib
        print("\n=== A→C 프롬프트 차이 ===")
        for l in difflib.unified_diff(prompts[0].splitlines(), prompts[2 * a.n].splitlines(), "A", "C", lineterm="", n=0):
            print(l)
        print("\n=== B→C 프롬프트 차이 ===")
        for l in difflib.unified_diff(prompts[a.n].splitlines(), prompts[2 * a.n].splitlines(), "B", "C", lineterm="", n=0):
            print(l)
        return

    out = ROOT / "data" / "measure" / f"structure_compare_{a.user}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("\n조건별 평균 (문단 단위로 먼저 평균)")
    for label in ("A 전", "B d만", "C d+규칙"):
        rs = [r for r in rows if r["조건"] == label]
        ratios = [st.mean(r["문장 수 비율"] for r in rs if r["문단"] == i) for i in range(1, len(paras) + 1)]
        conn = [st.mean(r["영어 접속 표현 수"] for r in rs if r["문단"] == i) for i in range(1, len(paras) + 1)]
        dev = [abs(x - 1) for x in ratios]
        print(f"  {label:9s} 문장 수 비율 평균 {st.mean(ratios):.2f}  |비율-1| 평균 {st.mean(dev):.2f}  영어 접속 표현 평균 {st.mean(conn):.1f}"
              f"  (한국어 접속어 평균 {st.mean(p['ko_conn'] for p in paras):.1f})")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
