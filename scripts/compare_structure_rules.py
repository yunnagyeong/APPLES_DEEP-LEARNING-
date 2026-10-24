"""23번·D-006 변경(상대 기준 지시문 + 문장 경계·접속 표현 보존 규칙)의 효과를 문장 수 비율로 비교한다.

조건 3개를 같은 입력·같은 RAG 예시로 번역한다(번역 생성만 호출, claim 추출·NLI 없음):
  A  전      : git <base>(기본 HEAD)의 src/generate.py (절대 임계값 지시문, 공통 규칙 없음)
  B  d만     : 작업 폴더의 generate.py, 문장 경계·접속 표현 규칙만 뺌 (지시문 변화만 반영)
  C  d+규칙  : 작업 폴더 그대로
  D  기능어 뺌: C에서 기능어 선택 지시(24번)만 뺌 (C와 D 차이 = 기능어 지시 효과)
측정(규칙 기반, API·LLM 판단 없음):
  문장 수 비율 = 영어 문장 수 / 한국어 문장 수 (1에서 멀면 문장 경계가 바뀐 것)
  접속 표현 수: 한국어 원문의 LOGICAL_CONNECTIVES 개수 vs 영어 번역의 접속 단어 개수(참고값)
입력: --from 사용자의 raw 문단. 번역 대상 사용자(--user)의 RAG 풀에 없는 글이어야 같은 문장이
예시로 되돌아오는 일이 없으므로, 기본은 다른 사람의 문서에서 고른다.

사용법: python scripts/compare_structure_rules.py --user suyoung --from ganghee --paras 6 --n 2
        python scripts/compare_structure_rules.py --dry   # API 없이 문단 선택·프롬프트 확인
결과: data/measure/structure_compare_<user>_<시각>.csv
호출 수: 문단당 임베딩 1 + 생성 4×n (문단 6 × n 2 = 임베딩 6 + 생성 48)
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
# 기능어 선택 지시(however/but/whereas/therefore/causal/purpose)가 겨냥하는 영어 표현 수(참고값)
FW_MARK = re.compile(r"\b(however|but|whereas|on the other hand|therefore|thus|since|because|in order to)\b", re.IGNORECASE)
# 접속부사 말고 연결어미도 원문 대응 확인에 쓰도록 목록으로 남김(자동 탐지, 참고값)
KO_ENDINGS = ("므로", "어서", "아서", "지만", "으나", "는데", "고자", "면서", "으며", "도록", "때문에")
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


GROUP_EN = {  # 기능어 그룹 → 영어 표지(참고값, 규칙 기반)
    "however": re.compile(r"\bhowever\b", re.I), "but": re.compile(r"\bbut\b", re.I),
    "whereas": re.compile(r"\b(whereas|on the other hand)\b", re.I),
    "therefore": re.compile(r"\b(therefore|thus|hence)\b", re.I),
    "causal": re.compile(r"\b(since|because)\b", re.I), "purpose_formal": re.compile(r"\bin order to\b", re.I),
}


def user_groups(instr_text: str, new) -> list[str]:
    """지시문에 들어간 기능어 그룹(FUNCTION_WORD_INSTRUCTIONS 문구가 있는 그룹)."""
    return [g for g, line in new.FUNCTION_WORD_INSTRUCTIONS.items() if line in instr_text]


def pick_paragraphs(src_users: list[str], n: int, groups: list[str] | None = None) -> list[dict]:
    """4~6문장·150~450자 문단(덩어리)을 고른다.

    groups가 주어지면(번역 대상 사용자의 기능어 지시 그룹) 원문에 그 그룹 표현이 있는 문단을
    약 2/3(`지시 대상`)으로, 없는 문단을 나머지(`대조군`: 원문에 없는 표지를 새로 넣는지 확인)로 고른다.
    groups가 없으면 예전처럼 접속어 1개 이상 문단을 고른다.
    수영·나경 raw는 문서 전체가 한 문단이라(문단 구분 없음, D-003) 7문장 이상이면 5문장씩 잘라 쓴다.
    """
    from src import style_features as sf
    kiwi = sf._get_kiwi()
    cands = []
    for src_user in src_users:
        for line in open(ROOT / f"data/raw/corpus_{src_user}_raw.jsonl", encoding="utf-8"):
            d = json.loads(line)
            if d.get("type") not in ("essay", "report"):
                continue
            for para in sf.split_paragraphs(d["text"]):
                sents = sf.split_sentences(para)
                chunks = [sents] if len(sents) <= 6 else [sents[i:i + 5] for i in range(0, len(sents), 5)]
                for ch in chunks:
                    text = " ".join(ch)
                    if not (4 <= len(ch) <= 6 and 150 <= len(text) <= 450):
                        continue
                    nconn = sum(text.count(c) for c in sf.LOGICAL_CONNECTIVES)
                    toks = kiwi.tokenize(text)
                    cnt = sf.count_function_word_groups(toks)
                    gcnt = sf.count_gate_groups(toks)  # 지시문 적용 조건(D-013)과 같은 기준으로 역할을 나눔
                    found = [c for c in sf.LOGICAL_CONNECTIVES for _ in range(text.count(c))] + \
                            [e for e in KO_ENDINGS for _ in range(text.count(e))]
                    cands.append({"doc": d["id"], "src": src_user, "ko": text, "ko_n": len(ch), "ko_conn": nconn,
                                  "ko_found": found, "ko_fw": {k: v for k, v in cnt.items() if v},
                                  "ko_target": sum(gcnt[g] for g in (groups or []))})
    if groups:
        hits = [c for c in cands if c["ko_target"] >= 1]
        nones = [c for c in cands if c["ko_target"] == 0]
        n_hit = round(n * 2 / 3)
        out = _spread(hits, n_hit) + _spread(nones, n - n_hit)
        for c in out:
            c["role"] = "지시 대상" if c["ko_target"] >= 1 else "대조군"
    else:
        out = _spread([c for c in cands if c["ko_conn"] >= 1], n)
        for c in out:
            c["role"] = "-"
    if not out:
        sys.exit(f"{src_users}에서 조건에 맞는 문단을 찾지 못했습니다.")
    return out


def _spread(items, k):
    """출처가 섞이도록 시작 위치를 주고 균등 간격으로 k개 고른다."""
    if k <= 0 or not items:
        return []
    step = max(1, len(items) // k)
    return items[step // 2::step][:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="suyoung")
    ap.add_argument("--from", dest="src", default="auto", help="쉼표로 여러 명. auto = 번역 대상이 아닌 두 사람")
    ap.add_argument("--paras", type=int, default=6)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--base", default="HEAD")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    from src import config, generate as new
    from src.retrieve import retrieve_similar

    old = load_base_generate(a.base)
    profile = json.load(open(config.style_profile_cache_path(a.user), encoding="utf-8"))
    srcs = [x for x in ("ganghee", "nakyung", "suyoung") if x != a.user] if a.src == "auto" else a.src.split(",")
    groups = user_groups(new._style_profile_to_instructions(profile, a.user), new)
    paras = pick_paragraphs(srcs, a.paras, groups)
    print(f"문단 {len(paras)}개 선택 (출처 {srcs}, 지시 그룹 {groups}, 문장 수 {[p['ko_n'] for p in paras]}, "
          f"역할 {[p['role'] for p in paras]})")
    rules = new.STRUCTURE_PRESERVATION_RULES
    fw_default = new.FUNCTION_WORD_MAX_LINES

    prompts = []
    if a.dry:
        fake = types.SimpleNamespace(models=types.SimpleNamespace(
            generate_content=lambda model, contents: (prompts.append(contents), types.SimpleNamespace(text="A. B."))[1]))
        old._ensure_client = lambda: fake
        new._ensure_client = lambda: fake

    # 문단마다 실제로 들어가는 기능어 줄 수(입력에 표현이 있을 때만 들어감, D-013). 대조군은 0이어야 함
    applied = []
    for p in paras:
        ins = new._style_profile_to_instructions(profile, a.user, source_text=p["ko"])
        applied.append(sum(1 for line in new.FUNCTION_WORD_INSTRUCTIONS.values() if line in ins))
    print("문단별 적용 기능어 줄 수:", list(zip([p["role"] for p in paras], applied)))

    rows = []
    for i, p in enumerate(paras, 1):
        ex = [] if a.dry else retrieve_similar(p["ko"], a.user, k=config.DEFAULT_TOP_K)
        for label, mod, r, fwmax in (("A 전", old, None, None), ("B d만", new, (), None),
                                     ("C d+규칙", new, rules, None), ("D 기능어 뺌", new, rules, 0)):
            if r is not None:
                new.STRUCTURE_PRESERVATION_RULES = r
            new.FUNCTION_WORD_MAX_LINES = fw_default if fwmax is None else fwmax
            for k in range(1, a.n + 1):
                en = mod.translate_with_style(p["ko"], profile, a.user, examples=ex).strip()
                ens = len(en_sentences(en))
                found = [m.group(0).lower() for m in EN_CONN.finditer(en)]
                fwn = len(FW_MARK.findall(en))
                gmark = sum(len(GROUP_EN[g].findall(en)) for g in groups)
                rows.append({"문단": i, "출처문서": p["doc"], "조건": label, "샘플": k,
                             "한국어 문장 수": p["ko_n"], "영어 문장 수": ens,
                             "문장 수 비율": round(ens / p["ko_n"], 2),
                             "한국어 접속어 수": p["ko_conn"], "영어 접속 표현 수": len(found),
                             "영어 접속 표현 목록": ", ".join(found),
                             "한국어 접속·연결어 목록": ", ".join(p["ko_found"]),
                             "원문 대응(사람 표시)": "",  # 영어 접속 표현마다 원문 대응 O/X를 목록 순서대로 적기
                             "기능어 지시 표현 수": fwn,
                             "문단 역할": p["role"], "출처": p["src"],
                             "원문 기능어 그룹(자동)": ", ".join(f"{k}:{v}" for k, v in p["ko_fw"].items()),
                             "원문 지시그룹 수": p["ko_target"], "영어 지시그룹 표지 수": gmark,
                             "적용 기능어 줄 수": applied[i - 1],
                             "한국어 원문": p["ko"],
                             "번역": en})
            new.STRUCTURE_PRESERVATION_RULES = rules
            new.FUNCTION_WORD_MAX_LINES = fw_default
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
    for label in ("A 전", "B d만", "C d+규칙", "D 기능어 뺌"):
        rs = [r for r in rows if r["조건"] == label]
        ratios = [st.mean(r["문장 수 비율"] for r in rs if r["문단"] == i) for i in range(1, len(paras) + 1)]
        conn = [st.mean(r["영어 접속 표현 수"] for r in rs if r["문단"] == i) for i in range(1, len(paras) + 1)]
        fwm = [st.mean(r["기능어 지시 표현 수"] for r in rs if r["문단"] == i) for i in range(1, len(paras) + 1)]
        dev = [abs(x - 1) for x in ratios]
        print(f"  {label:9s} 문장 수 비율 평균 {st.mean(ratios):.2f}  |비율-1| 평균 {st.mean(dev):.2f}  영어 접속 표현 평균 {st.mean(conn):.1f}"
              f"  기능어 지시 표현 평균 {st.mean(fwm):.1f}"
              f"  (한국어 접속어 평균 {st.mean(p['ko_conn'] for p in paras):.1f})")
    print(f"\n지시 그룹 {groups}: 원문에 있는 문단(지시 대상)에서는 영어 표지가 얼마나 나오는가 / 원문에 없는 문단(대조군)에서 새로 나오는가")
    for label in ("A 전", "B d만", "C d+규칙", "D 기능어 뺌"):
        rs = [r for r in rows if r["조건"] == label]
        def avg(role):
            xs = [st.mean(r["영어 지시그룹 표지 수"] for r in rs if r["문단"] == i)
                  for i in range(1, len(paras) + 1) if paras[i - 1]["role"] == role]
            return st.mean(xs) if xs else float("nan")
        print(f"  {label:9s} 지시 대상 문단 영어 표지 평균 {avg('지시 대상'):.2f}  | 대조군 문단 평균 {avg('대조군'):.2f}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
