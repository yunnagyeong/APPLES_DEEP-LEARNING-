"""위키문헌에서 저작권 만료 작가의 산문을 모아 새 인물 코퍼스(data/raw/corpus_<id>_raw.jsonl)로 만든다.

세 단계로 나눠 중간에 사람이 확인한다.
  1) list  : 작가 페이지에서 작품 목록을 받아 본문 길이를 세고 후보표(CSV)를 만든다
  2) (사람) : 후보표를 열어 keep 칸을 확인·수정한다 (기본: 산문 섹션이면서 2,000자 이상이면 1)
  3) build : keep=1인 위키문헌 글 + data/manual/<pid>/*.txt(다른 사이트에서 복사한 글)를 합쳐 팀 raw 형식으로 저장한다
  +) scan  : 저작권 만료 작가 여러 명의 작가 페이지만 훑어 산문 편수를 센다 (작가 고르기용)
  +) check : 옛 맞춤법 때문에 Kiwi가 어절을 통째로 고유명사(NNP)로 묶는 비율을 팀 글과 비교한다

쓰는 법 (프로젝트 루트, 인터넷 되는 컴퓨터):
  python scripts/collect_wikisource.py list  --author 이광수 --pid leegwangsu
  python scripts/collect_wikisource.py list  --author 채만식 --pid chaemansik
  python scripts/collect_wikisource.py list  --author 이효석 --pid leehyoseok
  #   → data/wikisource/<pid>_candidates.csv 를 엑셀로 열어 keep 칸 확인
  python scripts/collect_wikisource.py check --pid leegwangsu      # 옛 표기 점검 (keep=1 글 기준)
  python scripts/collect_wikisource.py build --pid leegwangsu      # → data/raw/corpus_leegwangsu_raw.jsonl
  #   위키문헌에 글이 모자라면: 한국사데이터베이스(db.history.go.kr) 등에서 글을 복사해
#   data/manual/<pid>/<제목>.txt 로 저장 (첫 줄에 'source: 주소'를 쓰면 출처로 기록) → build 다시 실행
# 그다음은 scripts/ko_new_corpus.py 맨 위 설명 그대로 (prepare → build_index → ko_doc_transfer)

필요: Python 표준 라이브러리만 (check만 kiwipiepy 필요). 위키미디어 규칙에 따라 요청 사이에 쉬고 User-Agent를 붙인다.
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "wikisource"
API = "https://ko.wikisource.org/w/api.php"
UA = "LinguaMe-corpus/0.1 (HUFS student project; educational use)"
MIN_CHARS = 2000
# 작가 페이지의 섹션 제목으로 산문 여부를 고른다 (사람이 CSV에서 최종 확인)
PROSE_HINTS = ("수필", "평론", "논설", "논문", "기행", "산문", "잡문", "감상", "칼럼", "서간", "일기", "에세이", "문학론")
NON_PROSE_HINTS = ("소설", "시", "시조", "희곡", "동화", "동시", "번역", "가사", "노래", "단편", "장편")


# ---------------------------------------------------------------- API
def api(params, retries=6):
    q = dict(params, format="json", formatversion="2", maxlag="5")
    url = API + "?" + urllib.parse.urlencode(q)
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            time.sleep(1.0)  # 서버 배려
            return data
        except urllib.error.HTTPError as e:
            if i == retries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 15 * (i + 1)
            print(f"  [대기] 서버가 잠시 쉬라고 합니다({e.code}). {wait}초 후 다시 시도", file=sys.stderr)
            time.sleep(wait)
        except Exception as e:  # 네트워크 오류는 잠깐 쉬고 다시
            if i == retries - 1:
                raise
            print(f"  [재시도] {e}", file=sys.stderr)
            time.sleep(10 * (i + 1))


def wikitext(title):
    d = api({"action": "parse", "page": title, "prop": "wikitext", "redirects": 1})
    if "error" in d:
        return None
    return d["parse"]["wikitext"]


def cache_path(pid):
    return OUT_DIR / "cache" / f"{pid}.json"


def load_cache(pid):
    p = cache_path(pid)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_cache(pid, cache):
    p = cache_path(pid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def fetch_many(titles, pid, batch=20):
    """본문을 20편씩 한 번에 받는다(요청 수를 줄여 429를 피함). 받은 것은 캐시에 저장해 다시 받지 않는다."""
    cache = load_cache(pid)
    todo = [t for t in titles if t not in cache]
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        d = api({"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                 "titles": "|".join(chunk), "redirects": 1})
        q = d.get("query", {})
        alias = {}
        for key in ("normalized", "redirects"):
            for m in q.get(key, []):
                alias[m["to"]] = alias.get(m["from"], m["from"])
        for page in q.get("pages", []):
            title = page.get("title")
            orig = alias.get(title, title)
            orig = alias.get(orig, orig)
            revs = page.get("revisions")
            cache[orig] = revs[0]["slots"]["main"]["content"] if revs else None
        for t in chunk:
            cache.setdefault(t, None)
        save_cache(pid, cache)
        print(f"  본문 받는 중 {min(k + batch, len(todo))}/{len(todo)}")
    return {t: cache.get(t) for t in titles}


# ---------------------------------------------------------------- 작가 페이지 → 작품 목록
LINK_RE = re.compile(r"\[\[([^\[\]|#]+)(?:\|[^\[\]]*)?\]\]")
HEAD_RE = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$")


def works_from_author(author):
    text = wikitext(f"저자:{author}")
    if text is None:
        sys.exit(f"[중단] 위키문헌에 '저자:{author}' 페이지가 없습니다.")
    section, out, seen = "(섹션 없음)", [], set()
    for line in text.splitlines():
        m = HEAD_RE.match(line.strip())
        if m:
            section = re.sub(r"[\[\]'{}]", "", m.group(2)).strip()
            continue
        for t in LINK_RE.findall(line):
            t = t.strip()
            if ":" in t or t in seen:  # 분류:, 저자:, 위키백과: 등은 작품이 아님
                continue
            seen.add(t)
            out.append((section, t))
    return out


def section_kind(section):
    if any(h in section for h in PROSE_HINTS):
        return "prose"
    if any(h in section for h in NON_PROSE_HINTS):
        return "non-prose"
    return "unknown"


# ---------------------------------------------------------------- 본문 정리
def strip_templates(s):
    out, depth, i = [], 0, 0
    while i < len(s):
        if s.startswith("{{", i):
            depth += 1; i += 2; continue
        if s.startswith("}}", i) and depth:
            depth -= 1; i += 2; continue
        if depth == 0:
            out.append(s[i])
        i += 1
    return "".join(out)


def clean(wt):
    s = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
    s = re.sub(r"<ref[^>/]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"<references\s*/?>", "", s)
    s = strip_templates(s)
    s = re.sub(r"\{\|.*?\|\}", "", s, flags=re.S)                     # 표
    s = re.sub(r"\[\[(?:분류|Category|파일|File|그림|Image):[^\]]*\]\]", "", s, flags=re.I)
    s = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", s)                # [[링크|글자]] → 글자
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", s)
    s = re.sub(r"'{2,}", "", s)                                        # 굵게·기울임
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)                            # 남은 태그
    s = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", s, flags=re.M)         # 소제목은 글자만
    s = re.sub(r"^[:*#;]+\s*", "", s, flags=re.M)                      # 들여쓰기·목록 기호
    s = s.replace("&nbsp;", " ")
    lines = [ln.strip() for ln in s.splitlines()]
    s = "\n".join(lines)
    s = re.sub(r"\n{3,}", "\n\n", s)                                   # 문단 경계는 빈 줄 하나
    return s.strip()


def hangul_chars(s):
    return sum(1 for ch in s if "가" <= ch <= "힣")


# ---------------------------------------------------------------- 명령
def cmd_list(a):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    works = works_from_author(a.author)
    print(f"[목록] {a.author}: 링크된 작품 {len(works)}개")
    need = [t for sec, t in works if section_kind(sec) != "non-prose"]   # 소설·시는 본문을 받지 않음
    print(f"  소설·시 등 {len(works) - len(need)}편은 건너뛰고, 산문·미분류 {len(need)}편만 본문을 받습니다")
    texts = fetch_many(need, a.pid)
    rows = []
    for section, title in works:
        kind = section_kind(section)
        if kind == "non-prose":
            rows.append(dict(keep=0, title=title, section=section, kind=kind, chars="", hangul="", note="본문 안 받음"))
            continue
        wt = texts.get(title)
        if wt is None:
            rows.append(dict(keep=0, title=title, section=section, kind=kind, chars=0, hangul=0, note="페이지 없음"))
            continue
        body = clean(wt)
        note = ""
        if len(body) < 300 and re.search(r"\[\[/", wt):
            note = "하위 페이지로 나뉜 글 (사람이 확인)"
        keep = int(kind == "prose" and len(body) >= MIN_CHARS)
        rows.append(dict(keep=keep, title=title, section=section, kind=kind,
                         chars=len(body), hangul=hangul_chars(body), note=note))
    path = OUT_DIR / f"{a.pid}_candidates.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["keep", "title", "section", "kind", "chars", "hangul", "note"])
        w.writeheader(); w.writerows(rows)
    k = sum(r["keep"] for r in rows)
    long_unknown = sum(1 for r in rows if r["kind"] == "unknown" and isinstance(r["chars"], int) and r["chars"] >= MIN_CHARS)
    secs = Counter(sec for sec, _ in works)
    print(f"\n섹션별 작품 수: " + ", ".join(f"{s_}({n})" for s_, n in secs.most_common()))
    print(f"후보표 → {path.relative_to(ROOT)}")
    print(f"자동 선택(keep=1): {k}편  (조건: 산문 섹션 + {MIN_CHARS}자 이상)")
    print(f"섹션이 애매해서 사람이 봐야 할 {MIN_CHARS}자 이상 글: {long_unknown}편 (kind=unknown)")
    if k < 30:
        print("주의: 30편이 안 됩니다. kind=unknown 글을 직접 확인해 keep을 1로 바꾸거나 작가를 바꾸세요.")


def kept_titles(pid, required=True):
    path = OUT_DIR / f"{pid}_candidates.csv"
    if not path.exists():
        if required:
            sys.exit(f"[중단] {path} 가 없습니다. 먼저 list를 실행하세요.")
        return []
    with path.open(encoding="utf-8-sig") as f:
        return [r["title"] for r in csv.DictReader(f) if str(r["keep"]).strip() == "1"]


def manual_docs(pid):
    """data/manual/<pid>/*.txt — 다른 사이트(한국사데이터베이스, 공유마당 등)에서 복사한 글.
    파일 하나 = 글 하나. 파일 이름이 제목, 첫 줄이 'source: 주소'면 출처로 저장한다."""
    d = ROOT / "data" / "manual" / pid
    out = []
    for f in sorted(d.glob("*.txt")) if d.exists() else []:
        raw = f.read_text(encoding="utf-8-sig", errors="replace")
        src = ""
        first, _, rest = raw.partition("\n")
        if first.lower().startswith("source:"):
            src, raw = first.split(":", 1)[1].strip(), rest
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(ln.strip() for ln in raw.splitlines())).strip()
        out.append((f.stem, body, src or f"manual:{f.name}"))
    return out


def cmd_build(a):
    titles = kept_titles(a.pid, required=False)
    texts = fetch_many(titles, a.pid) if titles else {}
    items = [(t, clean(texts.get(t) or ""), "https://ko.wikisource.org/wiki/" + urllib.parse.quote(t.replace(" ", "_")))
             for t in titles]
    seen = {t for t, _, _ in items}
    items += [m for m in manual_docs(a.pid) if m[0] not in seen]
    if not items:
        sys.exit(f"[중단] 모을 글이 없습니다. list를 돌리거나 data/manual/{a.pid}/ 에 .txt를 넣으세요.")
    docs, short = [], []
    for t, body, src in items:
        if len(body) < MIN_CHARS:
            short.append(t); continue
        docs.append({"id": f"doc_{len(docs) + 1:03d}", "title": t, "type": "essay", "text": body, "source": src})
    raw = ROOT / "data" / "raw" / f"corpus_{a.pid}_raw.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    n_wiki = sum(1 for d in docs if d["source"].startswith("https://ko.wikisource"))
    print(f"저장 → {raw.relative_to(ROOT)}  ({len(docs)}편: 위키문헌 {n_wiki}, 직접 넣은 글 {len(docs) - n_wiki})")
    if short:
        print(f"{MIN_CHARS}자 미만이라 뺀 글 {len(short)}편: {short[:10]}")
    if len(docs) < a.min_docs:
        print(f"주의: {a.min_docs}편 미만입니다. data/manual/{a.pid}/ 에 글을 더 넣고 다시 build 하세요.")


def whole_nnp_rate(kiwi, texts):
    """어절 하나가 통째로 NNP 한 개로 분석된 비율. 옛 표기를 Kiwi가 못 읽으면 이 값이 커진다."""
    total = whole = 0
    bad = Counter()
    for t in texts:
        for sent in re.split(r"(?<=[.!?])\s+|\n+", t):
            for eo in sent.split():
                core = re.sub(r"[^\w가-힣]", "", eo)
                if hangul_chars(core) < 2:
                    continue
                toks = kiwi.tokenize(core)
                total += 1
                if len(toks) == 1 and toks[0].tag == "NNP":
                    whole += 1; bad[core] += 1
    return (whole / total if total else 0.0), total, bad


def cmd_check(a):
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        sys.exit("kiwipiepy가 필요합니다: pip install kiwipiepy")
    kiwi = Kiwi()
    titles = kept_titles(a.pid, required=False)[: a.n]
    got = fetch_many(titles, a.pid) if titles else {}
    texts = [clean(got.get(t) or "") for t in titles] + [b for _, b, _ in manual_docs(a.pid)][: max(0, a.n - len(titles))]
    r, n, bad = whole_nnp_rate(kiwi, texts)
    base_texts = []
    for p in sorted((ROOT / "data" / "raw").glob("corpus_*_raw.jsonl")):
        if p.stem.replace("_raw", "").replace("corpus_", "") in ("ganghee", "nakyung", "suyoung"):
            base_texts += [json.loads(l)["text"] for l in p.open(encoding="utf-8") if l.strip()][:a.n]
    br, bn, _ = whole_nnp_rate(kiwi, base_texts) if base_texts else (None, 0, None)
    print(f"[옛 표기 점검] {a.pid}: 글 {len(texts)}편, 어절 {n}개")
    print(f"  통째 고유명사(NNP)로 분석된 어절 비율: {r:.1%}")
    if br is not None:
        print(f"  비교: 팀원 글 {br:.1%} (어절 {bn}개)")
    print("  가장 많이 뭉개진 어절:", ", ".join(f"{w}({c})" for w, c in bad.most_common(20)))
    print("  판단 기준(대략): 팀원 글보다 몇 배 이상 높으면 기능 형태소(조사·어미)를 많이 잃는다는 뜻 → 표기 정리나 작가 교체 검토")


# 1962년 이전 사망(저작권 만료), 사망 연도가 확실한 작가만
SCAN_AUTHORS = ["이광수", "김동인", "최남선", "한용운", "이상", "김유정", "나도향", "현진건", "방정환",
                "이육사", "윤동주", "노천명", "계용묵", "채만식", "이효석", "신채호", "안창호", "김소월",
                "최서해", "김교신", "박은식", "안국선", "이인직", "주요섭", "박인환"]


def cmd_scan(a):
    names = a.authors or SCAN_AUTHORS
    rows = []
    for n in names:
        text = wikitext(f"저자:{n}")
        if text is None:
            print(f"  {n}: 작가 페이지 없음"); continue
        section, kinds, secs = "(섹션 없음)", Counter(), Counter()
        for line in text.splitlines():
            m = HEAD_RE.match(line.strip())
            if m:
                section = re.sub(r"[\[\]'{}]", "", m.group(2)).strip(); continue
            for t in LINK_RE.findall(line):
                if ":" in t:
                    continue
                kinds[section_kind(section)] += 1; secs[section] += 1
        rows.append((n, kinds["prose"], kinds["unknown"], kinds["non-prose"], secs))
        print(f"  {n}: 산문 {kinds['prose']} · 미분류 {kinds['unknown']} · 소설·시 등 {kinds['non-prose']}")
    print("\n[산문이 많은 순] (본문 길이는 아직 안 셈. 상위 작가만 list로 2,000자 이상을 세세요)")
    for n, p_, u, np_, secs in sorted(rows, key=lambda r: -(r[1] + r[2])):
        top = ", ".join(f"{k}({v})" for k, v in secs.most_common(4))
        print(f"  {n:6s} 산문 {p_:3d} + 미분류 {u:3d}   | {top}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--author", required=True); p.add_argument("--pid", required=True)
    p = sub.add_parser("build"); p.add_argument("--pid", required=True); p.add_argument("--min-docs", type=int, default=20)
    p = sub.add_parser("scan"); p.add_argument("--authors", nargs="*")
    p = sub.add_parser("check"); p.add_argument("--pid", required=True); p.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    {"list": cmd_list, "build": cmd_build, "check": cmd_check, "scan": cmd_scan}[a.cmd](a)


if __name__ == "__main__":
    main()
