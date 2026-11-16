"""코퍼스 표기 정규화(한자 병기 제거). 형식 모방을 막는 범용 전처리. API 없음.
 - 漢字[한글] / 漢字 [한글] -> 한글 ,  (漢字) 괄호 안이 한자뿐이면 삭제
사용: python scripts/ko_normalize_corpus.py <id> [--suffix h]   data/raw/corpus_<id>_raw.jsonl -> corpus_<id><suffix>_raw.jsonl
"""
import json, re, sys, pathlib
HJ = r"一-鿿㐀-䶿"
PAT_ANN = re.compile(rf"(?:[{HJ}]+[ \t]*)+\[([^\]]*?)\]")
PAT_PAREN = re.compile(rf"\([{HJ}·,、 ]+\)")
def norm(t):
    t = PAT_ANN.sub(lambda m: m.group(1).replace("\n", "").strip(), t)
    return PAT_PAREN.sub("", t)
if __name__ == "__main__":
    cid = sys.argv[1]; suf = sys.argv[sys.argv.index("--suffix")+1] if "--suffix" in sys.argv else "h"
    root = pathlib.Path(__file__).resolve().parent.parent / "data/raw"
    src, dst = root/f"corpus_{cid}_raw.jsonl", root/f"corpus_{cid}{suf}_raw.jsonl"
    n = res = tot = 0
    with dst.open("w", encoding="utf-8") as f:
        for l in src.open(encoding="utf-8"):
            if not l.strip(): continue
            o = json.loads(l); a = o["text"]; o["text"] = norm(a)
            tot += len(a); res += len(re.findall(rf"[{HJ}]", o["text"])); n += 1
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"{cid}: {n}편 -> {dst.name} / 남은 한자 {res}자 (원문 {tot}자)")
