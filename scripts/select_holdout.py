"""홀드아웃 문서 선정 (D-030, 12·16번 개인화 비교용). 규칙 + 고정 시드 무작위, 손으로 고르지 않는다.

규칙(결과를 보기 전에 정한 것, 결과 파일 `rules`에도 기록):
 1. 고정: v2 `data/measure/heldout_sources.jsonl`의 긴 문단(L1~L8) 출처 문서(origin "<사람> doc_NNN").
 2. 채움: 사람×장르별 목표 편수(보고서 2 + 에세이 1)가 고정 문서만으로 모자라면, 같은 사람·장르의
    나머지 문서에서 `random.Random(seed).sample`로 채운다. 고정이 목표보다 많으면 그대로 둔다.
 3. 사후 조건(규칙으로만): 홀드아웃 문서 안의 후보 문단(4~6문장·150~450자)이
    번역 대상 사용자의 기능어 지시 그룹이 있으면 지시 대상 ≥ 4개·대조군 ≥ 2개, 지시 그룹이 없으면 후보 ≥ 6개.
    부족하면 다음 시드(seed+1, 최대 MAX_TRIES)로 채움 문서를 다시 뽑는다. 고정 문서는 바꾸지 않는다.
 4. 홀드아웃 문서는 RAG 풀(doc id 제외)과 `style_reference`(build_style_reference --holdout) 계산에서 뺀다.

사용: python scripts/select_holdout.py [--seed 20261101] [--save]  (저장 위치 data/measure/holdout_docs.json)
API·LLM 없음.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

AUTHORS = ["ganghee", "nakyung", "suyoung"]
KO = {"강희": "ganghee", "나경": "nakyung", "수영": "suyoung"}
TARGET = {"report": 2, "essay": 1}
MIN_TARGET_PARAS, MIN_CONTROL_PARAS, MIN_ANY_PARAS = 4, 2, 6
MAX_TRIES = 20
OUT = ROOT / "data/measure/holdout_docs.json"


def load_raw():
    docs = {}
    for u in AUTHORS:
        for line in open(ROOT / f"data/raw/corpus_{u}_raw.jsonl", encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                if d.get("type") in ("essay", "report"):
                    docs.setdefault(u, {})[d["id"]] = d["type"]
    return docs


def fixed_docs(docs):
    fixed = {u: [] for u in AUTHORS}
    for line in open(ROOT / "data/measure/heldout_sources.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        m = re.match(r"(\S+)\s+(doc_\d+)", r.get("origin", ""))
        if r.get("kind") == "긴 문단" and m and m.group(1) in KO:
            u, did = KO[m.group(1)], m.group(2)
            if did not in fixed[u]:
                fixed[u].append(did)
    for u in AUTHORS:
        for did in fixed[u]:
            assert did in docs[u], f"{u} {did}는 essay/report 문서가 아님"
    return fixed


def user_groups_now():
    from src import generate as g
    import compare_structure_rules as c
    out = {}
    for u in AUTHORS:
        out[u] = c.user_groups(g._style_profile_to_instructions({}, user=u), g)
    return out


def choose(seed, docs, fixed):
    rng = random.Random(seed)
    chosen = {}
    for u in AUTHORS:
        sel = list(fixed[u])
        have = Counter(docs[u][d] for d in sel)
        for genre in ("report", "essay"):
            need = TARGET[genre] - have[genre]
            if need > 0:
                pool = sorted(d for d, t in docs[u].items() if t == genre and d not in sel)
                sel += rng.sample(pool, min(need, len(pool)))
        chosen[u] = sorted(sel)
    return chosen


def check(chosen, groups):
    import compare_structure_rules as c
    res = {}
    for u in AUTHORS:
        cands = c.candidate_paragraphs([u], groups[u], {u: set(chosen[u])})
        if groups[u]:
            hit = sum(1 for x in cands if x["ko_target"] >= 1)
            ctl = sum(1 for x in cands if x["ko_target"] == 0)
            ok = hit >= MIN_TARGET_PARAS and ctl >= MIN_CONTROL_PARAS
            res[u] = {"groups": groups[u], "target_paragraphs": hit, "control_paragraphs": ctl, "ok": ok}
        else:
            n = len(cands)
            res[u] = {"groups": [], "candidate_paragraphs": n, "ok": n >= MIN_ANY_PARAS}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    docs = load_raw()
    fixed = fixed_docs(docs)
    groups = user_groups_now()
    for k in range(MAX_TRIES):
        seed = a.seed + k
        chosen = choose(seed, docs, fixed)
        res = check(chosen, groups)
        print(f"seed {seed}: " + " | ".join(f"{u} {'OK' if r['ok'] else 'NG'} {({kk: vv for kk, vv in r.items() if kk not in ('groups', 'ok')})}" for u, r in res.items()))
        if all(r["ok"] for r in res.values()):
            break
    else:
        sys.exit("사후 조건을 만족하는 시드를 찾지 못했습니다.")
    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed_requested": a.seed, "seed_used": seed, "tries": k + 1,
        "rules": [
            "고정: v2 heldout_sources.jsonl 긴 문단 출처 문서",
            f"채움: 사람×장르 목표 {TARGET} 미달분을 random.Random(seed).sample로, 고정은 바꾸지 않음",
            f"사후 조건: 지시 그룹 있으면 지시 대상 문단 ≥{MIN_TARGET_PARAS}·대조군 ≥{MIN_CONTROL_PARAS}, 없으면 후보 ≥{MIN_ANY_PARAS} (문단 4~6문장·150~450자). 부족하면 다음 시드",
            "홀드아웃 문서는 RAG 풀(doc id)과 style_reference 계산에서 제외",
        ],
        "fixed": fixed, "holdout": chosen,
        "types": {u: {d: docs[u][d] for d in chosen[u]} for u in AUTHORS},
        "post_check": res,
        "note": "post_check는 선정 시점(홀드아웃 제외 재계산 전) 지시 그룹 기준. 재계산 뒤 지시 그룹이 바뀌면 다시 확인.",
    }
    print(json.dumps({k: out[k] for k in ("seed_used", "tries", "holdout", "types", "post_check")}, ensure_ascii=False, indent=1))
    if a.save:
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("저장:", OUT)


if __name__ == "__main__":
    main()
