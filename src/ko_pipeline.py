"""한→한 문체 변환 → 의미 검증 → FAIL일 때만 재번역 (강희).

ko_style.convert_style(변환, 팀원)을 감싸서 결과마다 처리를 나눈다.
  PASS   → 그대로 둔다.
  REVIEW → 고치지 않는다. 문제 문장(ko_verify.review_points)을 표시해 사람이 그 부분을 주의해서 보게 한다.
           (REVIEW는 오탐이 섞여 있어 자동으로 고치면 멀쩡한 문체를 망칠 수 있다.)
  FAIL   → 기계가 다시 쓴다. 문제 유형에 따라
           ① 머리말·제목·끝맺음 인사가 섞였으면 먼저 규칙으로 지운다 (API 호출 없음, 재번역 횟수에 안 셈)
           ② 빈 출력이면 같은 프롬프트로 다시 생성
           ③ 그 밖(숫자 불일치·예시 베낌 등)은 "지적된 부분만 고치는" 최소 수정 프롬프트로 다시 생성.
              이 프롬프트에는 검색 예시를 넣지 않는다(예시를 또 베끼지 않게). 문체는 이전 변환문을 그대로 두게 해서 지킨다.
           재번역은 최대 max_retries번(기본 1). 고칠 대상은 critical 문제만이고 major는 사람 몫으로 남긴다.
시도마다 다시 검증하고, 가장 좋은 시도(PASS > REVIEW > FAIL, 같으면 앞 시도 = 원래 문체에 가까움)를 돌려준다.
영어 쪽 src/pipeline.py의 재번역 원칙(최소 수정, 최선 시도 반환)과 같다.

비용: PASS·REVIEW는 변환 1회 그대로. FAIL만 생성 호출이 최대 max_retries번 늘고, 검증은 규칙이라 0원.
평가 실험(scripts/ko_doc_transfer.py)은 지금처럼 첫 시도로 잰다. 이 모듈을 평가에 쓸 때는 first_verdict(첫 시도)와
최종 verdict를 따로 적는다(재번역이 구해 낸 비율을 숨기지 않게).

    from src.ko_pipeline import convert_checked
    r = convert_checked(text, "ganghee")         # convert_style과 같은 인자 + max_retries
    r["converted"], r["verify"]                   # 최종(가장 좋은 시도) 변환문과 그 검증 결과
    r["check"]["status"]                          # PASS | REVIEW | FAIL
    r["check"]["review"]                          # REVIEW일 때 사람이 볼 문장들
    r["check"]["attempts"]                        # 시도 이력 [{n, how, verdict, summary, output, api_call}]
"""
from __future__ import annotations

DEFAULT_MAX_RETRIES = 1
_RANK = {"PASS": 2, "REVIEW": 1, "FAIL": 0}

FIX_PROMPT = """한 필자의 문체로 다시 쓴 글을 자동 의미 점검으로 확인했더니 아래 문제가 있었습니다.
문제를 고치는 데 필요한 부분만 최소로 고치세요.

규칙:
1. 지적된 부분만 고치고, 나머지 문장과 표현(필자의 문체)은 그대로 둔다.
2. 원문에 없는 내용을 더하거나, 있는 내용을 빼지 않는다.
3. 원문의 확신 정도(추측·단정·완곡)와 형식, 문단 나눔을 그대로 둔다.

원문:
{source}

고칠 글:
{previous}

찾은 문제:
{problems}

고친 글만 출력하세요. 설명이나 제목은 쓰지 마세요."""

MESSAGES = {
    "PASS": "의미 점검 통과",
    "REVIEW": "뜻이 바뀌었을 수 있는 곳이 있습니다. 표시된 문장을 사람이 확인하세요.",
    "FAIL": "자동 재번역 뒤에도 의미 오류가 남았습니다. 이 결과는 쓰기 전에 반드시 확인하세요.",
}


def fix_prompt(source: str, previous: str, verify_result: dict) -> str:
    from . import ko_verify
    return FIX_PROMPT.format(source=source.strip(), previous=previous.strip(),
                             problems=ko_verify.fix_instructions(verify_result))


def convert_checked(text: str, user: str, mode: str = "R", *, max_retries: int = DEFAULT_MAX_RETRIES,
                    examples: list[dict] | None = None, exclude_doc_ids: set[str] | None = None,
                    dummy_output: str | None = None, version: str = "new", boost: bool = False) -> dict:
    """convert_style + 검증 + FAIL이면 재번역. 반환은 convert_style 결과에 아래를 더한 것.
      converted, verify : 가장 좋은 시도의 변환문과 검증 결과 (화면·후속 코드는 이 둘만 보면 된다)
      check : {status, message, first_verdict, first_output, retries, best_attempt, attempts, review}
    dummy_output(데모 가짜 결과)일 때는 API를 부르지 않으므로 규칙 삭제(①)만 하고 재번역(②③)은 건너뛴다.
    """
    from . import ko_style, ko_verify

    r = ko_style.convert_style(text, user, mode, examples=examples, exclude_doc_ids=exclude_doc_ids,
                               dummy_output=dummy_output, version=version, boost=boost, verify=True)
    used = r["examples"] or None
    attempts = [{"n": 1, "how": "첫 변환", "output": r["converted"], "verify": r["verify"],
                 "api_call": dummy_output is None}]
    retries, note = 0, ""
    while attempts[-1]["verify"]["verdict"] == "FAIL":
        cur = attempts[-1]
        types = {i["type"] for i in ko_verify.retry_issues(cur["verify"])}
        if "META_TEXT" in types:  # ① 규칙 삭제 (공짜, 횟수에 안 셈)
            cleaned = ko_verify.strip_meta(text, cur["output"])
            if cleaned and cleaned != cur["output"]:
                attempts.append({"n": len(attempts) + 1, "how": "머리말 규칙 삭제", "output": cleaned,
                                 "verify": ko_verify.verify(text, cleaned, used), "api_call": False})
                continue
        if retries >= max_retries:
            break
        if dummy_output is not None:
            note = "가짜 결과 모드라 재번역을 건너뜀"
            break
        retries += 1
        if "EMPTY_OUTPUT" in types:  # ② 같은 프롬프트로 다시
            how, prompt = "같은 프롬프트로 다시 생성", r["prompt"]
        else:  # ③ 지적 부분만 최소 수정
            how, prompt = "지적 부분만 고쳐 다시 생성", fix_prompt(text, cur["output"], cur["verify"])
        try:
            new = ko_style.generate(prompt, "한→한 재번역")
        except Exception as e:  # 재번역이 실패해도 첫 결과는 돌려준다
            note = f"재번역 호출 실패: {type(e).__name__}: {str(e)[:120]}"
            break
        attempts.append({"n": len(attempts) + 1, "how": how, "output": new,
                         "verify": ko_verify.verify(text, new, used), "api_call": True, "prompt": prompt})

    best = max(range(len(attempts)), key=lambda i: (_RANK[attempts[i]["verify"]["verdict"]], -i))
    b = attempts[best]
    status = b["verify"]["verdict"]
    r["converted"], r["verify"] = b["output"], b["verify"]
    r["check"] = {
        "status": status,
        "message": MESSAGES[status],
        "first_verdict": attempts[0]["verify"]["verdict"],
        "first_output": attempts[0]["output"],
        "retries": retries,
        "best_attempt": best + 1,
        "attempts": [{k: a[k] for k in ("n", "how", "output", "api_call")} |
                     {"verdict": a["verify"]["verdict"], "summary": ko_verify.summarize(a["verify"])} for a in attempts],
        "review": ko_verify.review_points(b["verify"]) if status == "REVIEW" else [],
        "note": note,
    }
    return r


def history_line(check: dict) -> str:
    """'FAIL → 지적 부분만 고쳐 다시 생성 → PASS' 같은 한 줄 이력."""
    parts = [check["attempts"][0]["verdict"]]
    for a in check["attempts"][1:]:
        parts += [a["how"], a["verdict"]]
    return " → ".join(parts) + (f" (최종: {check['best_attempt']}번째 시도)" if len(check["attempts"]) > 1 else "")
