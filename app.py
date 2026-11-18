import time

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Lingua Me",
    layout="wide",
)

# ==================================================
# 말투 바꾸기 (src/ko_style.py, D-050)
#   한→영 번역 화면(이전 실험)은 11/18에 앱에서 뺐습니다. 코드는 git 기록(c959008 이전 app.py)에 있습니다.
#   중립적인 한국어를 선택한 사람의 문체로 다시 씁니다. 변환 뒤 규칙 기반 의미 점검(src/ko_verify)을 붙입니다.
#   RAG 예시는 전체 코퍼스에서 검색합니다(평가 실험 scripts/ko_style_transfer.py와 달리 평가 문서 제외 없음).
# ==================================================
KO_MODES = {
    "P+R": "문체 지시 + 작가 글 참고",
    "P": "문체 지시만",
    "R": "작가 글 참고 (RAG)",
}
from src.ko_style import DEMO_AUTHORS
# 데모는 작가 3인(이광수·이상·이육사)만 보여 준다(11/18, 팀원·블로그 코퍼스 대신 공개 작가 글로 교체).
# 팀원·블로그 문체를 다시 띄우려면: {**DEMO_AUTHORS, "나경": "nakyung", "강희": "ganghee", "수영": "suyoung", "스머프 할배 블로그": "blog"}
KO_PERSONAS = dict(DEMO_AUTHORS)
DUMMY_KO_SOURCE = (  # 데모 기본 예시 글(AI가 쓴 듯한 평범한 글, 3문단)
    "가을이 되면 많은 사람들이 책을 읽고 싶어진다. 선선한 바람과 맑은 하늘은 마음을 차분하게 만들어 주고, "
    "길어진 밤은 한 권의 책에 집중할 수 있는 시간을 준다. 이러한 이유로 가을은 예로부터 독서의 계절이라고 불려 왔다.\n\n"
    "그러나 오늘날 사람들은 바쁜 일상 속에서 책을 펼칠 여유를 찾기 어렵다. 휴대전화로 짧은 글과 영상을 보는 시간은 늘어났지만, "
    "한 권의 책을 끝까지 읽는 경험은 점점 줄어들고 있다. 이는 단순히 읽는 양이 줄어든 것이 아니라, "
    "하나의 생각을 오래 붙잡고 깊이 들여다보는 힘이 약해지고 있다는 것을 의미한다.\n\n"
    "따라서 독서는 지식을 얻는 수단일 뿐만 아니라 자신을 돌아보는 중요한 시간이라고 할 수 있다. "
    "하루에 몇 쪽이라도 꾸준히 읽는 습관을 기른다면, 우리는 더 넓은 시각으로 세상을 바라볼 수 있을 것이다. "
    "올가을에는 오래 미뤄 두었던 책 한 권을 꺼내 천천히 읽어 보는 것은 어떨까.")
DUMMY_KO_OUTPUTS = {  # 가짜 결과(화면 확인용): 실제 모델 출력이 아님
    "blog": "요새 AI가 쓴 글은 다들 비슷비슷해지잖아. 그래도 사람마다 글 쓰는 건 다르지. 그 차이가 말투에 고스란히 나오니 번역에서도 지켜야 해. 그러니 내 문체를 살려 주는 변환이 필요하지.",
    "ganghee": "최근 AI가 쓴 글은 서로 비슷해지는 경향이 있다. 그러나 사람마다 글을 쓰는 방식은 다르며, "
               "이 차이는 말투에 그대로 드러나므로 번역에서도 지켜져야 한다. 그래서 개인의 문체를 반영하는 변환이 필요하다.",
    "nakyung": "최근 AI가 쓴 글은 서로 비슷해지는 경향이 있다. 그러나 사람마다 글을 쓰는 방식은 다르다. "
               "이 차이는 말투에 그대로 드러나기에 번역에서도 지켜져야 한다. 따라서 개인의 문체를 반영하는 변환이 필요하다.",
    "suyoung": "최근 AI가 쓴 글은 서로 비슷해지는 경향이 있지만 사람마다 글을 쓰는 방식은 다르고, "
               "이 차이는 말투에 그대로 드러나는 것이어서 번역에서도 지켜져야 하며, 개인의 문체를 반영하는 변환이 필요하다.",
}


@st.cache_resource(show_spinner=False)
def warm_up_ko():
    """Kiwi를 미리 불러 둔다(한→한 모드의 기능 감지·문체 판정용)."""
    from src import style_features as sf
    sf._get_kiwi().tokenize("준비")
    return True


def run_ko_style(source, user, mode, use_dummy):
    from src import ko_pipeline
    start = time.time()
    dummy = None
    if use_dummy:
        dummy = DUMMY_KO_OUTPUTS.get(user) if source.strip() == DUMMY_KO_SOURCE and user in DUMMY_KO_OUTPUTS else "[가짜 결과] " + source
    # 변환 → 규칙 검증 → FAIL이면 자동 재번역(최대 1회), REVIEW는 표시만 (src/ko_pipeline.py, 강희)
    r = ko_pipeline.convert_checked(source, user, mode, dummy_output=dummy, unit="para")  # 문단마다 생성(11/18 수영 채택, B-066)
    r["seconds"] = time.time() - start
    return r


def render_ko_metrics(r, user, persona):
    """한→한 결과의 AI 획일화·스타일 지표 (src/ko_metrics, API 없음). 한→영 대시보드 지표의 한국어판(D-100)."""
    from src import ko_metrics
    st.markdown("**AI 말투 지표**")
    try:
        rows = ko_metrics.table(r["source"], r["converted"], user, persona)
    except BaseException as e:
        st.warning(f"지표 계산 오류 ({type(e).__name__}: {e})")
        return
    a, b = rows[0], rows[1]
    k1, k2, k3 = st.columns(3)
    k1.metric("AI 표현 (100어절당)", f"{b['AI 표현 (100어절당)']:.1f}",
              f"{b['AI 표현 (100어절당)'] - a['AI 표현 (100어절당)']:+.1f} (원문 대비)", delta_color="inverse")
    k2.metric("어휘 다양도 (MTLD)", f"{b['어휘 다양도 (MTLD)']:.1f}",
              f"{b['어휘 다양도 (MTLD)'] - a['어휘 다양도 (MTLD)']:+.1f} (원문 대비)")
    if b["문장 길이 일치도"] is not None:
        k3.metric(f"문장 길이 일치도 ({persona})", f"{b['문장 길이 일치도']:.2f}",
                  f"{b['문장 길이 일치도'] - a['문장 길이 일치도']:+.2f} (원문 대비)")
    st.dataframe(pd.DataFrame(rows).set_index("구분").round(3), width="stretch")
    fmt = lambda c: ", ".join(f"{k}×{v}" for k, v in c.most_common()) or "없음"
    st.caption(f"AI 표현 · 원문: {fmt(ko_metrics.find_markers(r['source']))} / 변환문: {fmt(ko_metrics.find_markers(r['converted']))}")
    mk = ko_metrics.load_markers()
    with st.expander("AI 표현 목록"):
        if mk.get("missing"):
            st.caption("data/ko_ai_markers.json 없음 (scripts/build_ko_ai_markers.py 실행)")
        else:
            st.caption(f"사람 글과 AI 재작성 글 {mk.get('n_pairs')}쌍 비교, AI 쪽에서 많이 나온 표현"
                       + (" (검토 전)" if not mk.get("reviewed") else ""))
            st.markdown(", ".join(m["label"] for m in mk["markers"] if m["kind"] == "ai_marker"))
            st.caption("'-습니다'체는 별도 집계. 소표본 기준이라 경향만 참고.")


STATUS_KO = {  # 화면에 보이는 뜻 확인 결과 문구
    "PASS": "통과",
    "REVIEW": "확인 필요",
    "FAIL": "실패 (재변환 후에도 오류 남음)",
}


def render_ko_check(r):
    """의미 점검 결과: PASS 그대로 / FAIL이었으면 자동 재번역 이력 / REVIEW면 사람이 볼 문장을 강조 (src/ko_pipeline.py)."""
    import html
    from src import ko_pipeline, ko_verify
    v, chk = r["verify"], r.get("check") or {}
    status = chk.get("status", v["verdict"])
    (st.success if status == "PASS" else st.error if status == "FAIL" else st.warning)(
        f"뜻 확인: {STATUS_KO.get(status, status)}")
    if len(chk.get("attempts", [])) > 1:
        st.caption("재변환: " + ko_pipeline.history_line(chk))
        with st.expander("첫 변환문"):
            st.caption(chk["attempts"][0]["summary"])
            st.text_area("첫 변환문", chk["first_output"], height=160, disabled=True, label_visibility="collapsed")
    if chk.get("note"):
        st.caption(chk["note"])
    if status != "REVIEW" or not chk.get("review"):
        return
    st.markdown("**확인할 문장** (자동 수정 안 함)")
    marks = [p["sentence"] for p in chk["review"] if p["where"] == "output"]
    if marks:
        body = "".join(f"<mark>{html.escape(t)}</mark>" if m else html.escape(t)
                       for t, m in ko_verify.mark_segments(r["converted"], marks))
        body = body.replace("\n", "<br>")  # 빈 줄이 있으면 HTML 블록이 끊겨 빈 상자로 보이는 문제 방지
        st.markdown(f"<div style='line-height:1.7'>{body}</div>", unsafe_allow_html=True)
    for p in chk["review"]:
        where = {"output": "변환문", "source": "원문"}.get(p["where"], "")
        line = f"- **{', '.join(p['labels'])}**"
        if p["sentence"]:
            line += f" · {where}: “{p['sentence']}”"
        st.markdown(line + f"  \n  {' / '.join(dict.fromkeys(p['notes']))}")


def render_ko_style(persona, use_dummy, mode):
    from src import config, ko_style
    user = KO_PERSONAS[persona]
    if config.CORPUS_SPLIT and not use_dummy:
        st.warning(f"LINGUA_CORPUS_SPLIT={config.CORPUS_SPLIT} 설정됨. 작가 글 일부만 검색 중. "
                   "Remove-Item Env:LINGUA_CORPUS_SPLIT 실행 후 재시작")

    source = st.text_area("원문", value=DUMMY_KO_SOURCE, height=260)
    n_chars = len(source.strip())
    st.caption(f"{n_chars:,}자")
    if 0 < n_chars < 300:  # 문체 판정 점수(1,000자 기준)는 화면에서 뺐으므로 아주 짧을 때만 안내
        st.info("글이 짧으면 문체 차이가 잘 안 드러남 (두세 문단 이상 권장)")

    if st.button("변환", type="primary", width="stretch", disabled=n_chars == 0):
        try:
            if not use_dummy:
                warm_up_ko()
            with st.spinner(f"{persona} 문체로 변환 중..."):
                r = run_ko_style(source, user, mode, use_dummy)
            r.update(persona=persona, source=source, dummy=use_dummy)
            st.session_state["ko_result"] = r
        except BaseException as e:  # API 키 없음(sys.exit) 포함
            st.error("API 키 없음 (.env의 GOOGLE_API_KEY 확인)"
                     if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}")

    r = st.session_state.get("ko_result")
    if not r:
        st.info("원문 입력 후 변환 버튼")
        return

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("원문")
        st.text_area("원문", r["source"], height=260, disabled=True, key="out_source", label_visibility="collapsed")
    with right:
        st.subheader(f"{r['persona']} 문체")
        st.text_area("변환문", r["converted"], height=260, disabled=True, label_visibility="collapsed")
        st.caption(("가짜 결과 · " if r["dummy"] else "") + f"{KO_MODES[r['mode']]} · {r['seconds']:.1f}초")

    v = r.get("verify")
    if v:
        render_ko_check(r)
        if v["issues"]:
            with st.expander(f"점검 항목 {len(v['issues'])}개"):
                for i in v["issues"]:
                    st.markdown(f"- [{i['severity']}] **{i['label_ko']}** · {i['note']}" + (f" (`{i['evidence']}`)" if i.get("evidence") else ""))
            if v["verdict"] == "REVIEW":
                st.caption("확인 필요 = 뜻이 바뀌었을 가능성 有")

    if "P" in r["mode"]:  # 지금 데모는 R만 쓰므로 표시되지 않음(이전 방식 호환용)
        st.markdown("**적용된 문체 지시**")
        for line in r["lines"] or ["(지시 없음)"]:
            st.markdown(f"- {line}")
    with st.expander("공통 규칙"):
        for i, rule in enumerate(ko_style.COMMON_RULES, 1):
            st.markdown(f"{i}. {rule}")

    if "R" in r["mode"]:
        with st.expander(f"RAG 검색 결과: {r['persona']} 글 {len(r['examples'])}편 (유사도순)"):
            for i, e in enumerate(r["examples"], 1):
                st.markdown(f"**{i}.** {e.get('topic') or e.get('type') or ''} · 유사도 {e['score']:.2f}")
                st.write(e["text"])

    try:  # 지표 계산이 실패해도 위 화면이 사라지지 않게
        render_ko_metrics(r, user, persona)
    except BaseException as e:
        st.warning(f"지표 표시 오류 ({type(e).__name__}: {e})")


# ==================================================
# 화면
# ==================================================
def main():
    st.title("Lingua Me")
    st.caption("개인화 번역 비교 및 AI 획일화 정량 비교 지표")
    with st.sidebar:
        st.header("설정")
        persona = st.radio("작가 선택", list(KO_PERSONAS))
        ko_mode = "R"  # 작가 글 예시(RAG)만, 문체 지시 없음 (수영 결정 11/12, D-096)
        use_dummy = False  # 가짜 결과 토글은 화면에서 뺐음. 화면만 확인하려면 True로 바꾸면 됨
    render_ko_style(persona, use_dummy, ko_mode)


if __name__ == "__main__":
    main()
