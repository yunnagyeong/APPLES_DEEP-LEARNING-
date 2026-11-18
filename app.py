import time

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Lingua Me",
    layout="wide",
)


# 글씨 크기·진하기 (부스 화면에서 잘 보이게)
STYLE = """
<style>
html, body, [data-testid="stAppViewContainer"] { font-size: 17px; }
[data-testid="stMainBlockContainer"] { max-width: 100%; padding-left: 3rem; padding-right: 3rem; }
h1 { font-size: 3.2rem !important; }
h3 { font-size: 1.8rem !important; color: #111 !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { font-size: 1.2rem; color: #111; line-height: 1.7; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { font-size: 1.15rem !important; color: #333 !important; opacity: 1 !important; }
textarea { font-size: 1.35rem !important; line-height: 1.8 !important; color: #000 !important; -webkit-text-fill-color: #000 !important; }
[data-testid="stWidgetLabel"] p { font-size: 1.3rem !important; font-weight: 700; color: #111; }
[data-testid="stSidebar"] { width: 320px !important; min-width: 320px !important; }
[data-testid="stSidebarUserContent"] { padding-top: 0 !important; }
[data-testid="stSidebarHeader"] { height: 2.2rem !important; min-height: 0 !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: 1.45rem !important; font-weight: 800 !important; color: #000 !important; margin-bottom: 0.3rem; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 0.15rem; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: 0.2rem 0; align-items: center; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size: 1.15rem !important; color: #000; font-weight: 600; }
[data-testid="stRadioOption"] > div > div:first-child { transform: scale(1.2); margin-right: 0.4rem; }
[data-testid="stExpander"] summary p { font-size: 1.2rem !important; color: #111; font-weight: 600; }
[data-testid="stMetricLabel"] p { font-size: 1.15rem !important; color: #222 !important; }
[data-testid="stMetricValue"] { font-size: 2.6rem !important; }
[data-testid="stAlert"] p { font-size: 1.3rem !important; font-weight: 600; }
button[kind="primary"] { min-height: 3.6rem; }
button[kind="primary"] p { font-size: 1.6rem !important; font-weight: 800; color: #fff !important; }
mark { background: #fff3bf; padding: 0 2px; }
textarea:disabled, [data-baseweb="textarea"]:has(textarea:disabled) { opacity: 1 !important; color: #000 !important; -webkit-text-fill-color: #000 !important; font-weight: 500 !important; cursor: text !important; }
[data-baseweb="textarea"], [data-baseweb="base-input"] { opacity: 1 !important; }
</style>
"""

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
# 작가 3인(이광수·이상·이육사) + 팀원 3인 + 블로그(11/18 다시 추가)
KO_PERSONAS = {**DEMO_AUTHORS, "나경": "nakyung", "강희": "ganghee", "수영": "suyoung", "스머프 할아버지": "blog"}
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
    "REVIEW": "대체로 같음 (표현이 많이 바뀐 문장 있음)",
    "FAIL": "실패 (재변환 후에도 오류 남음)",
}


def render_ko_check(r):
    """의미 점검 결과: PASS 그대로 / FAIL이었으면 자동 재번역 이력 / REVIEW면 사람이 볼 문장을 강조 (src/ko_pipeline.py)."""
    import html
    from src import ko_pipeline, ko_verify
    v, chk = r["verify"], r.get("check") or {}
    status = chk.get("status", v["verdict"])
    (st.success if status == "PASS" else st.error if status == "FAIL" else st.info)(
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
    exp = st.expander(f"참고: 표현이 많이 바뀐 문장 {len(chk['review'])}개")  # 기본은 접어 둠
    marks = [p["sentence"] for p in chk["review"] if p["where"] == "output"]
    if marks:
        body = "".join(f"<mark>{html.escape(t)}</mark>" if m else html.escape(t)
                       for t, m in ko_verify.mark_segments(r["converted"], marks))
        body = body.replace("\n", "<br>")  # 빈 줄이 있으면 HTML 블록이 끊겨 빈 상자로 보이는 문제 방지
        exp.markdown(f"<div style='line-height:1.7'>{body}</div>", unsafe_allow_html=True)
    for p in chk["review"]:
        where = {"output": "변환문", "source": "원문"}.get(p["where"], "")
        line = f"- {', '.join(p['labels'])}"
        if p["sentence"]:
            line += f" · {where}: “{p['sentence']}”"
        exp.markdown(line + f"  \n  {' / '.join(dict.fromkeys(p['notes']))}")


def render_ko_style(persona, use_dummy, mode):
    from src import config, ko_style
    user = KO_PERSONAS[persona]
    if config.CORPUS_SPLIT and not use_dummy:
        st.warning(f"LINGUA_CORPUS_SPLIT={config.CORPUS_SPLIT} 설정됨. 작가 글 일부만 검색 중. "
                   "Remove-Item Env:LINGUA_CORPUS_SPLIT 실행 후 재시작")

    source = st.text_area("원문", value=DUMMY_KO_SOURCE, height=300)
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
        st.text_area("원문", r["source"], height=420, disabled=True, key="out_source", label_visibility="collapsed")
    with right:
        st.subheader(f"{r['persona']} 문체")
        st.text_area("변환문", r["converted"], height=420, disabled=True, label_visibility="collapsed")
        st.caption(("가짜 결과 · " if r["dummy"] else "") + f"{KO_MODES[r['mode']]} · {r['seconds']:.1f}초")

    v = r.get("verify")
    if v:
        render_ko_check(r)
        if v["issues"]:
            with st.expander(f"점검 항목 {len(v['issues'])}개"):
                for i in v["issues"]:
                    st.markdown(f"- [{i['severity']}] **{i['label_ko']}** · {i['note']}" + (f" (`{i['evidence']}`)" if i.get("evidence") else ""))
            if v["verdict"] == "REVIEW":
                st.caption("표현이 많이 바뀐 문장은 참고용 표시 (자동 수정 안 함)")

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
# 사이드바: 작가 / 팀원 / 블로그를 묶음별로 나눠 보여 주고, 전체에서 한 명만 고르게 한다
PERSONA_GROUPS = [
    ("작가 선택", ["이광수", "이상", "이육사"]),
    ("팀원 선택", ["나경", "수영", "강희"]),
    ("기타", ["스머프 할아버지"]),
]


def _only(i):
    """i번 묶음에서 고르면 다른 묶음의 선택은 지운다."""
    for j in range(len(PERSONA_GROUPS)):
        if j != i:
            st.session_state[f"group_{j}"] = None


def pick_persona():
    if "group_0" not in st.session_state:  # 처음 켰을 때: 이광수만 선택
        for i, (_, names) in enumerate(PERSONA_GROUPS):
            st.session_state[f"group_{i}"] = names[0] if i == 0 else None
    for i, (label, names) in enumerate(PERSONA_GROUPS):
        if i == len(PERSONA_GROUPS) - 1:
            st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        st.radio(label, names, key=f"group_{i}", on_change=_only, args=(i,),
                 label_visibility="collapsed" if label == "기타" else "visible")
        if i == 0:
            st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    return next(st.session_state[f"group_{i}"] for i in range(len(PERSONA_GROUPS))
                if st.session_state.get(f"group_{i}"))


def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("Lingua Me")
    st.caption("개인화 번역 비교 및 AI 획일화 정량 비교 지표")
    with st.sidebar:
        persona = pick_persona()
        ko_mode = "R"  # 작가 글 예시(RAG)만, 문체 지시 없음 (수영 결정 11/12, D-096)
        use_dummy = False  # 가짜 결과 토글은 화면에서 뺐음. 화면만 확인하려면 True로 바꾸면 됨
    render_ko_style(persona, use_dummy, ko_mode)


if __name__ == "__main__":
    main()
