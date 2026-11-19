import time

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Lingua Me",
    page_icon="assets/logo.png",
    layout="wide",
)
st.logo("assets/logo.png", size="large")


# 글씨 크기·진하기 (부스 화면에서 잘 보이게). 글자색은 테마(라이트/다크)에 맡김 — 검정을 박아 두면 다크 모드에서 안 보인다
STYLE = """
<style>
html, body, [data-testid="stAppViewContainer"] { font-size: 17px; }
[data-testid="stMainBlockContainer"] { max-width: 100%; padding-left: 2.25rem; padding-right: 2.25rem; padding-top: 2.5rem; }
/* 한 화면 맞춤: 제목~변환 버튼, 변환 결과(제목+두 칸+설명)가 각각 화면 높이를 채우도록 칸 높이를 화면 높이에 맞춤 */
.st-key-ko_input textarea { height: max(260px, calc(100vh - 400px)) !important; }
.st-key-ko_results { scroll-margin-top: 3.5rem; }
.st-key-ko_results textarea { height: max(320px, calc(100vh - 255px)) !important; }
/* 뜻 확인 + 펼침 칸 묶음, 살펴볼 문장 항목: 테두리 없이 옅은 바탕 상자 */
.st-key-ko_details { background: color-mix(in srgb, currentColor 4%, transparent); border-radius: 1rem; padding: 1.25rem 1.5rem; }
[class*="st-key-ko_rv_"] { background: color-mix(in srgb, currentColor 5%, transparent); border-radius: 0.75rem; padding: 0.9rem 1.1rem; }
/* 살펴볼 문장 카드의 '원문'/'변환문' 표시: 테두리 알약(문제 유형 배지는 채운 회색이라 구분됨) */
.rv-tag { display: inline-block; min-width: 4.6em; box-sizing: border-box; margin-right: 0.5em; padding: 0 0.6em; border-radius: 999px; text-align: center;
  font-size: 0.8em; font-weight: 700; line-height: 1.7; color: color-mix(in srgb, currentColor 75%, transparent);
  border: 1px solid color-mix(in srgb, currentColor 25%, transparent); background: color-mix(in srgb, currentColor 6%, transparent); }
h1 { font-size: 3.2rem !important; }
h3 { font-size: 1.8rem !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { font-size: 1.2rem; line-height: 1.7; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { font-size: 1.15rem !important; opacity: 1 !important; }
textarea { font-size: 1.15rem !important; line-height: 1.8 !important; padding: 1rem 1.25rem !important; }
[data-testid="stWidgetLabel"] p { font-size: 1.3rem !important; font-weight: 700; }
[data-testid="stSidebar"] { width: 250px !important; min-width: 250px !important; }
[data-testid="stSidebarUserContent"] { padding-top: 0 !important; }
[data-testid="stSidebarHeader"] { height: auto !important; min-height: 0 !important; padding-top: 1.75rem !important; padding-bottom: 0.75rem !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: 1.45rem !important; font-weight: 800 !important; margin-bottom: 0.3rem; }
/* 사이드바 문체 선택: 제목 + 상자 하나 안에 이름을 한 줄씩(선택은 메인 색으로 채움). 회색 톤은 currentColor에서 섞어 라이트/다크 둘 다 맞게 */
[class*="st-key-pgtitle_"] p { font-size: 1.15rem !important; font-weight: 800; margin-bottom: 0.35rem; }
[class*="st-key-pgtitle_"]:not(.st-key-pgtitle_0) { margin-top: 1rem; }
.st-key-persona_picker [role="radiogroup"] { display: flex; flex-direction: column; gap: 6px; padding: 6px; border-radius: 0.8rem;
  border: 1px solid color-mix(in srgb, currentColor 12%, transparent); background: color-mix(in srgb, currentColor 4%, transparent); }
.st-key-persona_picker button[data-variant="segmented_control"] { width: 100%; min-height: 2.1rem; padding: 0 0.9rem !important;
  justify-content: flex-start; border: none !important;
  border-radius: 0.6rem !important; background: transparent !important; box-shadow: none !important; }
.st-key-persona_picker button[data-variant="segmented_control"] p { font-size: 1rem !important; font-weight: 700;
  color: color-mix(in srgb, currentColor 60%, transparent); }
.st-key-persona_picker button[data-variant="segmented_control"]:hover { background: color-mix(in srgb, currentColor 7%, transparent) !important; }
.st-key-persona_picker button[aria-checked="true"], .st-key-persona_picker button[aria-checked="true"]:hover { background: #980120 !important; }
.st-key-persona_picker button[aria-checked="true"] p { color: #fff !important; }
[data-testid="stExpander"] summary p { font-size: 1.1rem !important; font-weight: 600; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p, [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li { font-size: 1.05rem; }
.st-key-ko_review [data-testid="stMarkdownContainer"] p { font-size: 1rem; }
.st-key-ko_review [data-testid="stCaptionContainer"] p { font-size: 0.95rem !important; }
[data-testid="stMetricLabel"] p { font-size: 1.15rem !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { font-size: 2.2rem !important; font-weight: 700; }
[data-testid="stMetricDelta"] p { font-size: 0.95rem !important; }
[data-testid="stAlert"] p { font-size: 1.3rem !important; font-weight: 600; }
.st-key-ko_check p { font-size: 1.05rem !important; }
button[kind="primary"] { min-height: 3.6rem; }
button[kind="primary"] p { font-size: 1.6rem !important; font-weight: 800; color: #fff !important; }
mark { background: #fff3bf; color: #000; padding: 0 2px; }
textarea:disabled, [data-baseweb="textarea"]:has(textarea:disabled) { opacity: 1 !important; font-weight: 500 !important; cursor: text !important; }
[data-testid="stTextArea"] { color: inherit; }
[data-testid="stTextArea"] textarea:disabled { color: inherit !important; -webkit-text-fill-color: currentColor !important; }
[data-testid="stTextArea"] [data-testid="stWidgetLabel"], [data-testid="stTextArea"] [data-testid="stWidgetLabel"] * { color: inherit !important; opacity: 1 !important; }
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
    st.subheader("AI 말투 지표", anchor=False)
    try:
        rows = ko_metrics.table(r["source"], r["converted"], user, persona)
    except BaseException as e:
        st.warning(f"지표 계산 오류 ({type(e).__name__}: {e})")
        return
    a, b = rows[0], rows[1]
    k1, k2, k3 = st.columns(3, gap="medium")
    k1.metric("AI 표현 (100어절당)", f"{b['AI 표현 (100어절당)']:.1f}",
              f"{b['AI 표현 (100어절당)'] - a['AI 표현 (100어절당)']:+.1f} (원문 대비)", delta_color="inverse", border=True)
    k2.metric("어휘 다양도 (MTLD)", f"{b['어휘 다양도 (MTLD)']:.1f}",
              f"{b['어휘 다양도 (MTLD)'] - a['어휘 다양도 (MTLD)']:+.1f} (원문 대비)", border=True)
    if b["문장 길이 일치도"] is not None:
        k3.metric(f"문장 길이 일치도 ({persona})", f"{b['문장 길이 일치도']:.2f}",
                  f"{b['문장 길이 일치도'] - a['문장 길이 일치도']:+.2f} (원문 대비)", border=True)
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


STATUS_KO = {  # 뜻 확인 결과: 작은 배지 + 한 줄 안내. REVIEW는 오탐이 대부분이라 '혹시 모르니 한 번 보기' 정도로만 안내
    "PASS": (":green-badge[:material/check: 뜻 확인]", "원문의 뜻이 잘 전해졌어요."),
    "REVIEW": (":yellow-badge[:material/visibility: 뜻 확인]", "뜻은 대체로 잘 전해졌어요. 혹시 모르니 표시된 문장만 한 번 살펴보세요."),
    "FAIL": (":red-badge[:material/error: 뜻 확인]", "다시 변환해도 뜻이 달라진 부분이 남아 있어요. 아래 항목을 확인해 주세요."),
}


def render_ko_check(r):
    """의미 점검 결과: PASS 그대로 / FAIL이었으면 자동 재번역 이력 / REVIEW면 사람이 볼 문장을 강조 (src/ko_pipeline.py)."""
    import html
    from src import ko_pipeline, ko_verify
    v, chk = r["verify"], r.get("check") or {}
    status = chk.get("status", v["verdict"])
    badge, text = STATUS_KO.get(status, (f":gray-badge[뜻 확인: {status}]", ""))
    with st.container(key="ko_check"):
        st.markdown(f"{badge}&nbsp; {text}")
    if len(chk.get("attempts", [])) > 1:
        st.caption("재변환: " + ko_pipeline.history_line(chk))
        with st.expander("첫 변환문"):
            st.caption(chk["attempts"][0]["summary"])
            st.text_area("첫 변환문", chk["first_output"], height=160, disabled=True, label_visibility="collapsed")
    if chk.get("note"):
        st.caption(chk["note"])
    if status != "REVIEW" or not chk.get("review"):
        return
    exp = st.expander(f"살펴볼 문장 {len(chk['review'])}개", key="ko_review")  # 기본은 접어 둠
    marks = [p["sentence"] for p in chk["review"] if p["where"] == "output"]
    if marks:
        body = "".join(f"<mark>{html.escape(t)}</mark>" if m else html.escape(t)
                       for t, m in ko_verify.mark_segments(r["converted"], marks))
        body = body.replace("\n", "<br>")  # 빈 줄이 있으면 HTML 블록이 끊겨 빈 상자로 보이는 문제 방지
        exp.markdown(f"<div style='line-height:1.7'>{body}</div>", unsafe_allow_html=True)
        exp.space("small")
    missing = "(짝이 되는 문장을 찾지 못했어요)"
    for k, p in enumerate(chk["review"]):
        box = exp.container(key=f"ko_rv_{k}")
        lines = [" ".join(f":gray-badge[{l}]" for l in p["labels"])]
        if p["sentence"]:  # 지적된 문장의 반대쪽(원문↔변환문) 짝도 같이 보여 준다
            if p["where"] == "output":
                src_s, out_s = ko_verify.counterpart(p["sentence"], r["source"]), p["sentence"]
            else:
                src_s, out_s = p["sentence"], ko_verify.counterpart(p["sentence"], r["converted"])
            tag = lambda t: f"<span class='rv-tag'>{t}</span>"
            lines += [f"{tag('원문')}{html.escape(src_s or missing)}", f"{tag('변환문')}{html.escape(out_s or missing)}"]
        box.markdown("  \n".join(lines), unsafe_allow_html=True)
        box.caption(" / ".join(dict.fromkeys(p["notes"])))


def render_ko_style(persona, use_dummy, mode):
    from src import config, ko_style
    user = KO_PERSONAS[persona]
    if config.CORPUS_SPLIT and not use_dummy:
        st.warning(f"LINGUA_CORPUS_SPLIT={config.CORPUS_SPLIT} 설정됨. 작가 글 일부만 검색 중. "
                   "Remove-Item Env:LINGUA_CORPUS_SPLIT 실행 후 재시작")

    source = st.text_area("원문", value=DUMMY_KO_SOURCE, height=300, key="ko_input")
    n_chars = len(source.strip())
    st.caption(f"{n_chars:,}자&nbsp;&nbsp;<span style='font-size:0.85em; opacity:0.55'>"
               "문체 옵션을 선택하고 원문을 작성한 뒤 <b>변환</b> 버튼을 눌러 주세요.</span>", unsafe_allow_html=True)
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
            st.session_state["ko_scroll"] = True  # 결과가 그려진 뒤 결과 칸으로 한 번 스크롤
        except BaseException as e:  # API 키 없음(sys.exit) 포함
            st.error("API 키 없음 (.env의 GOOGLE_API_KEY 확인)"
                     if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}")

    r = st.session_state.get("ko_result")
    if not r:
        return

    st.space("large")
    with st.container(key="ko_results"):  # 변환 후 스크롤 위치. 칸 높이는 CSS에서 화면 높이에 맞춤
        st.subheader("변환 결과", anchor=False)
        left, right = st.columns(2, gap="medium")
        with left:
            st.text_area("원문", r["source"], height=460, disabled=True, key="out_source")
        with right:
            st.text_area(f"{r['persona']} 문체", r["converted"], height=460, disabled=True)
            st.caption(("가짜 결과 · " if r["dummy"] else "") + f"{KO_MODES[r['mode']]} · {r['seconds']:.1f}초")

    with st.container(key="ko_details"):  # 뜻 확인 + 펼침 칸들을 한 묶음으로(테두리 없는 상자)
        v = r.get("verify")
        if v:
            render_ko_check(r)
            if v["issues"]:
                with st.expander(f"점검 항목 {len(v['issues'])}개"):
                    for i in v["issues"]:
                        st.markdown(f"- [{i['severity']}] **{i['label_ko']}** · {i['note']}" + (f" (`{i['evidence']}`)" if i.get("evidence") else ""))

        if "P" in r["mode"]:  # 지금 데모는 R만 쓰므로 표시되지 않음(이전 방식 호환용)
            st.markdown("**적용된 문체 지시**")
            for line in r["lines"] or ["(지시 없음)"]:
                st.markdown(f"- {line}")
        with st.expander("공통 규칙"):
            for i, rule in enumerate(ko_style.RULE_SETS[ko_style.RULES_VERSION], 1):
                st.markdown(f"{i}. {rule}")

        if "R" in r["mode"]:
            with st.expander(f"RAG 검색 결과: {r['persona']} 글 {len(r['examples'])}편 (유사도순)"):
                for i, e in enumerate(r["examples"], 1):
                    st.markdown(f"**{i}.** {e.get('topic') or e.get('type') or ''} · 유사도 {e['score']:.2f}")
                    st.write(e["text"])

    st.space("large")
    try:  # 지표 계산이 실패해도 위 화면이 사라지지 않게
        render_ko_metrics(r, user, persona)
    except BaseException as e:
        st.warning(f"지표 표시 오류 ({type(e).__name__}: {e})")

    if st.session_state.pop("ko_scroll", False):  # 시각을 넣어 변환할 때마다 새 요소로 그려지게(같은 내용이면 스크립트가 다시 안 돎)
        st.html(f"<script>/* {time.time()} */ setTimeout(() => document.querySelector('.st-key-ko_results')"
                "?.scrollIntoView({behavior: 'smooth', block: 'start'}), 250);</script>", unsafe_allow_javascript=True)


# ==================================================
# 화면
# ==================================================
# 사이드바: 작가 / 팀원 / 블로그를 묶음별 카드로 나눠 보여 주고, 전체에서 한 명만 고르게 한다
PERSONA_GROUPS = [
    ("작가 문체 선택", ["이광수", "이상", "이육사"]),
    ("블로그 문체 선택", ["스머프 할아버지"]),
    ("개인 문체 선택", ["나경", "수영", "강희"]),
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
    with st.container(key="persona_picker"):
        for i, (label, names) in enumerate(PERSONA_GROUPS):
            with st.container(key=f"pgtitle_{i}"):
                st.markdown(f"**{label}**")
            st.segmented_control(label, names, key=f"group_{i}", on_change=_only, args=(i,),
                                 required=True, width="stretch", label_visibility="collapsed")
    return next(st.session_state[f"group_{i}"] for i in range(len(PERSONA_GROUPS))
                if st.session_state.get(f"group_{i}"))


def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("Lingua Me", anchor=False)
    st.caption("개인화 번역 비교 및 AI 획일화 정량 비교 지표")
    with st.sidebar:
        persona = pick_persona()
        ko_mode = "R"  # 작가 글 예시(RAG)만, 문체 지시 없음 (수영 결정 11/12, D-096)
        use_dummy = False  # 가짜 결과 토글은 화면에서 뺐음. 화면만 확인하려면 True로 바꾸면 됨
    render_ko_style(persona, use_dummy, ko_mode)


if __name__ == "__main__":
    main()
