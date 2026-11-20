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
.st-key-ko_input textarea { height: max(260px, calc(100vh - 420px)) !important; }
.st-key-ko_results, .st-key-ko_pending, .st-key-ko_details { scroll-margin-top: 3.5rem; }
.st-key-ko_results textarea, .st-key-ko_pending textarea, .ko-writing { height: max(320px, calc(100vh - 290px)) !important; }
/* 물음표 툴팁: 본문용 큰 글씨(1.2rem)가 툴팁에도 걸려서 따로 줄임 */
[data-testid="stTooltipContent"] p, [data-testid="stTooltipContent"] li { font-size: 0.9rem !important; line-height: 1.6 !important; margin-bottom: 0.5rem; }
[data-testid="stTooltipContent"] p:last-child { margin-bottom: 0; }
[data-testid="stTooltipContent"] [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
/* 오른쪽 위 실행 중 아이콘(달리는 사람·Stop) 숨김: 로딩은 결과 자리의 글줄 애니메이션으로 보여 줌 */
[data-testid="stStatusWidget"] { display: none !important; }
/* 결과 칸 이름 줄 + 복사 버튼(칸 위에 마우스를 올리면 나타남) */
[class*="st-key-ko_lbl_"] p { font-size: 1.3rem !important; font-weight: 700; }
[class*="st-key-ko_lbl_"] { margin-bottom: -0.6rem; }
.ko-copy { width: 2rem; height: 2rem; padding: 0; border-radius: 0.5rem; cursor: pointer; color: inherit;
  background: transparent; border: none; opacity: 0; transition: opacity 0.15s, background 0.15s;
  display: flex; align-items: center; justify-content: center; }
.ko-copy::before { content: ""; width: 1.15rem; height: 1.15rem; background: currentColor; opacity: 0.7;
  -webkit-mask: var(--ko-icon) center / contain no-repeat; mask: var(--ko-icon) center / contain no-repeat;
  --ko-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='8' y='2' width='8' height='4' rx='1'/%3E%3Cpath d='M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2'/%3E%3C/svg%3E"); }
.ko-copy.done::before { opacity: 1; background: #2E7D4F;
  --ko-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 6L9 17l-5-5'/%3E%3C/svg%3E"); }
.st-key-ko_results [data-testid="stColumn"]:hover .ko-copy, .ko-copy:focus-visible, .ko-copy.done { opacity: 1; }
.ko-copy:hover { background: color-mix(in srgb, currentColor 10%, transparent); }
/* 지표 표의 마우스 메뉴(다운로드·검색 등)가 위 카드를 가리지 않게 표 아래쪽에 띄움 */
.st-key-ko_metric_table [data-testid="stElementToolbar"] { top: auto !important; bottom: -2.6rem !important; }
.st-key-ko_metric_table { margin-bottom: 2rem; }
/* 결과 아래 작은 화살표: 누르면 점검 묶음으로 스크롤 */
.ko-next { display: flex; justify-content: center; margin-top: -0.6rem; }
.ko-next button { border: none; background: none; padding: 0.3rem 0.8rem 0.6rem; cursor: pointer; color: inherit; opacity: 0.45;
  transition: opacity 0.15s, transform 0.15s; }
.ko-next button::before { content: ""; display: block; width: 14px; height: 14px;
  border-right: 2px solid currentColor; border-bottom: 2px solid currentColor; transform: rotate(45deg); }
.ko-next button:hover { opacity: 0.9; transform: translateY(2px); }
/* 변환 중 자리: 글줄 모양 막대가 차례로 반짝임(색은 글자색에서 섞어 라이트/다크 공용) */
.ko-writing { box-sizing: border-box; padding: 1.5rem 1.4rem; border-radius: 0.5rem; overflow: hidden;
  background: color-mix(in srgb, currentColor 5%, transparent); }
.ko-line { height: 0.8rem; margin: 0 0 1.25rem; border-radius: 999px; background: color-mix(in srgb, currentColor 14%, transparent);
  animation: ko-pulse 1.4s ease-in-out infinite; }
.ko-gap { height: 1.25rem; }
@keyframes ko-pulse { 0%, 100% { opacity: 0.35; } 50% { opacity: 1; } }
.st-key-ko_pending_label p { font-size: 1.3rem !important; font-weight: 700; }
.st-key-ko_pending_label { margin-bottom: -13px; }
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
    "가을이 되면 많은 사람들이 책을 읽고 싶어 한다. 선선한 바람과 맑은 하늘은 마음을 차분하게 만들어 주고, "
    "길어진 밤은 한 권의 책에 집중할 수 있는 시간을 준다. 이러한 이유로 가을은 예로부터 독서의 계절이라고 불려 왔다.\n\n"
    "그러나 오늘날 사람들은 바쁜 일상 속에서 책을 펼칠 여유를 찾기 어렵다. 휴대전화로 짧은 글과 영상을 보는 시간은 늘어났지만, "
    "한 권의 책을 끝까지 읽는 경험은 점점 줄어들고 있다. 이는 단순히 읽는 양이 줄어들었다는 뜻이 아니라, "
    "하나의 생각을 오래 붙잡고 깊이 들여다보는 힘이 약해지고 있다는 뜻이다.\n\n"
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
    st.space("medium")
    st.subheader(f"{persona} 글과 비교 (참고)", anchor=False)
    try:
        rows = ko_metrics.table(r["source"], r["converted"], user, persona)
    except BaseException as e:
        st.warning(f"지표 계산 오류 ({type(e).__name__}: {e})")
        return
    a, b = rows[0], rows[1]
    ref, bands = rows[2] if len(rows) > 2 else {}, ko_metrics.person_bands(user)
    BAND_TXT = {"green": "보통 범위", "yellow": "범위 안", "red": "범위 밖", None: "기준 없음"}
    raw = {"comma_ec": ko_metrics.comma_after_ec_rate(r["converted"]), "mattr": b["어휘 다양도 (MATTR)"], "sent_len": b["평균 문장 길이 (어절)"]}
    col = {k: ko_metrics.band(raw[k], bands.get(k)) for k in raw}
    k1, k2, k3 = st.columns(3, gap="medium")
    if ref:  # 세 카드 모두 '목표 인물과의 거리', 색은 그 사람 실제 글들의 값 범위 기준(D-110)
        k1.metric("연결어미 뒤 쉼표", f"{abs(b['연결어미 뒤 쉼표 비율'] - ref['연결어미 뒤 쉼표 비율']) * 100:.0f}%p",
                  BAND_TXT[col["comma_ec"]],
                  delta_color=col["comma_ec"] or "gray", delta_arrow="off", border=True,
                  help=f"변환문 {b['연결어미 뒤 쉼표 비율']:.0%} · {persona} {ref['연결어미 뒤 쉼표 비율']:.0%} · 원문 {a['연결어미 뒤 쉼표 비율']:.0%}\n\n"
                       "연결어미(-고, -며, -지만 등) 바로 뒤에  \n쉼표를 찍는 비율이 목표 인물 글과 얼마나 다른지.\n\n"
                       "AI 글은 사람 글보다 이 비율이 높음 (KatFishNet, ACL 2025).")
    if b["어휘 다양도 일치도"] is not None and a["어휘 다양도 일치도"] is not None:
        k2.metric("어휘 다양도 일치도", f"{b['어휘 다양도 일치도']:.2f}",
                  BAND_TXT[col["mattr"]],
                  delta_color=col["mattr"] or "gray", delta_arrow="off", border=True,
                  help=f"원문 {a['어휘 다양도 일치도']:.2f} → 변환문 {b['어휘 다양도 일치도']:.2f}\n\n"
                       "같은 낱말을 다시 쓰는 정도(MATTR)가  \n목표 인물의 실제 글과 얼마나 비슷한지.\n\n"
                       "1 = 그 사람 글의 중앙값, 0 = 그 사람 글에서 보기 드문 정도(이상치 경계)로 다름.\n\n"
                       "AI 글은 사람 글보다 낱말을 덜 반복하는 편이라,  \n절대값이 아니라 목표 인물과의 차이로 봄.")
    if b["문장 길이 일치도"] is not None:
        k3.metric("문장 길이 일치도", f"{b['문장 길이 일치도']:.2f}",
                  BAND_TXT[col["sent_len"]],
                  delta_color=col["sent_len"] or "gray", delta_arrow="off", border=True,
                  help=f"원문 {a['문장 길이 일치도']:.2f} → 변환문 {b['문장 길이 일치도']:.2f}\n\n"
                       "평균 문장 길이(어절 수)가  \n목표 인물의 실제 글과 얼마나 비슷한지.\n\n"
                       "1 = 그 사람 글의 중앙값, 0 = 그 사람 글에서 보기 드문 정도(이상치 경계)로 다름.")
    with st.container(key="ko_metric_table"):
        st.dataframe(pd.DataFrame(rows).set_index("구분").round(3), width="stretch")


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


MANUAL_INPUT = "직접 입력"
DEFAULT_DEMO = "에세이 · 독서의 계절"


@st.cache_data(show_spinner=False)
def load_demo_inputs():
    """데모용 예시 원문 (data/demo/demo_inputs.json). {"장르 · 제목": 본문}"""
    import json
    from src import config
    items = json.loads((config.DATA_DIR / "demo" / "demo_inputs.json").read_text(encoding="utf-8"))
    return {f"{d['genre']} · {d['title']}": d["text"] for d in items}


def _fill_demo(demos):
    name = st.session_state["ko_demo"]
    st.session_state["ko_input"] = "" if name == MANUAL_INPUT else DUMMY_KO_SOURCE if name == DEFAULT_DEMO else demos[name]


def _source_edited():
    st.session_state["ko_demo"] = MANUAL_INPUT  # 예시를 고친 순간부터는 직접 입력한 글


_COPY_JS = """<script>
if (!window.__koCopy) {  // 문서에 한 번만 걸어 두는 클릭 위임(결과가 다시 그려져도 동작)
  window.__koCopy = true;
  document.addEventListener('click', (e) => {
    const b = e.target.closest('.ko-copy');
    if (!b) return;
    const ta = document.querySelectorAll('.st-key-ko_results textarea')[+b.dataset.idx];
    navigator.clipboard.writeText(ta ? ta.value : '').then(() => {
      b.classList.add('done'); b.setAttribute('aria-label', '복사됨');
      setTimeout(() => { b.classList.remove('done'); b.setAttribute('aria-label', '복사'); }, 1500);
    });
  });
}
</script>"""


def _label_with_copy(label, idx):
    """결과 칸 위 줄: 칸 이름 + (마우스를 올리면 나타나는) 복사 버튼. idx = 결과 칸 순서(0 원문, 1 변환문)."""
    with st.container(horizontal=True, vertical_alignment="center", key=f"ko_lbl_{idx}"):
        st.markdown(f"**{label}**", width="stretch")
        st.html(f"<button type='button' class='ko-copy' data-idx='{idx}' aria-label='복사' title='복사'></button>{_COPY_JS}",
                width="content", unsafe_allow_javascript=True)


def render_ko_style(persona, use_dummy, mode):
    from src import config, ko_style
    user = KO_PERSONAS[persona]
    if config.CORPUS_SPLIT and not use_dummy:
        st.warning(f"LINGUA_CORPUS_SPLIT={config.CORPUS_SPLIT} 설정됨. 작가 글 일부만 검색 중. "
                   "Remove-Item Env:LINGUA_CORPUS_SPLIT 실행 후 재시작")

    demos = load_demo_inputs()
    st.session_state.setdefault("ko_input", "")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown("**원문**", width="stretch")
        st.selectbox("예시 원문", [MANUAL_INPUT, DEFAULT_DEMO, *demos], key="ko_demo", on_change=_fill_demo, args=(demos,),
                     label_visibility="collapsed", width=340)
    source = st.text_area("원문", height=300, key="ko_input", label_visibility="collapsed", on_change=_source_edited,
                          placeholder="바꿀 글을 붙여 넣거나, 오른쪽 위에서 예시 글을 골라 보세요.")
    n_chars = len(source.strip())
    st.caption(f"{n_chars:,}자&nbsp;&nbsp;<span style='font-size:0.85em; opacity:0.55'>"
               "문체 옵션을 선택하고 원문을 작성한 뒤 <b>변환</b> 버튼을 눌러 주세요.</span>", unsafe_allow_html=True)
    if 0 < n_chars < 300:  # 문체 판정 점수(1,000자 기준)는 화면에서 뺐으므로 아주 짧을 때만 안내
        st.info("글이 짧으면 문체 차이가 잘 안 드러남 (두세 문단 이상 권장)")

    slot = None
    if st.button("변환", type="primary", width="stretch", disabled=n_chars == 0):
        slot = st.empty()  # 변환하는 동안 결과 자리에 원문 + 반짝이는 빈 칸을 그려 두고, 끝나면 같은 자리에 결과를 갈아 끼운다
        # (자리를 비웠다가 다시 그리면 페이지 길이가 줄었다 늘면서 스크롤이 위로 튐)
        with slot.container():
            st.space("large")
            with st.container(key="ko_pending"):
                st.subheader("변환 결과", anchor=False)
                left, right = st.columns(2, gap="medium")
                with left:
                    st.text_area("원문", source, height=460, disabled=True, key="pending_source")
                with right:
                    with st.container(key="ko_pending_label"):
                        st.markdown(f"**{persona} 문체**")
                    widths = [96, 88, 92, 70, 0, 94, 85, 90, 78, 60, 0, 92, 87, 95, 66]  # 문단처럼 보이는 줄 길이(%), 0 = 빈 줄
                    lines = "".join(f"<div class='ko-line' style='width:{w}%; animation-delay:{i * 0.08:.2f}s'></div>"
                                    if w else "<div class='ko-gap'></div>" for i, w in enumerate(widths))
                    st.html(f"<div class='ko-writing'>{lines}</div>")
                    st.caption(f"{persona} 문체로 다시 쓰는 중…")
            st.html(f"<script>/* {time.time()} */ setTimeout(() => document.querySelector('.st-key-ko_pending')"
                    "?.scrollIntoView({behavior: 'smooth', block: 'start'}), 150);</script>", unsafe_allow_javascript=True)
        try:
            if not use_dummy:
                warm_up_ko()
            r = run_ko_style(source, user, mode, use_dummy)
            r.update(persona=persona, source=source, dummy=use_dummy)
            st.session_state["ko_result"] = r
        except BaseException as e:  # API 키 없음(sys.exit) 포함
            slot.empty()
            st.error("API 키 없음 (.env의 GOOGLE_API_KEY 확인)"
                     if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}")

    r = st.session_state.get("ko_result")
    if not r:
        return

    with slot.container() if slot else st.container():
        st.space("large")
        with st.container(key="ko_results"):  # 칸 높이는 CSS에서 화면 높이에 맞춤
            st.subheader("변환 결과", anchor=False)
            left, right = st.columns(2, gap="medium")
            with left:
                _label_with_copy("원문", 0)
                st.text_area("원문", r["source"], height=460, disabled=True, key="out_source", label_visibility="collapsed")
            with right:
                _label_with_copy(f"{r['persona']} 문체", 1)
                st.text_area(f"{r['persona']} 문체", r["converted"], height=460, disabled=True, label_visibility="collapsed")
                st.caption(("가짜 결과 · " if r["dummy"] else "") + f"{KO_MODES[r['mode']]} · {r['seconds']:.1f}초")
            st.html("<div class='ko-next'><button type='button' id='ko-next-btn' aria-label='점검 결과로 이동'></button></div>"
                    "<script>document.getElementById('ko-next-btn').addEventListener('click', () => document.querySelector("
                    "'.st-key-ko_details')?.scrollIntoView({behavior: 'smooth', block: 'start'}));</script>",
                    unsafe_allow_javascript=True)

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

    try:  # 지표 계산이 실패해도 위 화면이 사라지지 않게
        render_ko_metrics(r, user, persona)
    except BaseException as e:
        st.warning(f"지표 표시 오류 ({type(e).__name__}: {e})")


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
