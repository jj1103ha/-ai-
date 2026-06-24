"""
소나무재선충병 방제 정책결정 지원 대시보드 (Streamlit)
실행:  streamlit run app.py
배포:  GitHub 푸시 후 Streamlit Community Cloud(무료)에 연결, Secrets에 GROQ_API_KEY 등록
"""
import os
import streamlit as st
# 클라우드 배포 시: Streamlit Secrets 에 등록한 키를 환경변수로 전달
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass
import pandas as pd
import decision_tools as T

st.set_page_config(page_title="재선충 방제 의사결정 지원", layout="wide")

st.title("🌲 소나무재선충병 방제 정책결정 지원 시스템")
st.caption("AI는 근거와 선택지를 제시합니다. 최종 판단과 책임은 담당 공무원에게 있습니다. "
           "모든 수치는 산림청 원자료 기반 ML 예측에서 산출됩니다.")

# ---------------- 사이드바: 시나리오 설정 ----------------
with st.sidebar:
    st.header("⚙️ 시나리오 설정")
    sido = st.selectbox("시도 필터", ["전국", "경북", "경남", "울산", "대구", "부산", "전남",
                                   "강원", "경기", "충남", "충북", "전북", "제주", "서울",
                                   "광주", "대전", "세종"], index=0)
    top_n = st.slider("표시 시군구 수", 5, 40, 15)
    st.divider()
    st.subheader("💰 예산 배분")
    budget_eok = st.number_input("총 예산 (억원)", 100, 5000, 1000, step=100)
    unit_cost = st.number_input("감염목 1본 방제단가 (원)", 5000, 50000, 15000, step=1000)
    min_share = st.slider("지역 최소배분 비율 (형평성)", 0.0, 0.05, 0.0, step=0.005)

sido_arg = None if sido == "전국" else sido

# ---------------- 1) 우선순위 ----------------
st.subheader(f"📊 방제 우선순위 — {sido} (차년도 예측 감염목 기준)")
rank = T.get_priority_ranking(top_n, sido_arg)
rdf = pd.DataFrame(rank["items"])
if not rdf.empty:
    show = rdf[["rank", "name", "predicted_infected_next", "recent_infected", "treat_rate"]]
    show.columns = ["순위", "시군구", "예측 감염목(차년도)", "최근 감염목", "방제완료율"]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(show, hide_index=True, use_container_width=True)
    with c2:
        chart = rdf.set_index("name")["predicted_infected_next"].head(min(15, top_n))
        st.bar_chart(chart)
else:
    st.info("해당 시도에 데이터가 없습니다.")

# ---------------- 2) 예산 시나리오 ----------------
st.subheader(f"💰 예산 {budget_eok:,}억원 배분 시나리오")
budget = T.simulate_budget(budget_eok * 100_000_000, unit_cost_won=unit_cost,
                           top_n=top_n, min_share=min_share)
if "allocations" in budget:
    bdf = pd.DataFrame(budget["allocations"])
    bdf["예산(억원)"] = (bdf["budget_won"] / 100_000_000).round(1)
    out = bdf[["name", "predicted_infected_next", "예산(억원)", "treatable_trees", "coverage"]]
    out.columns = ["시군구", "예측 감염목", "배정 예산(억원)", "방제가능 본수", "충당률"]
    st.dataframe(out, hide_index=True, use_container_width=True)
    st.caption("충당률 = 배정 예산으로 방제 가능한 본수 ÷ 예측 감염목. 단가는 근사치이며 실제와 다를 수 있습니다.")

# ---------------- 3) AI 챗봇 ----------------
st.divider()
st.subheader("💬 정책 질의응답 (AI 에이전트)")
st.caption("예: '경북에서 가장 시급한 5곳은?', '예산 1500억이면 어디에?', '울주군 내년 전망은?'")

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if q := st.chat_input("질문을 입력하세요"):
    st.session_state.history.append(("user", q))
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        with st.spinner("근거를 조회하는 중..."):
            try:
                import agent
                ans = agent.ask(q, verb