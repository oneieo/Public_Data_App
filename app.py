import streamlit as st

st.set_page_config(
    page_title="성남시 어린이 놀이터의 펫존 전환 입지 분석_오지선다",
    page_icon="🐾",
    layout="wide"
)

# 헤더
st.markdown("""
    <div style="text-align:center; padding: 40px 0 10px;">
        <p style=" font-size:20px; font-weight:600; letter-spacing:2px;">
            2026 성남시 공공데이터 활용 시각화 경진대회
        </p>
        <h1 style="color:#16A34A; font-size:40px; font-weight:800;">
            성남시 어린이 놀이터의 펫존 전환 입지 분석
        </h1>
        <p style="color:#64748b; font-size:16px;">
            아이 없는 놀이터, 반려동물의 새로운 쉼터로
        </p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# 분석 흐름 카드
st.subheader("📌 분석 흐름")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🔍 주제 선정")
    st.caption("반려동물 1,500만 시대\n저출생으로 유휴화된 놀이터 → 펫존 전환")

    
    st.markdown("#### 🔵 2차 놀이터 선정")
    st.caption("PCA → K-means/GMM 기각\n→ K-medoids 채택 → 19개 후보")

with col2:
    st.markdown("#### 🧹 데이터 가공")
    st.caption("인구 / 교통 / 반려동물 / 상권\n11개 데이터셋 수집 및 정제")

    st.markdown("#### 📍 입지 선정")
    st.caption("민간 제외 → 공공시설 9곳\n노후도 최상 우선 전환 대상 확정")

with col3:
    st.markdown("#### 📊 1차 행정동 선별")
    st.caption("상관분석 → VIF → 다중회귀분석\n가중치 산출 → Z점수 → 23개 동 선별")

    st.markdown("#### 🏆 기대효과")
    st.caption("성남시 · 애견인 · 소상공인\n3개 주체별 기대효과")

st.divider()
st.info("👈 왼쪽 사이드바에서 분석 단계를 선택하세요.")