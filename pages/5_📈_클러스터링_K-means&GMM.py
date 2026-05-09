import streamlit as st
import os

st.set_page_config(layout="wide")
st.title("🔵 클러스터링 비교 분석 (K-means & GMM)")
st.markdown("K-means와 GMM 두 방법을 시도했으나 **클래스 불균형** 문제로 최종 기각")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

st.markdown("---")

# ══════════════════════════════
# K-means
# ══════════════════════════════
st.subheader("❌ K-means 클러스터링")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Elbow + Silhouette**")
    st.image(os.path.join(OUTPUT_DIR, 'kmeans_01_elbow.png'), use_container_width=True)
with col2:
    st.markdown("**클러스터링 결과**")
    st.image(os.path.join(OUTPUT_DIR, 'kmeans_02_result.png'), use_container_width=True)

with st.expander("📌 K-means 기각 이유"):
    st.error("클래스 불균형 심함 → 분석 신뢰성 낮음")
    st.markdown("""
    - 군집 간 크기 차이가 과도하게 큼
    - 이상치에 민감한 평균 기반 방식
    - 소규모 데이터에 불리
    """)

st.markdown("---")

# ══════════════════════════════
# GMM
# ══════════════════════════════
st.subheader("❌ GMM 클러스터링")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**밀도 분포 (K 결정)**")
    st.image(os.path.join(OUTPUT_DIR, 'gmm_01_density.png'), use_container_width=True)
with col2:
    st.markdown("**클러스터링 결과**")
    st.image(os.path.join(OUTPUT_DIR, 'gmm_02_result.png'), use_container_width=True)

with st.expander("📌 GMM 기각 이유"):
    st.error("클래스 불균형 심함 → 분석 신뢰성 낮음")
    st.markdown("""
    - 확률 기반이나 군집 크기 불균형 해결 못함
    - 소규모 데이터에서 과적합 위험
    """)

st.markdown("---")
st.info("💡 두 방법 모두 기각 → **K-medoids 채택**")