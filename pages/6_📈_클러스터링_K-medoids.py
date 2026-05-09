import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(layout="wide")
st.title("✅ K-medoids 클러스터링 최종 결과")
st.markdown("K-means, GMM 기각 후 **K-medoids 채택** — 클래스 균형 양호, 이상치에 강건")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ── 데이터 로드 ──
cluster_df = pd.read_csv(os.path.join(OUTPUT_DIR, '07_클러스터링_전체결과.csv'), encoding='utf-8-sig')
dong_rep   = pd.read_csv(os.path.join(OUTPUT_DIR, '07_동별대표_후보.csv'),       encoding='utf-8-sig')

st.markdown("---")

# ══════════════════════════════
# 1. Elbow + 결과 이미지
# ══════════════════════════════
st.subheader("📊 최적 K 결정 및 클러스터링 결과")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Elbow + Silhouette**")
    st.image(os.path.join(OUTPUT_DIR, 'kmedoids_01_elbow.png'), use_container_width=True)
with col2:
    st.markdown("**K-medoids 클러스터링 결과**")
    st.image(os.path.join(OUTPUT_DIR, 'kmedoids_02_result.png'), use_container_width=True)

st.markdown("---")

# ══════════════════════════════
# 2. 클러스터별 평균 특성
# ══════════════════════════════
st.subheader("📋 클러스터별 평균 특성")

features = ['공공시설여부', '면적', '노후도', '펫존거리']
summary  = cluster_df.groupby('클러스터')[features].mean().round(2).reset_index()

fig = go.Figure()
for feat in features:
    fig.add_trace(go.Bar(
        name=feat,
        x=summary['클러스터'].astype(str),
        y=summary[feat],
    ))
fig.update_layout(
    barmode='group',
    title='클러스터별 평균 특성 비교',
    xaxis_title='클러스터',
    height=400,
    plot_bgcolor='white'
)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════
# 3. 누적합 그래프
# ══════════════════════════════
st.subheader("📈 동별 Z점수 누적합 (최적 개수 결정)")
st.image(os.path.join(OUTPUT_DIR, 'kmedoids_03_optimal_n.png'), use_container_width=True)

st.markdown("---")