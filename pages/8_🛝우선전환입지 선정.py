import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(layout="wide")
st.title("🐾 우선 전환 입지 최종 선정")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ── 데이터 로드 ──
best_df  = pd.read_csv(os.path.join(OUTPUT_DIR, '07_전환최적_클러스터.csv'), encoding='utf-8-sig')
final_df = pd.read_csv(os.path.join(OUTPUT_DIR, '07_최종선정.csv'),          encoding='utf-8-sig')

# ══════════════════════════════
# 1. 요약 지표
# ══════════════════════════════
st.subheader(f"🏆 최종 선정 {len(final_df)}곳")

cols = [c for c in ['선택순위','행정동','시설명','면적','노후도','공공시설여부','Z점수','소재지주소']
        if c in final_df.columns]
st.dataframe(final_df[cols], use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════
# 2. 지도 시각화
# ══════════════════════════════
st.subheader("📍 최종 선정 위치 지도")

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("**범례**")
    st.markdown("🔴 **최종 선정 입지**")
    st.markdown("⚫ 전환 최적 클러스터 후보")
    st.markdown("---")
    for _, row in final_df.iterrows():
        st.markdown(f"**{int(row.get('선택순위', ''))}위** {row.get('행정동', '')}  \n{row.get('시설명', '')}")

with col2:
    m = folium.Map(location=[37.4449, 127.1388], zoom_start=12)

    # 전체 후보 (회색)
    for _, row in best_df.iterrows():
        if pd.notna(row.get('위도')) and pd.notna(row.get('경도')):
            folium.CircleMarker(
                location=[row['위도'], row['경도']],
                radius=4,
                color='gray',
                fill=True,
                fill_opacity=0.4,
                tooltip=row.get('시설명', '')
            ).add_to(m)

    # 최종 선정 (빨간 별)
    for _, row in final_df.iterrows():
        if pd.notna(row.get('위도')) and pd.notna(row.get('경도')):
            folium.Marker(
                location=[row['위도'], row['경도']],
                popup=f"{row.get('행정동','')} - {row.get('시설명','')}",
                tooltip=f"⭐ {row.get('시설명', '')}",
                icon=folium.Icon(color='red', icon='star')
            ).add_to(m)

    st_folium(m, width=700, height=500)