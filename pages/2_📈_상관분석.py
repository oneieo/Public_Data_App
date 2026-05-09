import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide")
st.title("📈 상관분석 결과")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ── 데이터 로드 ──
corr = pd.read_csv(os.path.join(OUTPUT_DIR, '상관분석_결과.csv'), encoding='utf-8-sig', index_col=0)
vif  = pd.read_csv(os.path.join(OUTPUT_DIR, 'VIF_결과.csv'), encoding='utf-8-sig')

# ══════════════════════════════
# 1. 상관계수 히트맵 (Plotly)
# ══════════════════════════════
st.subheader("🔥 변수 간 상관관계 히트맵")

fig_heat = px.imshow(
    corr,
    text_auto='.2f',
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1,
    aspect='auto',
    title='변수 간 상관계수'
)
fig_heat.update_layout(
    height=500,
    coloraxis_colorbar=dict(title='상관계수')
)
st.plotly_chart(fig_heat, use_container_width=True)

# 판단 기준 설명
with st.expander("📌 판단 기준 보기"):
    st.markdown("""
    | 상관계수 | 판단 |
    |---|---|
    | 0.9 이상 | 🔴 변수 제거 또는 통합 검토 |
    | 0.7 ~ 0.9 | 🟡 주의 |
    | 0.7 미만 | 🟢 사용 가능 |
    """)

st.markdown("---")

# ══════════════════════════════
# 2. 상관계수 테이블 + 위험 변수 강조
# ══════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 상관계수 테이블")

    # 0.8 이상 쌍 강조
    def highlight_high_corr(val):
        if isinstance(val, float):
            if abs(val) >= 0.9:
                return 'background-color: #ff4444; color: white'
            elif abs(val) >= 0.7:
                return 'background-color: #ffaa00; color: white'
        return ''

    st.dataframe(
        corr.style.map(highlight_high_corr).format('{:.3f}'),
        use_container_width=True
    )

with col2:
    st.subheader("🔍 다중공선성 위험 쌍")

    danger_pairs = []
    vars_list = corr.columns.tolist()
    for i in range(len(vars_list)):
        for j in range(i+1, len(vars_list)):
            val = abs(corr.iloc[i, j])
            if val >= 0.7:
                status = '🔴 제거 검토' if val >= 0.9 else '🟡 주의'
                danger_pairs.append({
                    '변수1': vars_list[i],
                    '변수2': vars_list[j],
                    '상관계수': round(corr.iloc[i, j], 3),
                    '판단': status
                })

    if danger_pairs:
        st.dataframe(pd.DataFrame(danger_pairs), use_container_width=True)
    else:
        st.success("✅ 다중공선성 위험 변수 없음! 모든 변수 사용 가능")

st.markdown("---")

# ══════════════════════════════
# 3. VIF 바 차트
# ══════════════════════════════
st.subheader("📐 VIF (분산팽창계수)")

# 색상 설정
def get_vif_color(v):
    if v >= 10:
        return '#ff4444'
    elif v >= 5:
        return '#ffaa00'
    else:
        return '#2ecc71'

colors = [get_vif_color(v) for v in vif['VIF']]

fig_vif = go.Figure(go.Bar(
    x=vif['변수'],
    y=vif['VIF'],
    marker_color=colors,
    text=vif['VIF'].round(2),
    textposition='outside'
))
fig_vif.add_hline(y=5,  line_dash='dash', line_color='orange',
                  annotation_text='주의 (5)')
fig_vif.add_hline(y=10, line_dash='dash', line_color='red',
                  annotation_text='위험 (10)')
fig_vif.update_layout(
    title='변수별 VIF',
    yaxis_title='VIF',
    height=400,
    plot_bgcolor='white'
)
st.plotly_chart(fig_vif, use_container_width=True)

# VIF 테이블
vif['판단'] = vif['VIF'].apply(
    lambda v: '✅ 정상' if v < 5 else ('⚠️ 주의' if v < 10 else '❌ 제거 필요')
)
st.dataframe(vif, use_container_width=True, hide_index=True)