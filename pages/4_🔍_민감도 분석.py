import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide")
st.title("🔍 민감도 분석 결과")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ── 데이터 로드 ──
sc_df       = pd.read_csv(os.path.join(OUTPUT_DIR, '06_민감도_시나리오설정.csv'),    encoding='utf-8-sig')
results_df  = pd.read_csv(os.path.join(OUTPUT_DIR, '06_민감도_전체결과.csv'),        encoding='utf-8-sig')
top10_pivot = pd.read_csv(os.path.join(OUTPUT_DIR, '06_민감도_상위10동_순위변동.csv'), encoding='utf-8-sig', index_col=0)
stability   = pd.read_csv(os.path.join(OUTPUT_DIR, '06_민감도_안정성요약.csv'),      encoding='utf-8-sig')

# ══════════════════════════════
# 1. 요약 지표
# ══════════════════════════════
total_sc   = len(sc_df)
stable_cnt = len(stability[stability['안정성'] == '✅ 안정'])
warn_cnt   = len(stability[stability['안정성'] == '⚠️ 주의'])
bad_cnt    = len(stability[stability['안정성'] == '❌ 불안정'])

col1, col2, col3, col4 = st.columns(4)
col1.metric("총 시나리오 수",  f"{total_sc}개")
col2.metric("✅ 안정",        f"{stable_cnt}개")
col3.metric("⚠️ 주의",       f"{warn_cnt}개")
col4.metric("❌ 불안정",      f"{bad_cnt}개")

st.markdown("---")

# ══════════════════════════════
# 2. 안정성 요약 테이블
# ══════════════════════════════
st.subheader("📊 상위 10개 행정동 순위 안정성")

def color_stability(val):
    if val == '✅ 안정':
        return 'background-color: #d4edda; color: #155724'
    elif val == '⚠️ 주의':
        return 'background-color: #fff3cd; color: #856404'
    elif val == '❌ 불안정':
        return 'background-color: #f8d7da; color: #721c24'
    return ''

st.dataframe(
    stability.style.map(color_stability, subset=['안정성']),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ══════════════════════════════
# 3. 행정동별 순위 변동 인터랙티브 그래프
# ══════════════════════════════
st.subheader("📈 행정동별 시나리오 순위 변동")

top10_dongs = stability['행정동'].tolist()
selected_dong = st.selectbox("행정동 선택", top10_dongs)

dong_data  = results_df[results_df['행정동'] == selected_dong].copy()
base_rank  = int(dong_data[dong_data['scenario_id'] == 'BASE']['순위'].values[0])
adj_data   = dong_data[dong_data['scenario_id'] != 'BASE'].copy()

fig = go.Figure()

# 시나리오별 순위 점
fig.add_trace(go.Scatter(
    x=adj_data['scenario_name'],
    y=adj_data['순위'],
    mode='markers+lines',
    marker=dict(size=8, color='green'),
    name='시나리오별 순위',
    hovertemplate='%{x}<br>순위: %{y}위'
))

# 기본 순위 점선
fig.add_hline(
    y=base_rank,
    line_dash='dash',
    line_color='red',
    annotation_text=f'기본순위 {base_rank}위',
    annotation_position='top right'
)

fig.update_layout(
    title=f'{selected_dong} 순위 변동',
    xaxis_title='시나리오',
    yaxis_title='순위',
    yaxis_autorange='reversed',  # 낮은 순위가 위
    height=450,
    xaxis_tickangle=-45,
    plot_bgcolor='white'
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ══════════════════════════════
# 4. 시나리오 × 행정동 순위 히트맵
# ══════════════════════════════
st.subheader("🗺️ 시나리오별 상위 10동 순위 히트맵")

fig_heat = px.imshow(
    top10_pivot,
    text_auto=True,
    color_continuous_scale='RdYlGn_r',
    aspect='auto',
    title='시나리오 × 행정동 순위 (낮을수록 좋음)'
)
fig_heat.update_layout(height=600)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ══════════════════════════════
# 5. 시나리오 설정표
# ══════════════════════════════
with st.expander("📋 시나리오 설정 전체 보기"):
    st.dataframe(sc_df, use_container_width=True, hide_index=True)