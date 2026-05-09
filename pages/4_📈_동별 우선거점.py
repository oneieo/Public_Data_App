import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide")
st.title("📊 최적 동 개수 선정")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

# ── 데이터 로드 ──
df = pd.read_csv(os.path.join(DATA_DIR, 'zscore.csv'), encoding='utf-8-sig')

score_col = 'Z점수'

# 정렬 및 파생변수 생성
df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
df['순위']   = df.index + 1
df['누적합'] = df[score_col].cumsum()
df['기울기'] = df['누적합'].diff()
df.loc[0, '기울기'] = df.loc[0, score_col]

# 최적 지점 계산
threshold   = df['기울기'].max() * 0.05
optimal_idx = df[df['기울기'] < threshold].index[0]
optimal_k   = int(df.loc[optimal_idx, '순위'])

# ══════════════════════════════
# 1. 요약 지표
# ══════════════════════════════
col1, col2, col3 = st.columns(3)
col1.metric("전체 행정동 수", f"{len(df)}개")
col2.metric("최적 선정 동 수", f"{optimal_k}개")
col3.metric("선정 비율", f"{round(optimal_k/len(df)*100, 1)}%")

st.markdown("---")

# ══════════════════════════════
# 2. 누적합 그래프
# ══════════════════════════════
st.subheader("📈 누적합 기반 최적 동 개수 결정")

fig = go.Figure()

# 누적합 곡선
fig.add_trace(go.Scatter(
    x=df['순위'],
    y=df['누적합'],
    mode='lines+markers',
    marker=dict(size=6, color='steelblue'),
    line=dict(width=2),
    name='누적합',
    hovertemplate='순위: %{x}위<br>누적합: %{y:.3f}'
))

# 최적 지점 수직선
fig.add_vline(
    x=optimal_k,
    line_dash='dash',
    line_color='red',
    annotation_text=f'최적 개수: {optimal_k}개',
    annotation_position='top right',
    annotation_font_color='red'
)

# 음영
fig.add_vrect(
    x0=optimal_k - 0.5,
    x1=optimal_k + 0.5,
    fillcolor='gray',
    opacity=0.2,
    line_width=0
)

# 행정동 이름 hover
fig.update_traces(
    customdata=df['행정동'],
    hovertemplate='%{customdata}<br>순위: %{x}위<br>누적합: %{y:.3f}'
)

fig.update_layout(
    xaxis_title='동 개수 (순위)',
    yaxis_title='누적 Z점수',
    height=500,
    plot_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray')
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ══════════════════════════════
# 3. 선정된 상위 동 테이블
# ══════════════════════════════
st.subheader(f"🏆 선정된 상위 {optimal_k}개 행정동")

top_df = df[df['순위'] <= optimal_k][['순위', '행정동', score_col, '누적합']].copy()
top_df[score_col] = top_df[score_col].round(4)
top_df['누적합']  = top_df['누적합'].round(4)

st.dataframe(top_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════
# 4. 전체 순위 테이블
# ══════════════════════════════
with st.expander("📋 전체 행정동 순위 보기"):
    full_df = df[['순위', '행정동', score_col, '누적합', '기울기']].copy()
    full_df[score_col] = full_df[score_col].round(4)
    full_df['누적합']  = full_df['누적합'].round(4)
    full_df['기울기']  = full_df['기울기'].round(4)
    st.dataframe(full_df, use_container_width=True, hide_index=True)