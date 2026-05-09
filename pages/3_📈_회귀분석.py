import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

st.set_page_config(layout="wide", page_title="회귀분석 결과")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

X_VARS = ['인구수', '유동인구', '상권활성도', '교통편의도', '공시지가', '놀이터개수']
# ──────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────
@st.cache_data
def load_all():
    zscore   = pd.read_csv(os.path.join(OUTPUT_DIR, '01_Z-score_정규화.csv'),    encoding='utf-8-sig')
    reg      = pd.read_csv(os.path.join(OUTPUT_DIR, '03_다중회귀분석_결과.csv'), encoding='utf-8-sig')
    weight   = pd.read_csv(os.path.join(OUTPUT_DIR, '04_가중치_비율화.csv'),     encoding='utf-8-sig')
    zscore_r = pd.read_csv(os.path.join(OUTPUT_DIR, '05_최종_Z점수_순위.csv'),   encoding='utf-8-sig')
    return zscore, reg, weight, zscore_r

zscore_df, reg_df, weight_df, rank_df = load_all()

# 모델 요약값 파싱
reg_df['변수'] = reg_df['변수'].astype(str).str.strip()
model_row = reg_df[reg_df['변수'] == '[모델 요약]']
coef_df   = reg_df[reg_df['변수'] != '[모델 요약]'].copy()
r2_str     = model_row['유의성'].iloc[0]
adj_r2_str = model_row['유의여부'].iloc[0]
p_f        = float(model_row['p_value'].iloc[0])
r2_val      = float(r2_str.split('=')[1])
adj_r2_val  = float(adj_r2_str.split('=')[1].split('|')[0].strip())
f_val       = float(adj_r2_str.split('F=')[1].strip())

# ══════════════════════════════════════════
# 헤더
# ══════════════════════════════════════════
st.title("📈 회귀분석 결과")
st.caption("반려동물 인프라 수요 Z점수 산출을 위한 다중회귀분석")

st.markdown("---")

# ══════════════════════════════════════════
# 섹션 6. Z-score 정규화 전후 비교
# ══════════════════════════════════════════
st.subheader("📐 Z-score 정규화 전후 분포 비교")

# 원본 데이터는 rank_df의 _원본 컬럼에서 가져옴
origin_cols = {v: f'{v}_원본' for v in X_VARS if f'{v}_원본' in rank_df.columns}
z_cols      = {v: f'{v}_Z값'  for v in X_VARS if f'{v}_Z값'  in rank_df.columns}

vars_available = [v for v in X_VARS if v in origin_cols and v in z_cols]

selected_var = st.selectbox("변수 선택", vars_available)

if selected_var:
    before = rank_df[origin_cols[selected_var]].dropna()
    after  = rank_df[z_cols[selected_var]].dropna()

    fig5 = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'{selected_var} — 정규화 전', f'{selected_var} — Z-score 정규화 후']
    )

    fig5.add_trace(
        go.Histogram(x=before, nbinsx=25, marker_color='#6366f1', opacity=0.75, name='정규화 전'),
        row=1, col=1
    )
    fig5.add_trace(
        go.Histogram(x=after,  nbinsx=25, marker_color='#22c55e', opacity=0.75, name='정규화 후'),
        row=1, col=2
    )

    fig5.update_layout(
        template='plotly_white',
        showlegend=False,
        plot_bgcolor='white',
        height=350,
        margin=dict(t=60, b=20),
    )
    fig5.update_xaxes(gridcolor='#f1f5f9')
    fig5.update_yaxes(gridcolor='#f1f5f9', title_text='빈도')

    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("---")

# ══════════════════════════════════════════
# 섹션 1. 모델 요약 지표
# ══════════════════════════════════════════
st.subheader("🔢 모델 요약")

c1, c2, c3, c4 = st.columns(4)
c1.metric("R²",       f"{r2_val:.4f}")
c2.metric("수정 R²",  f"{adj_r2_val:.4f}")
c3.metric("F 통계량", f"{f_val:.4f}")
c4.metric(
    "F p-value", f"{p_f:.6f}",
    delta="유의 ✅" if p_f < 0.05 else "비유의 ❌",
    delta_color="normal" if p_f < 0.05 else "inverse"
)

st.markdown("---")

# ══════════════════════════════════════════
# 섹션 2. 회귀계수 테이블 + 시각화
# ══════════════════════════════════════════
st.subheader("📋 회귀계수 분석")

col1, col2 = st.columns([1, 2])

with col1:
    # 유의성 컬럼 색상 강조
    def highlight_sig(row):
        if row['유의여부'] == 'O':
            return ['background-color: rgba(99,102,241,0.1)'] * len(row)
        return [''] * len(row)

    display_cols = ['변수', '비표준화_회귀계수', '표준오차', 't통계량', 'p_value', '유의성', '유의여부']
    st.dataframe(
        coef_df[display_cols].style.apply(highlight_sig, axis=1),
        use_container_width=True,
        hide_index=True,
        height=300
    )
    st.caption("* p<0.05  ** p<0.01  *** p<0.001")

with col2:
    # 회귀계수 + 표준오차 에러바 차트
    sig_coef = coef_df[coef_df['변수'] != '절편'].copy()
    sig_coef['색상'] = sig_coef['유의여부'].map({'O': '#6366f1', 'X': '#94a3b8'})
    sig_coef['비표준화_회귀계수'] = pd.to_numeric(sig_coef['비표준화_회귀계수'])
    sig_coef['표준오차']           = pd.to_numeric(sig_coef['표준오차'])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sig_coef['변수'],
        y=sig_coef['비표준화_회귀계수'],
        error_y=dict(
            type='data',
            array=sig_coef['표준오차'].tolist(),
            visible=True,
            color='#475569',
            thickness=1.5,
            width=6
        ),
        marker=dict(
            color=sig_coef['색상'].tolist(),
            cornerradius=5,
            line=dict(color='white', width=1)
        ),
        text=sig_coef['유의성'],
        textposition='outside',
        textfont=dict(size=13, color='#6366f1'),
        name='회귀계수'
    ))

    fig.add_hline(y=0, line_color='#cbd5e1', line_width=1.5)

    fig.update_layout(
        title=dict(text='변수별 회귀계수 (에러바: 표준오차)', font=dict(size=15), x=0.5, xanchor='center'),
        template='plotly_white',
        yaxis_title='회귀계수',
        xaxis_title='',
        showlegend=False,
        plot_bgcolor='white',
        margin=dict(t=60, b=20),
        yaxis=dict(gridcolor='#f1f5f9'),
        height=320
    )
    # 유의/비유의 범례 수동 추가
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color='#6366f1', name='유의 (p<0.05)', showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color='#94a3b8', name='비유의',         showlegend=True))
    fig.update_layout(legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'))

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════
# 섹션 3. 가중치 비율화
# ══════════════════════════════════════════
st.subheader("⚖️ 변수별 가중치")

col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(
        weight_df[['변수', '가중치_절댓값비율(%)', '가중치_부호포함(%)', '방향']],
        use_container_width=True,
        hide_index=True
    )

with col2:
    weight_sorted = weight_df.sort_values('가중치_절댓값비율(%)', ascending=True)
    weight_sorted['방향색'] = weight_sorted['방향'].apply(
        lambda x: '#6366f1' if '높을수록' in x else '#f43f5e'
    )

    fig2 = go.Figure(go.Bar(
        x=weight_sorted['가중치_절댓값비율(%)'],
        y=weight_sorted['변수'],
        orientation='h',
        text=weight_sorted['가중치_절댓값비율(%)'].apply(lambda v: f"{v:.1f}%"),
        textposition='outside',
        marker=dict(
            color=weight_sorted['방향색'].tolist(),
            cornerradius=5,
            line=dict(color='white', width=1)
        ),
    ))

    # 방향 범례
    fig2.add_trace(go.Bar(x=[None], y=[None], orientation='h', marker_color='#6366f1', name='↑ 높을수록 유리', showlegend=True))
    fig2.add_trace(go.Bar(x=[None], y=[None], orientation='h', marker_color='#f43f5e', name='↓ 낮을수록 유리', showlegend=True))

    fig2.update_layout(
        title=dict(text='변수별 가중치 비율 (절댓값 기준)', font=dict(size=15), x=0.5, xanchor='center'),
        template='plotly_white',
        xaxis_title='가중치 (%)',
        yaxis_title='',
        showlegend=True,
        legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
        plot_bgcolor='white',
        margin=dict(t=60, r=80, b=20),
        xaxis=dict(gridcolor='#f1f5f9'),
        height=320
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════
# 섹션 4. 행정동별 최종 Z점수 순위
# ══════════════════════════════════════════
st.subheader("🏆 행정동별 최종 Z점수 순위")

top_n = st.slider("상위 N개 행정동 표시", min_value=5, max_value=len(rank_df), value=15, step=5)

col1, col2 = st.columns([1, 2])

top_df = rank_df.head(top_n).copy()

with col1:
    st.dataframe(
        top_df[['순위', '행정동', 'Z점수', '반려동물등록수']],
        use_container_width=True,
        hide_index=True,
        height=420
    )

with col2:
    fig3 = px.bar(
        top_df.sort_values('Z점수'),
        x='Z점수', y='행정동',
        orientation='h',
        text=top_df.sort_values('Z점수')['Z점수'].apply(lambda v: f"{v:.3f}"),
        color='Z점수',
        color_continuous_scale=['#c7d2fe', '#6366f1', '#312e81'],
        template='plotly_white',
        title=f'상위 {top_n}개 행정동 최종 Z점수'
    )
    fig3.update_traces(textposition='outside', marker=dict(cornerradius=4))
    fig3.update_layout(
        title=dict(x=0.5, xanchor='center', font=dict(size=15)),
        coloraxis_showscale=False,
        yaxis_title='',
        margin=dict(t=60, r=80, b=20),
        xaxis=dict(gridcolor='#f1f5f9'),
        height=420
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════
# 섹션 5. 변수별 기여값 히트맵
# ══════════════════════════════════════════
st.subheader("🗺️ 변수별 기여값 히트맵")

contrib_cols  = [f'{v}_기여값' for v in X_VARS if f'{v}_기여값' in rank_df.columns]
heatmap_input = rank_df.head(top_n).set_index('행정동')[contrib_cols]
heatmap_input.columns = [c.replace('_기여값', '') for c in heatmap_input.columns]

fig4 = px.imshow(
    heatmap_input,
    color_continuous_scale='RdBu',
    color_continuous_midpoint=0,
    aspect='auto',
    text_auto='.2f',
    template='plotly_white',
    title=f'상위 {top_n}개 행정동 변수별 기여값'
)
fig4.update_layout(
    title=dict(x=0.5, xanchor='center', font=dict(size=15)),
    xaxis_title='변수',
    yaxis_title='행정동',
    coloraxis_colorbar=dict(title='기여값'),
    margin=dict(t=60),
    height=max(350, top_n * 28)
)
fig4.update_traces(textfont=dict(size=10))
st.plotly_chart(fig4, use_container_width=True)




    
