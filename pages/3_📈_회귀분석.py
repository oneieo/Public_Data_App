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
DATA_DIR   = os.path.join(BASE_DIR, 'data')

X_VARS = ['인구수', '유동인구', '상권활성도', '교통편의도', '공시지가', '놀이터개수']

# ──────────────────────────────────────────
# 유틸: hex → rgba 변환
# ──────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=0.13):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

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
model_row  = reg_df[reg_df['변수'] == '[모델 요약]']
coef_df    = reg_df[reg_df['변수'] != '[모델 요약]'].copy()
r2_str     = model_row['유의성'].iloc[0]
adj_r2_str = model_row['유의여부'].iloc[0]
p_f        = float(model_row['p_value'].iloc[0])
r2_val     = float(r2_str.split('=')[1])
adj_r2_val = float(adj_r2_str.split('=')[1].split('|')[0].strip())
f_val      = float(adj_r2_str.split('F=')[1].strip())

# ══════════════════════════════════════════
# 헤더
# ══════════════════════════════════════════
st.title("📈 회귀분석 결과")
st.caption("반려동물 인프라 수요 Z점수 산출을 위한 다중회귀분석")
st.markdown("---")

# ══════════════════════════════════════════
# 섹션 1. Z-score 정규화 전후 비교
# ══════════════════════════════════════════
st.subheader("📐 Z-score 정규화 전후 분포 비교")

origin_cols    = {v: f'{v}_원본' for v in X_VARS if f'{v}_원본' in rank_df.columns}
z_cols         = {v: f'{v}_Z값'  for v in X_VARS if f'{v}_Z값'  in rank_df.columns}
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
        go.Histogram(x=before, nbinsx=25, marker_color="#cbcbcb", opacity=0.75, name='정규화 전'),
        row=1, col=1
    )
    fig5.add_trace(
        go.Histogram(x=after, nbinsx=25, marker_color="#56AAEE", opacity=0.75, name='정규화 후'),
        row=1, col=2
    )
    fig5.update_layout(
        template='plotly_white', showlegend=False,
        height=350, margin=dict(t=60, b=20),
    )
    fig5.update_xaxes(gridcolor='#f1f5f9')
    fig5.update_yaxes(gridcolor='#f1f5f9', title_text='빈도')
    st.plotly_chart(fig5, use_container_width=True, key="zscore_hist")

st.markdown("---")

# ══════════════════════════════════════════
# 섹션 2. 변수 설명
# ══════════════════════════════════════════
st.subheader("🔎 분석 변수 구성")
st.caption("다중회귀분석에 투입된 독립변수(X) 6개와 종속변수(Y) 1개입니다.")

# ── X 변수 메타 ───────────────────────────
x_meta = [
    ("인구수",     "#6366f1", "👥", "행정동 거주 인구 수",              "인구수"),
    ("유동인구",   "#3b82f6", "🚶", "행정동 내 이동·유동 인구 수",      "유동인구"),
    ("상권활성도", "#f59e0b", "🏪", "동물병원 + 동물약국\n+ 펫샵 합산", "동물병원수 + 동물약국수 + 펫샵수"),
    ("교통편의도", "#10b981", "🚌", "버스정류장 수\n+ 지하철역 수 합산", "버스정류장수 + 역개수"),
    ("공시지가",   "#ef4444", "🏘️", "행정동 평균\n공시지가 (원/㎡)",    "공시지가_평균"),
    ("놀이터개수", "#8b5cf6", "🛝", "행정동 내\n어린이 놀이터 수",      "놀이터개수"),
]

# ── Y 변수 메타 ───────────────────────────
y_color  = "#0ea5e9"
y_icon   = "🐾"
y_name   = "Y (수요/공급 비율)"
y_desc   = "반려동물 등록 수 ÷ (펫파크 수 + 1)"
y_detail = "값이 클수록 펫파크 공급 부족 → 전환 필요성 높음"

# ─────────────────────────────────────────
# (A) X 변수 카드 6개
# ─────────────────────────────────────────
st.markdown("##### 📥 독립변수 X — 6개")

x_cols = st.columns(6)
for col, (name, color, icon, desc, _) in zip(x_cols, x_meta):
    col.markdown(
        f"""
        <div style="
            background: linear-gradient(160deg, {color}1a, {color}06);
            border: 1.8px solid {color}66;
            border-radius: 12px;
            padding: 14px 8px 12px;
            text-align: center;
            height: 148px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 4px;
        ">
            <div style="font-size:28px; line-height:1">{icon}</div>
            <div style="
                font-weight:700; font-size:13px;
                color:{color}; margin-top:6px;
            ">{name}</div>
            <div style="
                font-size:10.5px; color:#64748b;
                line-height:1.45; margin-top:2px;
                white-space: pre-line;
            ">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# (B) X 수식 테이블 ➜ Y 카드
# ─────────────────────────────────────────
st.markdown("##### 📤 종속변수 Y — 1개")

left, mid, right = st.columns([4, 0.5, 2])

with left:
    formula_rows = [(name, f"← {formula}") for name, color, icon, desc, formula in x_meta]
    row_colors   = [color for _, color, _, _, _ in x_meta]

    fig_formula = go.Figure(go.Table(
        columnwidth=[120, 260],
        header=dict(
            values=["<b>변수</b>", "<b>원본 컬럼 / 계산식</b>"],
            fill_color="#1e293b",
            font=dict(color="white", size=12),
            align="center",
            height=34,
        ),
        cells=dict(
            values=[
                [r[0] for r in formula_rows],
                [r[1] for r in formula_rows],
            ],
            fill_color=[
                [hex_to_rgba(c, alpha=0.18) for c in row_colors],  # ✅ rgba 변환
                ["#f8fafc"] * 6,
            ],
            font=dict(color="#1e293b", size=12),
            align=["center", "left"],
            height=32,
            line_color="#e2e8f0",
        ),
    ))
    fig_formula.update_layout(
        margin=dict(t=10, b=10, l=0, r=0),
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_formula, use_container_width=True, key="x_formula_table")

with mid:
    st.markdown(
        """
        <div style="
            display:flex; align-items:center;
            justify-content:center; height:240px;
            font-size:36px; color:#94a3b8;
        ">➜</div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(160deg, {y_color}1a, {y_color}06);
            border: 2px solid {y_color}88;
            border-radius: 14px;
            padding: 22px 18px;
            text-align: center;
            height: 240px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 8px;
            box-shadow: 0 2px 12px {y_color}22;
        ">
            <div style="font-size:38px">{y_icon}</div>
            <div style="
                font-weight:800; font-size:14px;
                color:{y_color}; margin-top:4px;
                line-height:1.4;
            ">{y_name}</div>
            <div style="
                font-size:12px; font-weight:600;
                color:#0f172a; margin-top:2px;
                background:{y_color}18;
                border-radius:6px; padding:5px 8px;
            ">{y_desc}</div>
            <div style="font-size:11px; color:#64748b; line-height:1.5; margin-top:2px">
                {y_detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# (C) 파생변수 설명 expander
# ─────────────────────────────────────────
with st.expander("🔧 파생변수 생성 방법 자세히 보기"):
    st.markdown("""
| 파생변수 | 계산식 | 사용 데이터 |
|:--------:|:------:|:-----------|
| **상권활성도** | 동물병원수 + 동물약국수 + 펫샵수 | final_animal_hospital · final_animal_permercy · pet_shop |
| **교통편의도** | 버스정류장수 + 지하철역수 | bus_count · subway_station_count |
| **Y** | 반려동물등록수 ÷ (펫파크수 + 1) | final_pet_registration · final_pet_park_count |
    """)
    st.info("**+1 보정**: 펫파크가 0개인 동에서 분모=0이 되는 문제를 방지합니다.")

st.markdown("---")

# ══════════════════════════════════════════
# 섹션 3. 모델 요약 지표
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
# 섹션 4. 회귀계수 테이블 + 시각화
# ══════════════════════════════════════════
st.subheader("📋 회귀계수 분석")

col1, col2 = st.columns([1, 2])

with col1:
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
    sig_coef = coef_df[coef_df['변수'] != '절편'].copy()
    sig_coef['색상'] = sig_coef['유의여부'].map({'O': "#A8ADFF", 'X': '#cbcbcb'})
    sig_coef['비표준화_회귀계수'] = pd.to_numeric(sig_coef['비표준화_회귀계수'])
    sig_coef['표준오차']           = pd.to_numeric(sig_coef['표준오차'])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sig_coef['변수'],
        y=sig_coef['비표준화_회귀계수'],
        error_y=dict(
            type='data',
            array=sig_coef['표준오차'].tolist(),
            visible=True, color='#475569', thickness=1.5, width=6
        ),
        marker=dict(
            color=sig_coef['색상'].tolist(),
            cornerradius=5, line=dict(color='white', width=1)
        ),
        text=sig_coef['유의성'],
        textposition='outside',
        textfont=dict(size=13, color='#A8ADFF'),
        name='회귀계수'
    ))
    fig.add_hline(y=0, line_color='#cbd5e1', line_width=1.5)
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#A8ADFF", name='유의 (p<0.05)', showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#cbcbcb", name='비유의',         showlegend=True))
    fig.update_layout(
        title=dict(text='변수별 회귀계수 (에러바: 표준오차)', font=dict(size=15), x=0.5, xanchor='center'),
        template='plotly_white',
        yaxis_title='회귀계수', xaxis_title='',
        showlegend=True,
        legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
        margin=dict(t=60, b=20),
        yaxis=dict(gridcolor='#f1f5f9'),
        height=320
    )
    st.plotly_chart(fig, use_container_width=True, key="coef_bar")

st.markdown("---")

# ══════════════════════════════════════════
# 섹션 5. 가중치 비율화
# ══════════════════════════════════════════
st.subheader("⚖️ 변수별 가중치")

col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(
        weight_df[['변수', '가중치_절댓값비율(%)', '가중치_부호포함(%)', '방향']],
        use_container_width=True, hide_index=True
    )

with col2:
    weight_sorted = weight_df.sort_values('가중치_절댓값비율(%)', ascending=True)
    weight_sorted['방향색'] = weight_sorted['방향'].apply(
        lambda x: "#4cc1f8" if '높을수록' in x else "#f580a1"
    )
    fig2 = go.Figure(go.Bar(
        x=weight_sorted['가중치_절댓값비율(%)'],
        y=weight_sorted['변수'],
        orientation='h',
        text=weight_sorted['가중치_절댓값비율(%)'].apply(lambda v: f"{v:.1f}%"),
        textposition='outside',
        marker=dict(
            color=weight_sorted['방향색'].tolist(),
            cornerradius=5, line=dict(color='white', width=1)
        ),
    ))
    fig2.add_trace(go.Bar(x=[None], y=[None], orientation='h', marker_color="#57bfef", name='↑ 높을수록 유리', showlegend=True))
    fig2.add_trace(go.Bar(x=[None], y=[None], orientation='h', marker_color='#f580a1', name='↓ 낮을수록 유리', showlegend=True))
    fig2.update_layout(
        title=dict(text='변수별 가중치 비율 (절댓값 기준)', font=dict(size=15), x=0.5, xanchor='center'),
        template='plotly_white',
        xaxis_title='가중치 (%)', yaxis_title='',
        showlegend=True,
        legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
        margin=dict(t=60, r=80, b=20),
        xaxis=dict(gridcolor='#f1f5f9'),
        height=320
    )
    st.plotly_chart(fig2, use_container_width=True, key="weight_bar")

st.markdown("---")

# ══════════════════════════════════════════
# 섹션 6. 행정동별 최종 Z점수 순위
# ══════════════════════════════════════════
st.subheader("🏆 행정동별 최종 Z점수 순위")

top_n  = st.slider("상위 N개 행정동 표시", min_value=5, max_value=len(rank_df), value=15, step=5)
top_df = rank_df.head(top_n).copy()

col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(
        top_df[['순위', '행정동', 'Z점수', '반려동물등록수']],
        use_container_width=True, hide_index=True, height=420
    )

with col2:
    fig3 = px.bar(
        top_df.sort_values('Z점수'),
        x='Z점수', y='행정동', orientation='h',
        text=top_df.sort_values('Z점수')['Z점수'].apply(lambda v: f"{v:.3f}"),
        color='Z점수',
        color_continuous_scale=['#c7d2fe', '#6366f1', '#312e81'],
        template='plotly_white',
        title=f'상위 {top_n}개 행정동 최종 Z점수'
    )
    fig3.update_traces(textposition='outside', marker=dict(cornerradius=4))
    fig3.update_layout(
        title=dict(x=0.5, xanchor='center', font=dict(size=15)),
        coloraxis_showscale=False, yaxis_title='',
        margin=dict(t=60, r=80, b=20),
        xaxis=dict(gridcolor='#f1f5f9'),
        height=420
    )
    st.plotly_chart(fig3, use_container_width=True, key="rank_bar")

st.markdown("---")

# ══════════════════════════════════════════
# 섹션 7. 변수별 기여값 히트맵
# ══════════════════════════════════════════
st.subheader("🗺️ 변수별 기여값 히트맵")

contrib_cols  = [f'{v}_기여값' for v in X_VARS if f'{v}_기여값' in rank_df.columns]
heatmap_input = rank_df.head(top_n).set_index('행정동')[contrib_cols]
heatmap_input.columns = [c.replace('_기여값', '') for c in heatmap_input.columns]

fig4 = px.imshow(
    heatmap_input,
    color_continuous_scale='RdBu',
    color_continuous_midpoint=0,
    aspect='auto', text_auto='.2f',
    template='plotly_white',
    title=f'상위 {top_n}개 행정동 변수별 기여값'
)
fig4.update_layout(
    title=dict(x=0.5, xanchor='center', font=dict(size=15)),
    xaxis_title='변수', yaxis_title='행정동',
    coloraxis_colorbar=dict(title='기여값'),
    margin=dict(t=60),
    height=max(350, top_n * 28)
)
fig4.update_traces(textfont=dict(size=10))
st.plotly_chart(fig4, use_container_width=True, key="contrib_heatmap")