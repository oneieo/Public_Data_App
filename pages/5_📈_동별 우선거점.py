import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
import json
import os

st.set_page_config(layout="wide")
st.title("📊 최적 동 개수 선정")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

# ── 데이터 로드 ──
df = pd.read_csv(os.path.join(DATA_DIR, 'zscore.csv'), encoding='utf-8-sig')

# ── GeoJSON 로드 및 성남시 필터링 ──
@st.cache_data
def load_geojson():
    with open(os.path.join(DATA_DIR, 'hangjeongdong_경기도.geojson'), encoding='utf-8') as f:
        raw = json.load(f)

    # 성남시만 필터
    seongnam_features = [
        feat for feat in raw['features']
        if '성남' in feat['properties']['adm_nm']
    ]

    # 행정동명만 파싱해서 property에 추가 (매칭용)
    for feat in seongnam_features:
        full_name = feat['properties']['adm_nm']  # "경기도 성남시수정구 신흥1동"
        feat['properties']['dong'] = full_name.split()[-1]  # "신흥1동"

    return {'type': 'FeatureCollection', 'features': seongnam_features}

geo = load_geojson()

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
c1, c2, c3 = st.columns(3)
c1.metric("전체 행정동 수", f"{len(df)}개")
c2.metric("최적 선정 동 수", f"{optimal_k}개")
c3.metric("선정 비율", f"{round(optimal_k/len(df)*100, 1)}%")

st.markdown("---")

# ══════════════════════════════
# 2. 누적합 그래프 (Plotly)
# ══════════════════════════════
st.subheader("📈 누적합 기반 최적 동 개수 결정")

fig = go.Figure()

# 선택 영역 음영
fig.add_vrect(
    x0=0.5, x1=optimal_k,
    fillcolor='rgba(99,102,241,0.08)',
    line_width=0,
    annotation_text='선정 구간',
    annotation_position='top left',
    annotation_font=dict(color='#6366f1', size=11)
)

# 누적합 곡선 — 그라데이션 효과 (fill)
fig.add_trace(go.Scatter(
    x=df['순위'],
    y=df['누적합'],
    mode='lines',
    line=dict(color='#6366f1', width=2.5),
    fill='tozeroy',
    fillcolor='rgba(99,102,241,0.08)',
    name='누적합',
    customdata=np.stack([df['행정동'], df[score_col]], axis=-1),
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>'
        '순위: %{x}위<br>'
        'Z점수: %{customdata[1]:.3f}<br>'
        '누적합: %{y:.3f}'
        '<extra></extra>'
    )
))

# 마커 — 선정 구간만 강조
fig.add_trace(go.Scatter(
    x=df[df['순위'] <= optimal_k]['순위'],
    y=df[df['순위'] <= optimal_k]['누적합'],
    mode='markers',
    marker=dict(size=6, color='#6366f1', opacity=0.7),
    showlegend=False,
    hoverinfo='skip'
))

# 나머지 구간 마커 (연하게)
fig.add_trace(go.Scatter(
    x=df[df['순위'] > optimal_k]['순위'],
    y=df[df['순위'] > optimal_k]['누적합'],
    mode='markers',
    marker=dict(size=5, color='#94a3b8', opacity=0.5),
    showlegend=False,
    hoverinfo='skip'
))

# 최적 기준선
fig.add_vline(
    x=optimal_k,
    line_dash='dash',
    line_color='#ef4444',
    line_width=1.8,
    annotation_text=f'최적 {optimal_k}개',
    annotation_position='top right',
    annotation_font=dict(color='#ef4444', size=12)
)

# 최적 지점 마커
fig.add_trace(go.Scatter(
    x=[optimal_k],
    y=[df.loc[optimal_idx, '누적합']],
    mode='markers',
    marker=dict(size=12, color='#ef4444', symbol='diamond',
                line=dict(color='white', width=2)),
    name=f'최적 지점 ({optimal_k}위)',
    hovertemplate=f'최적 지점: {optimal_k}개<extra></extra>'
))

fig.update_layout(
    template='plotly_white',
    xaxis=dict(
        title='동 개수 (순위)',
        gridcolor='#f1f5f9',
        showline=True,
        linecolor='#e2e8f0'
    ),
    yaxis=dict(
        title='누적 Z점수',
        gridcolor='#f1f5f9',
        showline=True,
        linecolor='#e2e8f0'
    ),
    legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
    margin=dict(t=40, b=20),
    height=440,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True, key="cumsum_chart")

st.markdown("---")

# ══════════════════════════════
# 3. 행정동 순위 지도
# ══════════════════════════════
st.subheader("🗺️ 행정동별 Z점수 순위 지도")

col_left, col_right = st.columns([1, 2])

with col_left:
    # 색상 범례 설명
    st.markdown("""
    <div style="font-size:13px;line-height:2">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
            <div style="width:16px;height:16px;border-radius:3px;background:#312e81"></div>
            <span>상위권 (1 ~ {a}위)</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
            <div style="width:16px;height:16px;border-radius:3px;background:#6366f1"></div>
            <span>중위권</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">
            <div style="width:16px;height:16px;border-radius:3px;background:#e0e7ff"></div>
            <span>하위권 ({b}위 ~)</span>
        </div>
    </div>
    """.format(a=optimal_k, b=optimal_k+1), unsafe_allow_html=True)

    st.markdown("**선정된 상위 동**")
    top_df = df[df['순위'] <= optimal_k][['순위', '행정동', score_col]].copy()
    for _, row in top_df.iterrows():
        rank  = int(row['순위'])
        dong  = row['행정동']
        score = row[score_col]
        st.markdown(f"""
        <div style="
            display:flex;align-items:center;gap:8px;
            padding:5px 8px;margin-bottom:4px;
            border-radius:6px;background:rgba(99,102,241,0.08);
            font-size:12px;
        ">
            <div style="
                background:#6366f1;color:white;
                border-radius:50%;width:20px;height:20px;
                display:flex;align-items:center;justify-content:center;
                font-size:10px;font-weight:700;flex-shrink:0
            ">{rank}</div>
            <span style="font-weight:600">{dong}</span>
            <span style="color:#6366f1;margin-left:auto">{score:.3f}</span>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    m = folium.Map(
        location=[37.4449, 127.1388],
        zoom_start=12,
        tiles=None   # ← 배경 타일 없음 = 완전 흰색
    )

    rank_dict  = dict(zip(df['행정동'], df['순위']))
    score_dict = dict(zip(df['행정동'], df[score_col]))

    # ── 행정동 경계 색칠 ──
    folium.GeoJson(
        geo,
        name='행정동',
        style_function=lambda feat: {
            'fillColor':   '#6366f1' if rank_dict.get(feat['properties']['dong'], 999) <= optimal_k else 'white',
            'color':       '#94a3b8',
            'weight':      1.2,
            'fillOpacity': 0.75 if rank_dict.get(feat['properties']['dong'], 999) <= optimal_k else 0.3,
        },
        highlight_function=lambda feat: {
            'fillOpacity': 0.95,
            'weight':      2,
            'color':       '#4338ca'
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['dong'],
            aliases=['행정동'],
            style="font-family:sans-serif;font-size:12px;background:white;padding:6px 10px;border-radius:6px"
        )
    ).add_to(m)

    # ── 행정동명 + 순위 레이블 ──
    for feat in geo['features']:
        dong_name = feat['properties']['dong']
        rank      = rank_dict.get(dong_name, None)

        # 폴리곤 중심 계산
        coords = feat['geometry']['coordinates']
        if feat['geometry']['type'] == 'Polygon':
            pts = coords[0]
        else:
            pts = coords[0][0]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        is_selected = rank is not None and rank <= optimal_k

        if is_selected:
            # 선정된 동 — 흰색 동명 + 순위 뱃지
            label_html = f"""
            <div style="
                display:flex;flex-direction:column;
                align-items:center;gap:2px;
                pointer-events:none;
            ">
                <div style="
                    background:#312e81;color:white;
                    border-radius:50%;width:18px;height:18px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:9px;font-weight:700;
                    border:1.5px solid white;
                    box-shadow:0 1px 3px rgba(0,0,0,0.3);
                ">{rank}</div>
                <div style="
                    color:white;
                    font-size:10px;
                    font-weight:600;
                    font-family:sans-serif;
                    text-shadow:0 1px 2px rgba(0,0,0,0.5);
                    white-space:nowrap;
                ">{dong_name}</div>
            </div>
            """
        else:
            # 나머지 동 — 회색 동명만
            label_html = f"""
            <div style="
                color:#64748b;
                font-size:9px;
                font-weight:500;
                font-family:sans-serif;
                white-space:nowrap;
                pointer-events:none;
            ">{dong_name}</div>
            """

        folium.Marker(
            location=[cy, cx],
            icon=folium.DivIcon(
                html=label_html,
                icon_size=(80, 40),
                icon_anchor=(40, 20)
            )
        ).add_to(m)

    st_folium(m, use_container_width=True, height=540, key="rank_map")

st.markdown("---")

# ══════════════════════════════
# 4. 선정된 상위 동 테이블
# ══════════════════════════════
st.subheader(f"🏆 선정된 상위 {optimal_k}개 행정동")

top_df = df[df['순위'] <= optimal_k][['순위', '행정동', score_col, '누적합']].copy()
top_df[score_col] = top_df[score_col].round(4)
top_df['누적합']  = top_df['누적합'].round(4)
st.dataframe(top_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════
# 5. 전체 순위 테이블
# ══════════════════════════════
with st.expander("📋 전체 행정동 순위 보기"):
    full_df = df[['순위', '행정동', score_col, '누적합', '기울기']].copy()
    for col in [score_col, '누적합', '기울기']:
        full_df[col] = full_df[col].round(4)
    st.dataframe(full_df, use_container_width=True, hide_index=True)