import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import os
from folium.plugins import MarkerCluster
import branca.colormap as cm

st.set_page_config(layout="wide")
st.title("🐾 우선 전환 입지 최종 선정")
st.caption("전환 용이성 기준: 공공시설만 대상 (민간시설 제외)")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ── 데이터 로드 ──
best_df  = pd.read_csv(os.path.join(OUTPUT_DIR, '07_전환최적_클러스터.csv'), encoding='utf-8-sig')
final_df = pd.read_csv(os.path.join(OUTPUT_DIR, '07_최종선정.csv'),          encoding='utf-8-sig')

# ── 공공시설만 필터링 ──
public_df = final_df[final_df['공공시설여부'] == 1].copy().reset_index(drop=True)
public_df['선택순위'] = range(1, len(public_df) + 1)

# ══════════════════════════════
# 1. 요약 배너
# ══════════════════════════════
total    = len(final_df)
public_n = len(public_df)
private_n = total - public_n

c1, c2, c3 = st.columns(3)
c1.metric("전체 후보", f"{total}곳")
c2.metric("🔴 공공시설 (전환 대상)", f"{public_n}곳")
c3.metric("⬜ 민간시설 (제외)", f"{private_n}곳")

st.markdown("---")

# ══════════════════════════════
# 2. 최종 선정 테이블
# ══════════════════════════════
st.subheader(f"🏆 전환 대상 공공시설 {public_n}곳")

display_cols = [c for c in ['선택순위', '행정동', '시설명', '면적', '노후도', 'Z점수', '소재지주소']
                if c in public_df.columns]

# 노후도 기준 색상 강조
def highlight_old(row):
    if '노후도' in row.index and row['노후도'] >= 20:
        return ['background-color: rgba(239,68,68,0.1)'] * len(row)
    return [''] * len(row)

st.dataframe(
    public_df[display_cols].style.apply(highlight_old, axis=1),
    use_container_width=True,
    hide_index=True
)
st.caption("🔴 노후도 20년 이상 행 강조")

st.markdown("---")

# ══════════════════════════════
# 3. 시각화 — 면적 vs 노후도
# ══════════════════════════════
st.subheader("📊 선정 시설 분포")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.scatter(
        public_df,
        x='면적', y='노후도',
        size='Z점수' if 'Z점수' in public_df.columns else None,
        color='행정동',
        hover_data=['시설명', '소재지주소'] if '소재지주소' in public_df.columns else ['시설명'],
        text='시설명',
        template='plotly_white',
        title='면적 vs 노후도 (버블 크기: Z점수)'
    )
    fig1.update_traces(
        textposition='top center',
        textfont=dict(size=9),
        marker=dict(opacity=0.8, line=dict(color='white', width=1))
    )
    fig1.update_layout(
        xaxis_title='면적 (㎡)', yaxis_title='노후도 (년)',
        margin=dict(t=50, b=20), height=380,
        showlegend=True
    )
    # 노후도 20년 기준선
    fig1.add_hline(
        y=20, line_dash='dash', line_color='#ef4444',
        annotation_text='노후도 20년', annotation_position='right'
    )
    st.plotly_chart(fig1, use_container_width=True, key="scatter_area_age")

with col2:
    if 'Z점수' in public_df.columns:
        fig2 = px.bar(
            public_df.sort_values('Z점수', ascending=True),
            x='Z점수', y='시설명',
            orientation='h',
            color='Z점수',
            color_continuous_scale=['#c7d2fe', '#6366f1', '#312e81'],
            text=public_df.sort_values('Z점수')['Z점수'].round(2),
            template='plotly_white',
            title='시설별 Z점수 (수요 지수)'
        )
        fig2.update_traces(
            textposition='outside',
            marker=dict(cornerradius=4)
        )
        fig2.update_layout(
            coloraxis_showscale=False,
            xaxis_title='Z점수', yaxis_title='',
            margin=dict(t=50, r=60, b=20),
            xaxis=dict(gridcolor='#f1f5f9'),
            height=380
        )
        st.plotly_chart(fig2, use_container_width=True, key="bar_zscore")

st.markdown("---")

# ══════════════════════════════
# 4. 지도 시각화
# ══════════════════════════════
# st.subheader("📍 최종 선정 위치 지도")

# col_left, col_right = st.columns([1, 3])

# with col_left:
#     st.markdown("**범례**")
#     st.markdown("⭐ **공공시설 (전환 대상)**")
#     st.markdown("⚫ 전체 후보 (참고)")
#     st.markdown("---")
#     for _, row in public_df.iterrows():
#         rank = int(row.get('선택순위', ''))
#         dong = row.get('행정동', '')
#         name = row.get('시설명', '')
#         age  = row.get('노후도', '')
#         area = row.get('면적', '')
#         st.markdown(
#             f"**{rank}위** {dong}  \n"
#             f"📍 {name}  \n"
#             f"🕐 노후도 {age:.0f}년 | 📐 {area:.0f}㎡"
#         )
#         st.markdown("")

# with col_right:
#     # 지도 중심: 공공시설 평균 좌표
#     lat_col = '위도' if '위도' in public_df.columns else None
#     lng_col = '경도' if '경도' in public_df.columns else None

#     if lat_col and lng_col:
#         center_lat = public_df[lat_col].mean()
#         center_lng = public_df[lng_col].mean()
#     else:
#         center_lat, center_lng = 37.4449, 127.1388

#     m = folium.Map(location=[center_lat, center_lng], zoom_start=13,
#                    tiles='CartoDB positron')  # 깔끔한 배경 지도

#     # 전체 후보 (회색 작은 점)
#     for _, row in best_df.iterrows():
#         if pd.notna(row.get('위도')) and pd.notna(row.get('경도')):
#             is_public = row.get('공공시설여부', 0) == 1
#             folium.CircleMarker(
#                 location=[row['위도'], row['경도']],
#                 radius=3,
#                 color='#94a3b8',
#                 fill=True,
#                 fill_opacity=0.3,
#                 tooltip=row.get('시설명', '')
#             ).add_to(m)

#     # 공공시설 최종 선정 (빨간 별 + 번호 팝업)
#     for _, row in public_df.iterrows():
#         if pd.notna(row.get('위도')) and pd.notna(row.get('경도')):
#             rank = int(row.get('선택순위', 0))
#             popup_html = f"""
#             <div style="font-family: sans-serif; min-width: 180px;">
#                 <b style="color:#ef4444;">⭐ {rank}순위</b><br>
#                 <b>{row.get('시설명','')}</b><br>
#                 📍 {row.get('행정동','')}<br>
#                 🕐 노후도: {row.get('노후도', ''):.0f}년<br>
#                 📐 면적: {row.get('면적', ''):.0f}㎡
#             </div>
#             """
#             folium.Marker(
#                 location=[row['위도'], row['경도']],
#                 popup=folium.Popup(popup_html, max_width=220),
#                 tooltip=f"⭐ {rank}순위 | {row.get('시설명', '')}",
#                 icon=folium.Icon(color='red', icon='star', prefix='fa')
#             ).add_to(m)

#             # 순위 번호 레이블
#             folium.Marker(
#                 location=[row['위도'], row['경도']],
#                 icon=folium.DivIcon(
#                     html=f'''<div style="
#                         background:#ef4444; color:white;
#                         border-radius:50%; width:20px; height:20px;
#                         display:flex; align-items:center; justify-content:center;
#                         font-size:11px; font-weight:bold;
#                         border:2px solid white;
#                         box-shadow:0 1px 4px rgba(0,0,0,0.3);
#                         margin-left:10px; margin-top:-30px;
#                     ">{rank}</div>''',
#                     icon_size=(20, 20),
#                     icon_anchor=(0, 0)
#                 )
#             ).add_to(m)

#     st_folium(m, width=750, height=520, key="final_map")

st.subheader("📍 최종 선정 위치 지도")

col_left, col_right = st.columns([1, 3])

with col_left:
    st.markdown("""
    <div style="margin-bottom:12px">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <div style="width:12px;height:12px;border-radius:50%;
                        background:#ef4444;border:2px solid white;
                        box-shadow:0 0 0 2px #ef4444"></div>
            <span style="font-size:13px"><b>공공시설</b> (전환 대상)</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:#94a3b8;opacity:0.6"></div>
            <span style="font-size:13px;color:#64748b">전체 후보</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    for _, row in public_df.iterrows():
        rank = int(row.get('선택순위', 0))
        dong = row.get('행정동', '')
        name = row.get('시설명', '')
        age  = row.get('노후도', 0)
        area = row.get('면적', 0)

        st.markdown(f"""
        <div style="
            border-left: 3px solid #ef4444;
            padding: 8px 12px;
            margin-bottom: 10px;
            border-radius: 0 6px 6px 0;
            background: rgba(239,68,68,0.05);
        ">
            <div style="font-size:11px;color:#ef4444;font-weight:600">
                {rank}순위 · {dong}
            </div>
            <div style="font-size:13px;font-weight:600;margin:2px 0">
                {name}
            </div>
            <div style="font-size:11px;color:#64748b">
                🕐 {int(age)}년 · 📐 {int(area)}㎡
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    center_lat = public_df['위도'].mean()
    center_lon = public_df['경도'].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
        attr='CartoDB',
        prefer_canvas=True
    )

    # 전체 후보 — 작고 연한 회색 점
    for _, row in best_df.iterrows():
        if pd.notna(row.get('위도')) and pd.notna(row.get('경도')):
            folium.CircleMarker(
                location=[row['위도'], row['경도']],
                radius=4,
                color='#94a3b8',
                fill=True,
                fill_color='#94a3b8',
                fill_opacity=0.35,
                weight=1,
                tooltip=folium.Tooltip(
                    row.get('시설명', ''),
                    style="font-family:sans-serif;font-size:12px"
                )
            ).add_to(m)

    # 공공 최종 선정 — 커스텀 HTML 마커
    for _, row in public_df.iterrows():
        if pd.notna(row.get('위도')) and pd.notna(row.get('경도')):
            rank = int(row.get('선택순위', 0))
            name = row.get('시설명', '')
            dong = row.get('행정동', '')
            age  = row.get('노후도', 0)
            area = row.get('면적', 0)
            addr = row.get('소재지주소', '')

            # 번호 뱃지 마커
            icon_html = f"""
            <div style="
                background: #ef4444;
                color: white;
                border-radius: 50%;
                width: 28px;
                height: 28px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: 700;
                font-family: sans-serif;
                border: 2.5px solid white;
                box-shadow: 0 2px 8px rgba(239,68,68,0.5);
                cursor: pointer;
            ">{rank}</div>
            """

            popup_html = f"""
            <div style="
                font-family: sans-serif;
                min-width: 200px;
                padding: 4px 2px;
            ">
                <div style="
                    display:flex;align-items:center;gap:6px;
                    margin-bottom:8px;
                ">
                    <div style="
                        background:#ef4444;color:white;
                        border-radius:50%;width:22px;height:22px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:11px;font-weight:700;flex-shrink:0
                    ">{rank}</div>
                    <b style="font-size:14px">{name}</b>
                </div>
                <div style="font-size:12px;color:#475569;line-height:1.9">
                    📍 {dong}<br>
                    🕐 노후도 {int(age)}년<br>
                    📐 면적 {int(area)} ㎡<br>
                    🗺 {addr}
                </div>
            </div>
            """

            folium.Marker(
                location=[row['위도'], row['경도']],
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=folium.Tooltip(
                    f"<b style='color:#ef4444'>{rank}순위</b> {name}",
                    style="font-family:sans-serif;font-size:12px"
                ),
                icon=folium.DivIcon(
                    html=icon_html,
                    icon_size=(28, 28),
                    icon_anchor=(14, 14),
                )
            ).add_to(m)

    st_folium(m, use_container_width=True, height=540, key="final_map")
    st.caption("🖱️ 마커 클릭 시 상세정보 | 스크롤로 줌 인/아웃")