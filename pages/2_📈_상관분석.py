import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide")
st.title("📈 상관분석 결과")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

# ══════════════════════════════════════════
# 0. 변수 설명 섹션 (신규)
# ══════════════════════════════════════════
st.subheader("📌 분석 변수 설명")

# ── X 변수 카드 ───────────────────────────

x_meta = [
    ("인구수",      "#6366f1", "👥", "행정동 거주 인구 수"),
    ("유동인구",    "#3b82f6", "🚶", "행정동 내 이동 인구 수"),
    ("상권활성도",  "#f59e0b", "🏪", "동물병원 + 동물약국 + 펫샵 수 합산"),
    ("교통편의도",  "#10b981", "🚌", "버스정류장 수 + 지하철역 수 합산"),
    ("공시지가",    "#ef4444", "🏘️", "행정동 평균 공시지가 (원/㎡)"),
    ("놀이터개수",  "#8b5cf6", "🛝", "행정동 내 어린이 놀이터 수"),
]

cols = st.columns(6)
for col, (name, color, icon, desc) in zip(cols, x_meta):
    col.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}18, {color}08);
            border: 1.5px solid {color}55;
            border-radius: 10px;
            padding: 14px 10px;
            text-align: center;
            height: 130px;
        ">
            <div style="font-size:26px">{icon}</div>
            <div style="font-weight:700; font-size:13px; color:{color}; margin:4px 0">{name}</div>
            <div style="font-size:11px; color:#64748b; line-height:1.4">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── 파생변수 생성 방법 expander ───────────
with st.expander("🔧 파생변수 생성 방법 보기"):
    st.markdown("""
    | 파생변수 | 계산식 | 설명 |
    |:--------:|:------:|:-----|
    | **상권활성도** | 동물병원수 + 동물약국수 + 펫샵수 | 반려동물 관련 상업 인프라 밀도 |
    | **교통편의도** | 버스정류장수 + 지하철역수 | 접근성 지표 |
    | **Y (수요/공급비)** | 반려동물등록수 ÷ (펫파크수 + 1) | 펫파크 전환 필요성 지수 |
    """)

    st.info("**+1 보정**: 펫파크가 0개인 동의 분모가 0이 되는 문제를 방지하기 위해 +1 처리")

st.markdown("---")

# ── 변수별 기초통계 (데이터 있을 때만) ───
def read_csv_safe(path):
    for enc in ['utf-8-sig', 'cp949', 'euc-kr', 'utf-8']:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None

@st.cache_data
def load_raw():
    files = {
        'population': 'final_population.csv',
        'move_pop':   'move_population.csv',
        'hospital':   'final_animal_hospital.csv',
        'pharmacy':   'final_animal_permercy.csv',
        'pet_shop':   'pet_shop.csv',
        'bus':        'bus_count.csv',
        'subway':     'subway_station_count.csv',
        'money':      'money.csv',
        'playground': 'final_playground_count.csv',
        'pet_reg':    'final_pet_registration.csv',
        'pet_park':   'final_pet_park_count.csv',
    }
    dfs = {}
    for key, fname in files.items():
        df = read_csv_safe(os.path.join(DATA_DIR, fname))
        if df is not None:
            dfs[key] = df
    return dfs

dfs = load_raw()

if len(dfs) == len(['population','move_pop','hospital','pharmacy',
                     'pet_shop','bus','subway','money',
                     'playground','pet_reg','pet_park']):
    try:
        hospital_cnt = dfs['hospital'].groupby('행정동').size().reset_index(name='동물병원수')
        pharmacy_cnt = dfs['pharmacy'].groupby('행정동').size().reset_index(name='동물약국수')

        move_col  = [c for c in dfs['move_pop'].columns  if c != '행정동'][0]
        shop_col  = [c for c in dfs['pet_shop'].columns  if c != '행정동'][0]
        bus_col   = [c for c in dfs['bus'].columns       if c != '행정동'][0]
        sub_col   = [c for c in dfs['subway'].columns    if c != '행정동'][0]
        play_col  = [c for c in dfs['playground'].columns if c != '행정동'][0]
        park_col  = [c for c in dfs['pet_park'].columns  if c != '행정동'][0]

        df_all = dfs['population'][['행정동', '인구수']].copy()
        df_all = df_all.merge(dfs['move_pop'][['행정동', move_col]].rename(columns={move_col: '유동인구'}), on='행정동', how='left')
        df_all = df_all.merge(hospital_cnt, on='행정동', how='left')
        df_all = df_all.merge(pharmacy_cnt, on='행정동', how='left')
        df_all = df_all.merge(dfs['pet_shop'][['행정동', shop_col]].rename(columns={shop_col: '펫샵수'}), on='행정동', how='left')
        df_all = df_all.merge(dfs['bus'][['행정동', bus_col]].rename(columns={bus_col: '버스정류장수'}), on='행정동', how='left')
        df_all = df_all.merge(dfs['subway'][['행정동', sub_col]].rename(columns={sub_col: '역개수'}), on='행정동', how='left')
        df_all = df_all.merge(dfs['money'][['행정동', '공시지가_평균']].rename(columns={'공시지가_평균': '공시지가'}), on='행정동', how='left')
        df_all = df_all.merge(dfs['playground'][['행정동', play_col]].rename(columns={play_col: '놀이터개수'}), on='행정동', how='left')
        df_all = df_all.merge(dfs['pet_reg'][['행정동', '합계']].rename(columns={'합계': '반려동물등록수'}), on='행정동', how='left')
        df_all = df_all.merge(dfs['pet_park'][['행정동', park_col]].rename(columns={park_col: '펫파크수'}), on='행정동', how='left')
        df_all = df_all.fillna(0)

        for col in df_all.columns:
            if col != '행정동':
                df_all[col] = df_all[col].astype(str).str.replace(',', '').astype(float)

        df_all['상권활성도'] = df_all['동물병원수'] + df_all['동물약국수'] + df_all['펫샵수']
        df_all['교통편의도'] = df_all['버스정류장수'] + df_all['역개수']
        df_all['Y']         = df_all['반려동물등록수'] / (df_all['펫파크수'] + 1)

        X_vars    = ['인구수', '유동인구', '상권활성도', '교통편의도', '공시지가', '놀이터개수']
        stat_cols = X_vars + ['Y']
        stat_df   = df_all[stat_cols].describe().T.round(2)
        stat_df.index.name = '변수'

        st.markdown("**📋 변수별 기초통계**")

        # 변수별 분포 박스플롯
        box_data = df_all[X_vars].copy()
        # 정규화 (스케일 차이 크므로 시각화용)
        box_norm = (box_data - box_data.min()) / (box_data.max() - box_data.min() + 1e-9)
        box_norm['행정동'] = df_all['행정동']
        box_melt = box_norm.melt(id_vars='행정동', var_name='변수', value_name='정규화값')

        var_colors = {name: color for name, color, _, _ in x_meta}

        fig_box = go.Figure()
        for name, color, _, _ in x_meta:
            sub = box_melt[box_melt['변수'] == name]
            fig_box.add_trace(go.Box(
                y=sub['정규화값'],
                name=name,
                marker_color=color,
                boxmean='sd',
                hovertemplate=f'<b>{name}</b><br>정규화값: %{{y:.3f}}<extra></extra>',
            ))

        fig_box.update_layout(
            template='plotly_white',
            title=dict(text='독립변수 분포 비교 (MinMax 정규화)', x=0.5, xanchor='center'),
            yaxis=dict(title='정규화값 (0~1)', gridcolor='#f1f5f9'),
            xaxis=dict(title='변수'),
            showlegend=False,
            margin=dict(t=50, b=20), height=340,
        )
        st.plotly_chart(fig_box, use_container_width=True, key="var_boxplot")

        st.dataframe(
            stat_df.rename(columns={
                'count': '개수', 'mean': '평균', 'std': '표준편차',
                'min': '최솟값', '25%': '1사분위', '50%': '중앙값',
                '75%': '3사분위', 'max': '최댓값'
            }),
            use_container_width=True,
        )

        st.markdown("---")

    except Exception as e:
        st.warning(f"기초통계 계산 중 오류 (히트맵은 정상 표시됩니다): {e}")
        st.markdown("---")
else:
    st.markdown("---")

# ══════════════════════════════════════════
# 1. 상관계수 히트맵
# ══════════════════════════════════════════
corr = pd.read_csv(os.path.join(OUTPUT_DIR, '상관분석_결과.csv'), encoding='utf-8-sig', index_col=0)
vif  = pd.read_csv(os.path.join(OUTPUT_DIR, 'VIF_결과.csv'),     encoding='utf-8-sig')

st.subheader("🔥 변수 간 상관관계 히트맵")

fig_heat = px.imshow(
    corr,
    text_auto='.2f',
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1,
    aspect='auto',
    title='변수 간 상관계수',
)
fig_heat.update_layout(
    height=500,
    coloraxis_colorbar=dict(title='상관계수'),
)
st.plotly_chart(fig_heat, use_container_width=True, key="corr_heatmap")

st.markdown("---")

# ══════════════════════════════════════════
# 2. 상관계수 테이블 + 위험 변수 강조
# ══════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 상관계수 테이블")

    def highlight_high_corr(val):
        if isinstance(val, float):
            if abs(val) >= 0.9:
                return 'background-color: #ff4444; color: white'
            elif abs(val) >= 0.7:
                return 'background-color: #ffaa00; color: white'
        return ''

    st.dataframe(
        corr.style.map(highlight_high_corr).format('{:.3f}'),
        use_container_width=True,
    )

with col2:
    st.subheader("🔍 다중공선성 위험 쌍")

    danger_pairs = []
    vars_list = corr.columns.tolist()
    for i in range(len(vars_list)):
        for j in range(i + 1, len(vars_list)):
            val = abs(corr.iloc[i, j])
            if val >= 0.7:
                status = '🔴 제거 검토' if val >= 0.9 else '🟡 주의'
                danger_pairs.append({
                    '변수1': vars_list[i],
                    '변수2': vars_list[j],
                    '상관계수': round(corr.iloc[i, j], 3),
                    '판단': status,
                })

    if danger_pairs:
        st.dataframe(pd.DataFrame(danger_pairs), use_container_width=True)
    else:
        st.success("✅ 다중공선성 위험 변수 없음! 모든 변수 사용 가능")

with st.expander("📌 판단 기준 보기"):
    st.markdown("""
    | 상관계수 | 판단 |
    |---|---|
    | 0.9 이상 | 🔴 변수 제거 또는 통합 검토 |
    | 0.7 ~ 0.9 | 🟡 주의 |
    | 0.7 미만  | 🟢 사용 가능 |
    """)

st.markdown("---")

# ══════════════════════════════════════════
# 3. VIF 바 차트
# ══════════════════════════════════════════
st.subheader("📐 VIF (분산팽창계수)")

def get_vif_color(v):
    if v >= 10: return '#ef4444'
    elif v >= 5: return '#f59e0b'
    else:        return '#22c55e'

fig_vif = go.Figure(go.Bar(
    x=vif['변수'],
    y=vif['VIF'],
    marker_color=[get_vif_color(v) for v in vif['VIF']],
    marker=dict(cornerradius=5),
    text=vif['VIF'].round(2),
    textposition='outside',
))
fig_vif.add_hline(y=5,  line_dash='dash', line_color='#f59e0b',
                  annotation_text='주의 (5)',  annotation_position='top right')
fig_vif.add_hline(y=10, line_dash='dash', line_color='#ef4444',
                  annotation_text='위험 (10)', annotation_position='top right')
fig_vif.update_layout(
    template='plotly_white',
    title=dict(text='변수별 VIF', x=0.5, xanchor='center'),
    yaxis=dict(title='VIF', gridcolor='#f1f5f9'),
    xaxis_title='변수',
    margin=dict(t=50, b=20), height=400,
)
st.plotly_chart(fig_vif, use_container_width=True, key="vif_bar")

vif['판단'] = vif['VIF'].apply(
    lambda v: '✅ 정상' if v < 5 else ('⚠️ 주의' if v < 10 else '❌ 제거 필요')
)
st.dataframe(vif, use_container_width=True, hide_index=True)