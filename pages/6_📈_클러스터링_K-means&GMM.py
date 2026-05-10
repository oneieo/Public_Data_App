import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde

st.set_page_config(layout="wide")
st.title("🔵 클러스터링 비교 분석 (K-means & GMM)")
st.markdown("K-means와 GMM 두 방법을 시도했으나 **클래스 불균형** 문제로 최종 기각")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

st.markdown("---")

@st.cache_data
def load_cluster_data():
    ground = pd.read_csv(os.path.join(DATA_DIR, 'final_playground.csv'), encoding='utf-8-sig')
    pet    = pd.read_csv(os.path.join(DATA_DIR, 'final_pet_park_.csv'),  encoding='utf-8-sig')
    zscore = pd.read_csv(os.path.join(DATA_DIR, 'zscore.csv'),           encoding='utf-8-sig')

    top23  = zscore[zscore['순위'] <= 23][['행정동','Z점수','순위']].copy()
    ground = ground[ground['행정동'].isin(top23['행정동'])].copy().reset_index(drop=True)

    ground['설치연도'] = pd.to_datetime(ground['설치일자'], errors='coerce').dt.year
    ground['노후도']   = (2025 - ground['설치연도']).clip(lower=0)
    ground['노후도']   = ground['노후도'].fillna(ground['노후도'].median())
    ground = ground[ground['노후도'] > 0].copy().reset_index(drop=True)

    ground['위도'] = pd.to_numeric(ground['위도'], errors='coerce')
    ground['경도'] = pd.to_numeric(ground['경도'], errors='coerce')
    pet['위도']    = pd.to_numeric(pet['위도'], errors='coerce')
    pet['경도']    = pd.to_numeric(pet['경도'], errors='coerce')
    ground = ground.dropna(subset=['위도','경도']).copy().reset_index(drop=True)
    pet    = pet.dropna(subset=['위도','경도']).copy()

    ground_coords     = np.radians(ground[['위도','경도']].values)
    pet_coords        = np.radians(pet[['위도','경도']].values)
    dist_matrix       = haversine_distances(ground_coords, pet_coords) * 6371
    ground['펫존거리'] = dist_matrix.min(axis=1)

    ground['공공시설여부'] = ground['민간_공공구분'].map({'공공': 1, '민간': 0})
    ground['면적'] = pd.to_numeric(ground['면적(제곱미터)'], errors='coerce')
    ground['면적'] = ground['면적'].fillna(ground['면적'].median())

    return ground

ground   = load_cluster_data()
ground_r = ground.reset_index(drop=True)
features = ['공공시설여부', '면적', '노후도', '펫존거리']
COLORS   = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b']

# ══════════════════════════════════════════
# PCA 사전 계산 (전체 페이지에서 공유)
# ══════════════════════════════════════════
@st.cache_data
def compute_pca(ground_df):
    feats = ['공공시설여부', '면적', '노후도', '펫존거리']
    X = ground_df[feats].copy().fillna(ground_df[feats].mean(numeric_only=True))
    scaler    = MinMaxScaler()
    X_scaled  = scaler.fit_transform(X)
    pca       = PCA(n_components=2)
    X_pca     = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    loadings  = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feats
    ).round(3)
    return X_pca, explained, loadings

X_pca_shared, explained_shared, loadings_shared = compute_pca(ground)

# ══════════════════════════════════════════
# 클러스터 변수 생성
# ══════════════════════════════════════════
st.subheader("🔧 클러스터 변수 생성")

# ── 1) 사용 변수 카드 ──────────────────────
st.markdown("##### 📌 사용 변수")
c1, c2, c3, c4 = st.columns(4)
c1.info("**공공시설여부**\n\n공공=1 / 민간=0\n이진 변수")
c2.info("**면적** (㎡)\n\n놀이터 면적\n결측 → 중앙값 대체")
c3.info("**노후도** (년)\n\n2025 - 설치연도\n오래될수록 높음")
c4.info("**펫존거리** (km)\n\n가장 가까운\n반려동물 공원까지")

st.markdown("---")

# ══════════════════════════════════════════
# PCA 차원 축소 설명 섹션
# ══════════════════════════════════════════
st.markdown("##### 🔬 PCA 차원 축소 (4차원 → 2차원)")
st.caption(
    "4개 변수를 MinMaxScaler로 정규화한 뒤 PCA로 2개 주성분(PC1·PC2)으로 압축합니다. "
    "클러스터링은 이 2차원 공간에서 수행됩니다."
)

# ── (A) 설명력 지표 카드 ──────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("PC1 설명력", f"{explained_shared[0]*100:.1f}%",
          help="PC1이 원본 데이터 분산을 설명하는 비율")
m2.metric("PC2 설명력", f"{explained_shared[1]*100:.1f}%",
          help="PC2가 원본 데이터 분산을 설명하는 비율")
m3.metric("누적 설명력", f"{explained_shared.sum()*100:.1f}%",
          help="PC1 + PC2 합산 설명력")
m4.metric("압축 차원", "4D → 2D",
          help="클러스터링에 사용되는 최종 차원 수")

st.markdown("")

# ── (B) 분산 설명력 막대 + 로딩 히트맵 ───
col_var, col_heat = st.columns([1, 1.2])

with col_var:
    st.markdown("**① 주성분별 분산 설명력**")
    var_df = pd.DataFrame({
        '주성분': ['PC1', 'PC2'],
        '개별 설명력 (%)': [round(explained_shared[0]*100, 1),
                            round(explained_shared[1]*100, 1)],
        '누적 설명력 (%)': [round(explained_shared[0]*100, 1),
                            round(explained_shared.sum()*100, 1)],
    })

    fig_var = go.Figure()
    fig_var.add_trace(go.Bar(
        x=var_df['주성분'],
        y=var_df['개별 설명력 (%)'],
        name='개별 설명력',
        marker=dict(color=['#6366f1', '#a5b4fc'], cornerradius=6),
        text=[f"{v}%" for v in var_df['개별 설명력 (%)']],
        textposition='outside',
        width=0.4,
    ))
    fig_var.add_trace(go.Scatter(
        x=var_df['주성분'],
        y=var_df['누적 설명력 (%)'],
        name='누적 설명력',
        mode='lines+markers+text',
        text=[f"{v}%" for v in var_df['누적 설명력 (%)']],
        textposition='top center',
        line=dict(color='#f59e0b', width=2, dash='dot'),
        marker=dict(size=10, color='#f59e0b', symbol='diamond'),
    ))
    fig_var.update_layout(
        template='plotly_white',
        yaxis=dict(title='설명력 (%)', range=[0, 115], gridcolor='#f1f5f9'),
        xaxis=dict(title='주성분'),
        legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center'),
        margin=dict(t=30, b=10),
        height=310,
        bargap=0.4,
    )
    st.plotly_chart(fig_var, use_container_width=True, key="pca_variance")

with col_heat:
    st.markdown("**② 주성분 로딩 (변수 기여도)**")
    load_vals = loadings_shared.values
    load_vars = loadings_shared.index.tolist()
    load_pcs  = ['PC1', 'PC2']

    fig_heat = go.Figure(go.Heatmap(
        z=load_vals,
        x=load_pcs,
        y=load_vars,
        colorscale=[
            [0.0,  '#b91c1c'],
            [0.25, '#ef4444'],
            [0.5,  '#f8fafc'],
            [0.75, '#6366f1'],
            [1.0,  '#312e81'],
        ],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:+.3f}" for v in row] for row in load_vals],
        texttemplate='%{text}',
        textfont=dict(size=14, family='monospace'),
        showscale=True,
        colorbar=dict(
            title='로딩값',
            tickvals=[-1, -0.5, 0, 0.5, 1],
            len=0.85,
            thickness=14,
            tickfont=dict(size=11),
        ),
        hoverongaps=False,
        hovertemplate='변수: %{y}<br>주성분: %{x}<br>로딩: %{z:.3f}<extra></extra>',
    ))
    fig_heat.update_layout(
        template='plotly_white',
        xaxis=dict(title='주성분', side='top', tickfont=dict(size=13, color='#1e293b')),
        yaxis=dict(title='', autorange='reversed', tickfont=dict(size=12)),
        margin=dict(t=40, b=10, l=10, r=10),
        height=310,
    )
    st.plotly_chart(fig_heat, use_container_width=True, key="pca_loading_heat")

st.caption(
    "🔵 양수(파란색): 해당 PC 방향으로 기여  |  "
    "🔴 음수(빨간색): 반대 방향으로 기여  |  "
    "절댓값이 클수록 기여도 높음"
)

st.markdown("")

# ── (C) Biplot ────────────────────────────
st.markdown("**③ Biplot — 데이터 분포 & 변수 방향**")

ARROW_COLORS = ['#6366f1', '#f59e0b', '#ef4444', '#10b981']
scale        = 2.2

np.random.seed(42)
n_sample   = min(400, len(X_pca_shared))
sample_idx = np.random.choice(len(X_pca_shared), size=n_sample, replace=False)
sample_x   = X_pca_shared[sample_idx, 0]
sample_y   = X_pca_shared[sample_idx, 1]

fig_bi = go.Figure()

# 데이터 포인트
fig_bi.add_trace(go.Scatter(
    x=sample_x, y=sample_y,
    mode='markers',
    marker=dict(color='#cbd5e1', size=5, opacity=0.55,
                line=dict(color='#94a3b8', width=0.3)),
    name='놀이터',
    hovertemplate='PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra>놀이터</extra>',
))

# 로딩 화살표 + 레이블
for i, var in enumerate(features):
    lx  = float(loadings_shared.loc[var, 'PC1']) * scale
    ly  = float(loadings_shared.loc[var, 'PC2']) * scale
    clr = ARROW_COLORS[i]

    # 화살표 몸통
    fig_bi.add_trace(go.Scatter(
        x=[0, lx], y=[0, ly],
        mode='lines',
        line=dict(color=clr, width=2.5),
        showlegend=False, hoverinfo='skip',
    ))
    # 화살표 끝 마커
    fig_bi.add_trace(go.Scatter(
        x=[lx], y=[ly],
        mode='markers',
        marker=dict(symbol='arrow', size=14, color=clr,
                    angleref='previous', standoff=0),
        showlegend=False, hoverinfo='skip',
    ))
    # 변수명 레이블
    fig_bi.add_annotation(
        x=lx * 1.15, y=ly * 1.15,
        text=f"<b>{var}</b>",
        showarrow=False,
        font=dict(color=clr, size=12, family='Malgun Gothic'),
        bgcolor='rgba(255,255,255,0.8)',
        borderpad=2,
    )

# 원점 십자선
ax_range = max(abs(X_pca_shared).max() * 1.05, abs(scale) * 1.15)
fig_bi.add_shape(type='line', x0=0, x1=0, y0=-ax_range, y1=ax_range,
                 line=dict(color='#e2e8f0', width=1, dash='dot'))
fig_bi.add_shape(type='line', x0=-ax_range, x1=ax_range, y0=0, y1=0,
                 line=dict(color='#e2e8f0', width=1, dash='dot'))

fig_bi.update_layout(
    template='plotly_white',
    xaxis=dict(
        title=f'PC1 ({explained_shared[0]*100:.1f}%)',
        gridcolor='#f1f5f9', zeroline=False,
        range=[-ax_range, ax_range],
    ),
    yaxis=dict(
        title=f'PC2 ({explained_shared[1]*100:.1f}%)',
        gridcolor='#f1f5f9', zeroline=False,
        range=[-ax_range, ax_range],
        scaleanchor='x', scaleratio=1,
    ),
    legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center'),
    margin=dict(t=20, b=20, l=20, r=20),
    height=460,
    hovermode='closest',
)
st.plotly_chart(fig_bi, use_container_width=True, key="pca_biplot")

# ── (D) PC1/PC2 해석 expander ─────────────
with st.expander("📖 PC1 / PC2 해석 보기"):
    pc1_main = loadings_shared['PC1'].abs().idxmax()
    pc2_main = loadings_shared['PC2'].abs().idxmax()
    pc1_sign = "＋" if loadings_shared.loc[pc1_main, 'PC1'] > 0 else "－"
    pc2_sign = "＋" if loadings_shared.loc[pc2_main, 'PC2'] > 0 else "－"

    st.markdown(f"""
| 주성분 | 설명력 | 주요 기여 변수 | 로딩 부호 | 해석 |
|:------:|:------:|:-------------:|:---------:|:-----|
| **PC1** | {explained_shared[0]*100:.1f}% | **{pc1_main}** | {pc1_sign} | PC1 값이 높을수록 **{pc1_main}** 특성이 강함 |
| **PC2** | {explained_shared[1]*100:.1f}% | **{pc2_main}** | {pc2_sign} | PC2 값이 높을수록 **{pc2_main}** 특성이 강함 |

> **로딩(Loading)**: 원본 변수가 주성분에 얼마나, 어느 방향으로 기여하는지를 나타내는 계수입니다.  
> Biplot 화살표가 **같은 방향**이면 두 변수는 양의 상관, **반대 방향**이면 음의 상관입니다.
    """)

    fig_load_exp = go.Figure(go.Heatmap(
        z=loadings_shared.values,
        x=['PC1', 'PC2'],
        y=loadings_shared.index.tolist(),
        colorscale=[
            [0.0, '#b91c1c'], [0.25, '#ef4444'],
            [0.5, '#f8fafc'],
            [0.75, '#6366f1'], [1.0, '#312e81'],
        ],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:+.3f}" for v in row] for row in loadings_shared.values],
        texttemplate='%{text}',
        textfont=dict(size=13, family='monospace'),
        showscale=True,
        colorbar=dict(title='로딩값', tickvals=[-1,-0.5,0,0.5,1],
                      len=0.85, thickness=12),
        hovertemplate='변수: %{y}<br>주성분: %{x}<br>로딩: %{z:.3f}<extra></extra>',
    ))
    fig_load_exp.update_layout(
        template='plotly_white',
        xaxis=dict(side='top', tickfont=dict(size=13)),
        yaxis=dict(autorange='reversed', tickfont=dict(size=12)),
        margin=dict(t=30, b=10, l=10, r=10),
        height=220,
    )
    st.plotly_chart(fig_load_exp, use_container_width=True, key="pca_load_exp")

st.markdown("---")

# ══════════════════════════════════════════
# 변수별 분포
# ══════════════════════════════════════════
st.markdown("##### 📊 변수별 분포")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(ground, x='면적', nbins=40, title='면적 분포',
                        color_discrete_sequence=['#6366f1'], template='plotly_white')
    fig1.update_layout(bargap=0.05, showlegend=False,
                       xaxis_title='면적 (㎡)', yaxis_title='개수',
                       margin=dict(t=50, b=20), height=280)
    st.plotly_chart(fig1, use_container_width=True, key="hist_area")

with col2:
    fig2 = px.histogram(ground, x='노후도', nbins=30, title='노후도 분포',
                        color_discrete_sequence=['#f59e0b'], template='plotly_white')
    fig2.update_layout(bargap=0.05, showlegend=False,
                       xaxis_title='노후도 (년)', yaxis_title='개수',
                       margin=dict(t=50, b=20), height=280)
    st.plotly_chart(fig2, use_container_width=True, key="hist_age")

col3, col4 = st.columns(2)

with col3:
    fig3 = px.histogram(ground, x='펫존거리', nbins=35, title='펫존거리 분포',
                        color_discrete_sequence=['#10b981'], template='plotly_white')
    fig3.update_layout(bargap=0.05, showlegend=False,
                       xaxis_title='거리 (km)', yaxis_title='개수',
                       margin=dict(t=50, b=20), height=280)
    st.plotly_chart(fig3, use_container_width=True, key="hist_dist")

with col4:
    pub_counts = ground['공공시설여부'].map({1: '공공', 0: '민간'}).value_counts().reset_index()
    pub_counts.columns = ['구분', '개수']
    fig4 = px.pie(pub_counts, names='구분', values='개수', title='공공 / 민간 비율',
                  color_discrete_sequence=['#6366f1', '#94a3b8'], template='plotly_white')
    fig4.update_traces(textposition='inside', textinfo='percent+label')
    fig4.update_layout(margin=dict(t=50, b=20), height=280, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True, key="pie_public")

st.markdown("---")

st.markdown("##### 🔗 변수 간 관계")
col1, col2 = st.columns(2)

with col1:
    fig5 = px.scatter(ground, x='면적', y='노후도',
                      color=ground_r['공공시설여부'].map({1: '공공', 0: '민간'}),
                      color_discrete_map={'공공': '#6366f1', '민간': '#f43f5e'},
                      opacity=0.6, title='면적 vs 노후도',
                      labels={'color': '구분'}, template='plotly_white')
    fig5.update_traces(marker=dict(size=5))
    fig5.update_layout(xaxis_title='면적 (㎡)', yaxis_title='노후도 (년)',
                       margin=dict(t=50, b=20), height=300)
    st.plotly_chart(fig5, use_container_width=True, key="scatter_area_age")

with col2:
    fig6 = px.scatter(ground, x='펫존거리', y='면적',
                      color=ground_r['공공시설여부'].map({1: '공공', 0: '민간'}),
                      color_discrete_map={'공공': '#6366f1', '민간': '#f43f5e'},
                      opacity=0.6, title='펫존거리 vs 면적',
                      labels={'color': '구분'}, template='plotly_white')
    fig6.update_traces(marker=dict(size=5))
    fig6.update_layout(xaxis_title='펫존거리 (km)', yaxis_title='면적 (㎡)',
                       margin=dict(t=50, b=20), height=300)
    st.plotly_chart(fig6, use_container_width=True, key="scatter_dist_area")

st.markdown("---")

st.markdown("##### 🏘️ 행정동별 변수 평균")
dong_mean     = ground.groupby('행정동')[['면적', '노후도', '펫존거리']].mean().reset_index()
selected_feat = st.selectbox("변수 선택", ['면적', '노후도', '펫존거리'])

fig7 = px.bar(
    dong_mean.sort_values(selected_feat, ascending=True),
    x=selected_feat, y='행정동', orientation='h', color=selected_feat,
    color_continuous_scale={
        '면적':     ['#c7d2fe', '#6366f1', '#312e81'],
        '노후도':   ['#fef3c7', '#f59e0b', '#92400e'],
        '펫존거리': ['#d1fae5', '#10b981', '#064e3b'],
    }[selected_feat],
    template='plotly_white', title=f'행정동별 평균 {selected_feat}'
)
fig7.update_traces(
    text=dong_mean.sort_values(selected_feat)[selected_feat].round(1),
    textposition='outside', marker=dict(cornerradius=4)
)
fig7.update_layout(coloraxis_showscale=False, yaxis_title='',
                   margin=dict(t=50, r=60, b=20),
                   xaxis=dict(gridcolor='#f1f5f9'),
                   height=max(350, len(dong_mean) * 26))
st.plotly_chart(fig7, use_container_width=True, key="bar_dong")

st.markdown("---")

st.markdown("##### 🔥 변수 간 상관관계")
corr = ground[features].corr().round(3)
fig8 = px.imshow(corr, color_continuous_scale='RdBu', color_continuous_midpoint=0,
                 text_auto=True, aspect='auto', template='plotly_white',
                 title='클러스터 변수 간 상관관계')
fig8.update_layout(title=dict(x=0.5, xanchor='center'), margin=dict(t=60), height=350)
fig8.update_traces(textfont=dict(size=12))
st.plotly_chart(fig8, use_container_width=True, key="heatmap_corr")

st.markdown("##### 📋 요약 통계")
st.dataframe(
    ground[features].describe().round(2).T.rename(columns={
        'count': '개수', 'mean': '평균', 'std': '표준편차',
        'min': '최솟값', '25%': '1사분위', '50%': '중앙값',
        '75%': '3사분위', 'max': '최댓값'
    }),
    use_container_width=True
)

st.markdown("---")

# ══════════════════════════════
# K-means
# ══════════════════════════════
st.subheader("❌ K-means 클러스터링")

@st.cache_data
def run_kmeans(ground_df):
    ground_df = ground_df.reset_index(drop=True)
    features  = ['공공시설여부', '면적', '노후도', '펫존거리']
    X = ground_df[features].copy().fillna(ground_df[features].mean(numeric_only=True))

    scaler    = MinMaxScaler()
    X_scaled  = scaler.fit_transform(X)
    pca       = PCA(n_components=2)
    X_pca     = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    K_range = range(2, 7)
    inertias, silhouettes = [], []
    for k in K_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_pca, labels))

    optimal_k = list(K_range)[silhouettes.index(max(silhouettes))]
    labels    = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit_predict(X_pca)

    return X_pca, explained, list(K_range), inertias, silhouettes, optimal_k, labels

X_pca, explained, K_range, inertias, silhouettes, optimal_k, labels = run_kmeans(ground)

sizes = [int(np.sum(labels == i)) for i in range(optimal_k)]
ratio = max(sizes) / (min(sizes) + 1e-9)

cluster_df = pd.DataFrame({
    'PC1':      X_pca[:, 0],
    'PC2':      X_pca[:, 1],
    '클러스터': [f'클러스터 {i+1} ({sizes[i]}개)' for i in labels],
    '행정동':   ground_r['행정동'].values,
    '면적':     ground_r['면적'].values,
    '노후도':   ground_r['노후도'].values,
    '펫존거리': ground_r['펫존거리'].values,
})

st.markdown("##### 📐 최적 K 결정")
col1, col2 = st.columns(2)

with col1:
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=K_range, y=inertias, mode='lines+markers+text',
        text=[f'{v:.1f}' for v in inertias], textposition='top center',
        line=dict(color='#3b82f6', width=2), marker=dict(size=8, color='#3b82f6'),
        name='Inertia'
    ))
    fig_elbow.update_layout(
        title=dict(text='Elbow Method', x=0.5, xanchor='center'),
        template='plotly_white',
        xaxis=dict(title='K', tickvals=K_range, gridcolor='#f1f5f9'),
        yaxis=dict(title='Inertia', gridcolor='#f1f5f9'),
        margin=dict(t=50, b=20), height=300
    )
    st.plotly_chart(fig_elbow, use_container_width=True, key="km_elbow")

with col2:
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(
        x=K_range, y=silhouettes, mode='lines+markers+text',
        text=[f'{v:.3f}' for v in silhouettes], textposition='top center',
        line=dict(color='#ef4444', width=2), marker=dict(size=8, color='#ef4444'),
        name='Silhouette'
    ))
    fig_sil.add_vline(x=optimal_k, line_dash='dash', line_color='#6366f1',
                      annotation_text=f'최적 K={optimal_k}',
                      annotation_position='top right')
    fig_sil.update_layout(
        title=dict(text='Silhouette Score', x=0.5, xanchor='center'),
        template='plotly_white',
        xaxis=dict(title='K', tickvals=K_range, gridcolor='#f1f5f9'),
        yaxis=dict(title='Silhouette Score', gridcolor='#f1f5f9'),
        margin=dict(t=50, b=20), height=300
    )
    st.plotly_chart(fig_sil, use_container_width=True, key="km_silhouette")

st.markdown("---")
st.markdown("##### 🔵 클러스터링 결과")

color_map = {f'클러스터 {i+1} ({sizes[i]}개)': COLORS[i] for i in range(optimal_k)}
 
fig_km = px.scatter(
    cluster_df, x='PC1', y='PC2', color='클러스터',
    color_discrete_map=color_map,
    hover_data=['행정동', '면적', '노후도', '펫존거리'],
    opacity=0.8, template='plotly_white',
    title=f'K-means 클러스터링 결과 (K={optimal_k}) — 불균형 비율 {ratio:.1f}배 ❌'
)
fig_km.update_traces(marker=dict(size=7, line=dict(color='white', width=0.5)))
fig_km.update_layout(
    title=dict(x=0.5, xanchor='center', font=dict(size=14, color='#ef4444')),
    xaxis=dict(title=f'PC1 ({explained[0]*100:.0f}%)', gridcolor='#f1f5f9'),
    yaxis=dict(title=f'PC2 ({explained[1]*100:.0f}%)', gridcolor='#f1f5f9'),
    legend=dict(title='클러스터'), margin=dict(t=60, b=20), height=480
)
 
# ── Convex Hull 외곽선 + 배경색 ──────────────────
from scipy.spatial import ConvexHull
 
for i in range(optimal_k):
    mask  = labels == i
    pts   = X_pca[mask]          # shape (n, 2)
    color = COLORS[i]
 
    if mask.sum() >= 3:
        try:
            hull  = ConvexHull(pts)
            verts = np.append(hull.vertices, hull.vertices[0])  # 닫힘
            fig_km.add_trace(go.Scatter(
                x=pts[verts, 0],
                y=pts[verts, 1],
                mode='lines',
                line=dict(color=color, width=1.8, dash='dot'),
                fill='toself',
                fillcolor=color,
                opacity=0.12,
                showlegend=False,
                hoverinfo='skip',
            ))
        except Exception:
            pass
 
# ── 클러스터 중심 마커 ────────────────────────────
for i in range(optimal_k):
    mask = labels == i
    fig_km.add_trace(go.Scatter(
        x=[X_pca[mask, 0].mean()], y=[X_pca[mask, 1].mean()],
        mode='markers+text',
        marker=dict(symbol='x', size=14, color=COLORS[i], line=dict(width=2)),
        text=[f'C{i+1}'], textposition='top center',
        textfont=dict(size=11, color=COLORS[i]), showlegend=False
    ))
 
st.plotly_chart(fig_km, use_container_width=True, key="km_scatter")
 
size_df = pd.DataFrame({
    '클러스터': [f'클러스터 {i+1}' for i in range(optimal_k)],
    '개수': sizes
})
fig_size = px.bar(
    size_df, x='클러스터', y='개수', text='개수', color='클러스터',
    color_discrete_map={f'클러스터 {i+1}': COLORS[i] for i in range(optimal_k)},
    template='plotly_white', title='클러스터별 데이터 수'
)
fig_size.update_traces(textposition='outside', marker=dict(cornerradius=5))
fig_size.update_layout(showlegend=False, title=dict(x=0.5, xanchor='center'),
                       yaxis=dict(gridcolor='#f1f5f9'),
                       margin=dict(t=50, b=20), height=280)
st.plotly_chart(fig_size, use_container_width=True, key="km_size")
 
with st.expander("📌 K-means 기각 이유"):
    st.error(f"클래스 불균형 심함 ({min(sizes)}개 vs {max(sizes)}개, {ratio:.1f}배 차이) → 분석 신뢰성 낮음")
    st.markdown("""
    - 군집 간 크기 차이가 과도하게 큼
    - 이상치에 민감한 평균 기반 방식
    - 소규모 데이터에 불리
    """)
 
st.markdown("---")

# ══════════════════════════════
# GMM
# ══════════════════════════════
st.subheader("❌ GMM 클러스터링")

@st.cache_data
def run_gmm(ground_df):
    ground_df = ground_df.reset_index(drop=True)
    features  = ['공공시설여부', '면적', '노후도', '펫존거리']
    X = ground_df[features].copy().fillna(ground_df[features].mean(numeric_only=True))

    scaler    = MinMaxScaler()
    X_scaled  = scaler.fit_transform(X)
    pca       = PCA(n_components=2)
    X_pca     = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    kde_results = {}
    for pc_idx, pc_name in enumerate(['PC1', 'PC2']):
        data  = X_pca[:, pc_idx]
        kde   = gaussian_kde(data, bw_method=0.3)
        x     = np.linspace(data.min(), data.max(), 500)
        y     = kde(x)
        peaks = [(x[i], y[i]) for i in range(1, len(y)-1)
                 if y[i] > y[i-1] and y[i] > y[i+1]]
        kde_results[pc_name] = {'x': x, 'y': y, 'peaks': peaks}

    optimal_k = max(2, len(kde_results['PC1']['peaks']))
    labels    = GaussianMixture(n_components=optimal_k, random_state=42).fit_predict(X_pca)
    sizes     = [int(np.sum(labels == i)) for i in range(optimal_k)]
    ratio     = max(sizes) / (min(sizes) + 1e-9)

    return X_pca, explained, kde_results, optimal_k, labels, sizes, ratio

X_pca_g, explained_g, kde_results, optimal_k_g, labels_g, sizes_g, ratio_g = run_gmm(ground)

st.markdown("##### 📐 최적 K 결정 — KDE 밀도 봉우리")
col1, col2 = st.columns(2)

for col, pc_name in zip([col1, col2], ['PC1', 'PC2']):
    res   = kde_results[pc_name]
    x, y  = res['x'], res['y']
    peaks = res['peaks']

    fig_kde = go.Figure()
    fig_kde.add_trace(go.Scatter(
        x=x, y=y, mode='lines', fill='tozeroy',
        fillcolor='rgba(59,130,246,0.15)',
        line=dict(color='#3b82f6', width=2), name='밀도'
    ))
    for i, (peak_x, peak_y) in enumerate(peaks):
        fig_kde.add_vline(
            x=peak_x, line_dash='dash', line_color='#ef4444', line_width=1.5,
            annotation_text=f'봉우리 {i+1}\n({peak_x:.2f})',
            annotation_position='top right',
            annotation_font=dict(color='#ef4444', size=10)
        )
    if peaks:
        fig_kde.add_trace(go.Scatter(
            x=[p[0] for p in peaks], y=[p[1] for p in peaks],
            mode='markers',
            marker=dict(color='#ef4444', size=9, symbol='circle'),
            name=f'봉우리 ({len(peaks)}개)', showlegend=True
        ))
    fig_kde.update_layout(
        title=dict(text=f'{pc_name} 밀도 분포 — 봉우리 {len(peaks)}개 → GMM K={len(peaks)}',
                   x=0.5, xanchor='center', font=dict(size=13)),
        template='plotly_white',
        xaxis=dict(title=pc_name, gridcolor='#f1f5f9'),
        yaxis=dict(title='밀도', gridcolor='#f1f5f9'),
        legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
        margin=dict(t=60, b=20), height=320
    )
    with col:
        st.plotly_chart(fig_kde, use_container_width=True, key=f"gmm_kde_{pc_name}")

st.caption(f"PC1 봉우리 {len(kde_results['PC1']['peaks'])}개 기준 → GMM 최적 K = **{optimal_k_g}**")
st.markdown("---")

st.markdown("##### 🔴 클러스터링 결과")
 
cluster_df_g = pd.DataFrame({
    'PC1':      X_pca_g[:, 0],
    'PC2':      X_pca_g[:, 1],
    '클러스터': [f'클러스터 {i+1} ({sizes_g[i]}개)' for i in labels_g],
    '행정동':   ground_r['행정동'].values,
    '면적':     ground_r['면적'].values,
    '노후도':   ground_r['노후도'].values,
    '펫존거리': ground_r['펫존거리'].values,
})
 
color_map_g = {f'클러스터 {i+1} ({sizes_g[i]}개)': COLORS[i] for i in range(optimal_k_g)}
 
fig_gmm = px.scatter(
    cluster_df_g, x='PC1', y='PC2', color='클러스터',
    color_discrete_map=color_map_g,
    hover_data=['행정동', '면적', '노후도', '펫존거리'],
    opacity=0.8, template='plotly_white',
    title=f'GMM 클러스터링 결과 (K={optimal_k_g}) — 불균형 비율 {ratio_g:.1f}배 ❌'
)
fig_gmm.update_traces(marker=dict(size=7, line=dict(color='white', width=0.5)))
fig_gmm.update_layout(
    title=dict(x=0.5, xanchor='center', font=dict(size=14, color='#ef4444')),
    xaxis=dict(title=f'PC1 ({explained_g[0]*100:.0f}%)', gridcolor='#f1f5f9'),
    yaxis=dict(title=f'PC2 ({explained_g[1]*100:.0f}%)', gridcolor='#f1f5f9'),
    legend=dict(title='클러스터'), margin=dict(t=60, b=20), height=480
)
 
# ── Convex Hull 외곽선 + 배경색 ──────────────────
from scipy.spatial import ConvexHull
 
for i in range(optimal_k_g):
    mask  = labels_g == i
    pts   = X_pca_g[mask]
    color = COLORS[i]
 
    if mask.sum() >= 3:
        try:
            hull  = ConvexHull(pts)
            verts = np.append(hull.vertices, hull.vertices[0])  # 닫힘
            fig_gmm.add_trace(go.Scatter(
                x=pts[verts, 0],
                y=pts[verts, 1],
                mode='lines',
                line=dict(color=color, width=1.8, dash='dot'),
                fill='toself',
                fillcolor=color,
                opacity=0.12,
                showlegend=False,
                hoverinfo='skip',
            ))
        except Exception:
            pass
 
# ── 클러스터 중심 마커 ────────────────────────────
for i in range(optimal_k_g):
    mask = labels_g == i
    fig_gmm.add_trace(go.Scatter(
        x=[X_pca_g[mask, 0].mean()], y=[X_pca_g[mask, 1].mean()],
        mode='markers+text',
        marker=dict(symbol='x', size=14, color=COLORS[i], line=dict(width=2)),
        text=[f'C{i+1}'], textposition='top center',
        textfont=dict(size=11, color=COLORS[i]), showlegend=False
    ))
 
st.plotly_chart(fig_gmm, use_container_width=True, key="gmm_scatter")
 
size_df_g = pd.DataFrame({
    '클러스터': [f'클러스터 {i+1}' for i in range(optimal_k_g)],
    '개수': sizes_g
})
fig_size_g = px.bar(
    size_df_g, x='클러스터', y='개수', text='개수', color='클러스터',
    color_discrete_map={f'클러스터 {i+1}': COLORS[i] for i in range(optimal_k_g)},
    template='plotly_white', title='클러스터별 데이터 수'
)
fig_size_g.update_traces(textposition='outside', marker=dict(cornerradius=5))
fig_size_g.update_layout(showlegend=False, title=dict(x=0.5, xanchor='center'),
                          yaxis=dict(gridcolor='#f1f5f9'),
                          margin=dict(t=50, b=20), height=280)
st.plotly_chart(fig_size_g, use_container_width=True, key="gmm_size")
 
with st.expander("📌 GMM 기각 이유"):
    st.error(f"클래스 불균형 심함 ({min(sizes_g)}개 vs {max(sizes_g)}개, {ratio_g:.1f}배 차이) → 분석 신뢰성 낮음")
    st.markdown("""
    - 확률 기반이나 군집 크기 불균형 해결 못함
    - 소규모 데이터에서 과적합 위험
    """)

st.markdown("---")
st.info("💡 두 방법 모두 기각 → **K-medoids 채택**")