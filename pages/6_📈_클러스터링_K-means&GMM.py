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

# ══════════════════════════════════════════
# 클러스터 변수 생성
# ══════════════════════════════════════════
st.subheader("🔧 클러스터 변수 생성")

# CSV 로드 (분석 코드와 동일한 전처리 결과)
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

    from sklearn.metrics.pairwise import haversine_distances
    import numpy as np
    ground_coords     = np.radians(ground[['위도','경도']].values)
    pet_coords        = np.radians(pet[['위도','경도']].values)
    dist_matrix       = haversine_distances(ground_coords, pet_coords) * 6371
    ground['펫존거리'] = dist_matrix.min(axis=1)

    ground['공공시설여부'] = ground['민간_공공구분'].map({'공공': 1, '민간': 0})
    ground['면적'] = pd.to_numeric(ground['면적(제곱미터)'], errors='coerce')
    ground['면적'] = ground['면적'].fillna(ground['면적'].median())

    return ground

ground = load_cluster_data()
features = ['공공시설여부', '면적', '노후도', '펫존거리']

# ──────────────────────────────────────────
# 변수 설명 카드
# ──────────────────────────────────────────
st.markdown("##### 📌 사용 변수")
c1, c2, c3, c4 = st.columns(4)
c1.info("**공공시설여부**\n\n공공=1 / 민간=0\n이진 변수")
c2.info("**면적** (㎡)\n\n놀이터 면적\n결측 → 중앙값 대체")
c3.info("**노후도** (년)\n\n2025 - 설치연도\n오래될수록 높음")
c4.info("**펫존거리** (km)\n\n가장 가까운\n반려동물 공원까지")

st.markdown("---")

# ──────────────────────────────────────────
# 1. 변수별 분포 (히스토그램)
# ──────────────────────────────────────────
st.markdown("##### 📊 변수별 분포")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        ground, x='면적', nbins=40,
        title='면적 분포',
        color_discrete_sequence=['#6366f1'],
        template='plotly_white'
    )
    fig1.update_layout(
        bargap=0.05, showlegend=False,
        xaxis_title='면적 (㎡)', yaxis_title='개수',
        margin=dict(t=50, b=20), height=280
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(
        ground, x='노후도', nbins=30,
        title='노후도 분포',
        color_discrete_sequence=['#f59e0b'],
        template='plotly_white'
    )
    fig2.update_layout(
        bargap=0.05, showlegend=False,
        xaxis_title='노후도 (년)', yaxis_title='개수',
        margin=dict(t=50, b=20), height=280
    )
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig3 = px.histogram(
        ground, x='펫존거리', nbins=35,
        title='펫존거리 분포',
        color_discrete_sequence=['#10b981'],
        template='plotly_white'
    )
    fig3.update_layout(
        bargap=0.05, showlegend=False,
        xaxis_title='거리 (km)', yaxis_title='개수',
        margin=dict(t=50, b=20), height=280
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    pub_counts = ground['공공시설여부'].map({1: '공공', 0: '민간'}).value_counts().reset_index()
    pub_counts.columns = ['구분', '개수']
    fig4 = px.pie(
        pub_counts, names='구분', values='개수',
        title='공공 / 민간 비율',
        color_discrete_sequence=['#6366f1', '#94a3b8'],
        template='plotly_white'
    )
    fig4.update_traces(textposition='inside', textinfo='percent+label')
    fig4.update_layout(margin=dict(t=50, b=20), height=280, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────
# 2. 변수 간 관계 (산점도)
# ──────────────────────────────────────────
st.markdown("##### 🔗 변수 간 관계")

col1, col2 = st.columns(2)

with col1:
    fig5 = px.scatter(
        ground, x='면적', y='노후도',
        color=ground['공공시설여부'].map({1: '공공', 0: '민간'}),
        color_discrete_map={'공공': '#6366f1', '민간': '#f43f5e'},
        opacity=0.6,
        title='면적 vs 노후도',
        labels={'color': '구분'},
        template='plotly_white'
    )
    fig5.update_traces(marker=dict(size=5))
    fig5.update_layout(
        xaxis_title='면적 (㎡)', yaxis_title='노후도 (년)',
        margin=dict(t=50, b=20), height=300
    )
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    fig6 = px.scatter(
        ground, x='펫존거리', y='면적',
        color=ground['공공시설여부'].map({1: '공공', 0: '민간'}),
        color_discrete_map={'공공': '#6366f1', '민간': '#f43f5e'},
        opacity=0.6,
        title='펫존거리 vs 면적',
        labels={'color': '구분'},
        template='plotly_white'
    )
    fig6.update_traces(marker=dict(size=5))
    fig6.update_layout(
        xaxis_title='펫존거리 (km)', yaxis_title='면적 (㎡)',
        margin=dict(t=50, b=20), height=300
    )
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────
# 3. 행정동별 변수 평균 비교
# ──────────────────────────────────────────
st.markdown("##### 🏘️ 행정동별 변수 평균")

dong_mean = ground.groupby('행정동')[['면적', '노후도', '펫존거리']].mean().reset_index()

selected_feat = st.selectbox("변수 선택", ['면적', '노후도', '펫존거리'])

fig7 = px.bar(
    dong_mean.sort_values(selected_feat, ascending=True),
    x=selected_feat, y='행정동',
    orientation='h',
    color=selected_feat,
    color_continuous_scale={
        '면적':    ['#c7d2fe', '#6366f1', '#312e81'],
        '노후도':  ['#fef3c7', '#f59e0b', '#92400e'],
        '펫존거리':['#d1fae5', '#10b981', '#064e3b'],
    }[selected_feat],
    template='plotly_white',
    title=f'행정동별 평균 {selected_feat}'
)
fig7.update_traces(
    text=dong_mean.sort_values(selected_feat)[selected_feat].round(1),
    textposition='outside',
    marker=dict(cornerradius=4)
)
fig7.update_layout(
    coloraxis_showscale=False,
    yaxis_title='', margin=dict(t=50, r=60, b=20),
    xaxis=dict(gridcolor='#f1f5f9'),
    height=max(350, len(dong_mean) * 26)
)
st.plotly_chart(fig7, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────
# 4. 상관관계 히트맵
# ──────────────────────────────────────────
st.markdown("##### 🔥 변수 간 상관관계")

corr = ground[features].corr().round(3)

fig8 = px.imshow(
    corr,
    color_continuous_scale='RdBu',
    color_continuous_midpoint=0,
    text_auto=True,
    aspect='auto',
    template='plotly_white',
    title='클러스터 변수 간 상관관계'
)
fig8.update_layout(
    title=dict(x=0.5, xanchor='center'),
    margin=dict(t=60),
    height=350
)
fig8.update_traces(textfont=dict(size=12))
st.plotly_chart(fig8, use_container_width=True)

# ──────────────────────────────────────────
# 5. 요약 통계
# ──────────────────────────────────────────
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
def run_kmeans(ground_df):  # ← ground_df로 받음
    ground_df = ground_df.reset_index(drop=True)  # ← 함수 안에서 리셋
    features  = ['공공시설여부', '면적', '노후도', '펫존거리']
    X = ground_df[features].copy().fillna(ground_df[features].mean(numeric_only=True))

    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

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

# ← ground 전체를 넘김 (len(ground) → ground)
X_pca, explained, K_range, inertias, silhouettes, optimal_k, labels = run_kmeans(ground)

# 이후 cluster_df 만드는 부분도 수정
sizes = [int(np.sum(labels == i)) for i in range(optimal_k)]
ratio = max(sizes) / (min(sizes) + 1e-9)

ground_r = ground.reset_index(drop=True)  # ← 여기서도 리셋

cluster_df = pd.DataFrame({
    'PC1':      X_pca[:, 0],
    'PC2':      X_pca[:, 1],
    '클러스터': [f'클러스터 {i+1} ({sizes[i]}개)' for i in labels],
    '행정동':   ground_r['행정동'].values,  # ← ground_r 사용
    '면적':     ground_r['면적'].values,
    '노후도':   ground_r['노후도'].values,
    '펫존거리': ground_r['펫존거리'].values,
})

# ──────────────────────────────────────────
# 1. Elbow + Silhouette
# ──────────────────────────────────────────
st.markdown("##### 📐 최적 K 결정")

col1, col2 = st.columns(2)

with col1:
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=K_range, y=inertias,
        mode='lines+markers+text',
        text=[f'{v:.1f}' for v in inertias],
        textposition='top center',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=8, color='#3b82f6'),
        name='Inertia'
    ))
    fig_elbow.update_layout(
        title=dict(text='Elbow Method', x=0.5, xanchor='center'),
        template='plotly_white',
        xaxis=dict(title='K', tickvals=K_range, gridcolor='#f1f5f9'),
        yaxis=dict(title='Inertia', gridcolor='#f1f5f9'),
        margin=dict(t=50, b=20),
        height=300
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

with col2:
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(
        x=K_range, y=silhouettes,
        mode='lines+markers+text',
        text=[f'{v:.3f}' for v in silhouettes],
        textposition='top center',
        line=dict(color='#ef4444', width=2),
        marker=dict(size=8, color='#ef4444'),
        name='Silhouette'
    ))
    # 최적 K 강조선
    fig_sil.add_vline(
        x=optimal_k, line_dash='dash', line_color='#6366f1',
        annotation_text=f'최적 K={optimal_k}',
        annotation_position='top right'
    )
    fig_sil.update_layout(
        title=dict(text='Silhouette Score', x=0.5, xanchor='center'),
        template='plotly_white',
        xaxis=dict(title='K', tickvals=K_range, gridcolor='#f1f5f9'),
        yaxis=dict(title='Silhouette Score', gridcolor='#f1f5f9'),
        margin=dict(t=50, b=20),
        height=300
    )
    st.plotly_chart(fig_sil, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────
# 2. 클러스터링 결과 산점도
# ──────────────────────────────────────────
st.markdown("##### 🔵 클러스터링 결과")

sizes = [int(np.sum(labels == i)) for i in range(optimal_k)]
ratio = max(sizes) / (min(sizes) + 1e-9)

cluster_df = pd.DataFrame({
    'PC1':      X_pca[:, 0],
    'PC2':      X_pca[:, 1],
    '클러스터': [f'클러스터 {i+1} ({sizes[i]}개)' for i in labels],
    '행정동':   ground['행정동'].values,
    '면적':     ground['면적'].values,
    '노후도':   ground['노후도'].values,
    '펫존거리': ground['펫존거리'].values,
})

COLORS = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b']
color_map = {
    f'클러스터 {i+1} ({sizes[i]}개)': COLORS[i]
    for i in range(optimal_k)
}

fig_km = px.scatter(
    cluster_df,
    x='PC1', y='PC2',
    color='클러스터',
    color_discrete_map=color_map,
    hover_data=['행정동', '면적', '노후도', '펫존거리'],
    opacity=0.8,
    template='plotly_white',
    title=f'K-means 클러스터링 결과 (K={optimal_k}) — 불균형 비율 {ratio:.1f}배 ❌'
)
fig_km.update_traces(marker=dict(size=7, line=dict(color='white', width=0.5)))
fig_km.update_layout(
    title=dict(
        x=0.5, xanchor='center',
        font=dict(size=14, color='#ef4444')
    ),
    xaxis=dict(
        title=f'PC1 ({explained[0]*100:.0f}%)',
        gridcolor='#f1f5f9'
    ),
    yaxis=dict(
        title=f'PC2 ({explained[1]*100:.0f}%)',
        gridcolor='#f1f5f9'
    ),
    legend=dict(title='클러스터', orientation='v'),
    margin=dict(t=60, b=20),
    height=480
)

# 클러스터 중심 표시
for i in range(optimal_k):
    mask  = labels == i
    cx    = X_pca[mask, 0].mean()
    cy    = X_pca[mask, 1].mean()
    fig_km.add_trace(go.Scatter(
        x=[cx], y=[cy],
        mode='markers+text',
        marker=dict(symbol='x', size=14, color=COLORS[i], line=dict(width=2)),
        text=[f'C{i+1}'],
        textposition='top center',
        textfont=dict(size=11, color=COLORS[i]),
        showlegend=False
    ))

st.plotly_chart(fig_km, use_container_width=True)

# 클러스터별 크기 바 차트
size_df = pd.DataFrame({
    '클러스터': [f'클러스터 {i+1}' for i in range(optimal_k)],
    '개수': sizes,
    '색상': COLORS[:optimal_k]
})

fig_size = px.bar(
    size_df, x='클러스터', y='개수',
    text='개수',
    color='클러스터',
    color_discrete_map={f'클러스터 {i+1}': COLORS[i] for i in range(optimal_k)},
    template='plotly_white',
    title='클러스터별 데이터 수'
)
fig_size.update_traces(
    textposition='outside',
    marker=dict(cornerradius=5)
)
fig_size.update_layout(
    showlegend=False,
    title=dict(x=0.5, xanchor='center'),
    yaxis=dict(gridcolor='#f1f5f9'),
    margin=dict(t=50, b=20),
    height=280
)
st.plotly_chart(fig_size, use_container_width=True)

with st.expander("📌 K-means 기각 이유"):
    st.error(f"클래스 불균형 심함 ({min(sizes)}개 vs {max(sizes)}개, {ratio:.1f}배 차이) → 분석 신뢰성 낮음")
    st.markdown("""
    - 군집 간 크기 차이가 과도하게 큼
    - 이상치에 민감한 평균 기반 방식
    - 소규모 데이터에 불리
    """)

st.markdown("---")

# ══════════════════════════════
# GMM 클러스터링
# ══════════════════════════════

st.subheader("❌ GMM 클러스터링")

@st.cache_data
def run_gmm(ground_df):  # ground 전체를 받음
    ground_df = ground_df.reset_index(drop=True)  # 함수 안에서 리셋
    features = ['공공시설여부', '면적', '노후도', '펫존거리']
    X = ground_df[features].copy().fillna(ground_df[features].mean(numeric_only=True))

    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

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
        kde_results[pc_name] = {'x': x, 'y': y, 'peaks': peaks, 'data': data}

    n_peaks   = len(kde_results['PC1']['peaks'])
    optimal_k = max(2, n_peaks)

    labels = GaussianMixture(n_components=optimal_k, random_state=42).fit_predict(X_pca)
    sizes  = [int(np.sum(labels == i)) for i in range(optimal_k)]
    ratio  = max(sizes) / (min(sizes) + 1e-9)

    return X_pca, explained, kde_results, optimal_k, labels, sizes, ratio

# 호출할 때 ground 전체 넘기기
X_pca_g, explained_g, kde_results, optimal_k_g, labels_g, sizes_g, ratio_g = run_gmm(ground)

# ──────────────────────────────────────────
# 1. KDE 밀도 그래프 (K 결정)
# ──────────────────────────────────────────
st.markdown("##### 📐 최적 K 결정 — KDE 밀도 봉우리")

col1, col2 = st.columns(2)

COLORS = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b']

for col, pc_name in zip([col1, col2], ['PC1', 'PC2']):
    res    = kde_results[pc_name]
    x, y   = res['x'], res['y']
    peaks  = res['peaks']

    fig_kde = go.Figure()

    # KDE 곡선
    fig_kde.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(59,130,246,0.15)',
        line=dict(color='#3b82f6', width=2),
        name='밀도'
    ))

    # 봉우리 수직선 + 어노테이션
    for i, (peak_x, peak_y) in enumerate(peaks):
        fig_kde.add_vline(
            x=peak_x, line_dash='dash', line_color='#ef4444', line_width=1.5,
            annotation_text=f'봉우리 {i+1}\n({peak_x:.2f})',
            annotation_position='top right',
            annotation_font=dict(color='#ef4444', size=10)
    )

    # 봉우리 마커
    if peaks:
        fig_kde.add_trace(go.Scatter(
            x=[p[0] for p in peaks],
            y=[p[1] for p in peaks],
            mode='markers',
            marker=dict(color='#ef4444', size=9, symbol='circle'),
            name=f'봉우리 ({len(peaks)}개)',
            showlegend=True
        ))

    fig_kde.update_layout(
        title=dict(
            text=f'{pc_name} 밀도 분포 — 봉우리 {len(peaks)}개 → GMM K={len(peaks)}',
            x=0.5, xanchor='center', font=dict(size=13)
        ),
        template='plotly_white',
        xaxis=dict(title=pc_name, gridcolor='#f1f5f9'),
        yaxis=dict(title='밀도', gridcolor='#f1f5f9'),
        legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
        margin=dict(t=60, b=20),
        height=320
    )

    with col:
        st.plotly_chart(fig_kde, use_container_width=True)

st.caption(f"PC1 봉우리 {len(kde_results['PC1']['peaks'])}개 기준 → GMM 최적 K = **{optimal_k_g}**")

st.markdown("---")

# ──────────────────────────────────────────
# 2. GMM 클러스터링 결과 산점도
# ──────────────────────────────────────────
st.markdown("##### 🔴 클러스터링 결과")

ground_r = ground.reset_index(drop=True)

cluster_df_g = pd.DataFrame({
    'PC1':      X_pca_g[:, 0],
    'PC2':      X_pca_g[:, 1],
    '클러스터': [f'클러스터 {i+1} ({sizes_g[i]}개)' for i in labels_g],
    '행정동':   ground_r['행정동'].values,
    '면적':     ground_r['면적'].values,
    '노후도':   ground_r['노후도'].values,
    '펫존거리': ground_r['펫존거리'].values,
})

color_map_g = {
    f'클러스터 {i+1} ({sizes_g[i]}개)': COLORS[i]
    for i in range(optimal_k_g)
}

fig_gmm = px.scatter(
    cluster_df_g,
    x='PC1', y='PC2',
    color='클러스터',
    color_discrete_map=color_map_g,
    hover_data=['행정동', '면적', '노후도', '펫존거리'],
    opacity=0.8,
    template='plotly_white',
    title=f'GMM 클러스터링 결과 (K={optimal_k_g}) — 불균형 비율 {ratio_g:.1f}배 ❌'
)
fig_gmm.update_traces(marker=dict(size=7, line=dict(color='white', width=0.5)))
fig_gmm.update_layout(
    title=dict(
        x=0.5, xanchor='center',
        font=dict(size=14, color='#ef4444')
    ),
    xaxis=dict(title=f'PC1 ({explained_g[0]*100:.0f}%)', gridcolor='#f1f5f9'),
    yaxis=dict(title=f'PC2 ({explained_g[1]*100:.0f}%)', gridcolor='#f1f5f9'),
    legend=dict(title='클러스터'),
    margin=dict(t=60, b=20),
    height=480
)

# 클러스터 중심 표시
for i in range(optimal_k_g):
    mask = labels_g == i
    cx   = X_pca_g[mask, 0].mean()
    cy   = X_pca_g[mask, 1].mean()
    fig_gmm.add_trace(go.Scatter(
        x=[cx], y=[cy],
        mode='markers+text',
        marker=dict(symbol='x', size=14, color=COLORS[i], line=dict(width=2)),
        text=[f'C{i+1}'],
        textposition='top center',
        textfont=dict(size=11, color=COLORS[i]),
        showlegend=False
    ))

st.plotly_chart(fig_gmm, use_container_width=True)

# 클러스터별 크기 바 차트
size_df_g = pd.DataFrame({
    '클러스터': [f'클러스터 {i+1}' for i in range(optimal_k_g)],
    '개수': sizes_g,
})

fig_size_g = px.bar(
    size_df_g, x='클러스터', y='개수',
    text='개수',
    color='클러스터',
    color_discrete_map={f'클러스터 {i+1}': COLORS[i] for i in range(optimal_k_g)},
    template='plotly_white',
    title='클러스터별 데이터 수'
)
fig_size_g.update_traces(
    textposition='outside',
    marker=dict(cornerradius=5)
)
fig_size_g.update_layout(
    showlegend=False,
    title=dict(x=0.5, xanchor='center'),
    yaxis=dict(gridcolor='#f1f5f9'),
    margin=dict(t=50, b=20),
    height=280
)
st.plotly_chart(fig_size_g, use_container_width=True)

with st.expander("📌 GMM 기각 이유"):
    st.error(f"클래스 불균형 심함 ({min(sizes_g)}개 vs {max(sizes_g)}개, {ratio_g:.1f}배 차이) → 분석 신뢰성 낮음")
    st.markdown("""
    - 확률 기반이나 군집 크기 불균형 해결 못함
    - 소규모 데이터에서 과적합 위험
    """)

st.markdown("---")
st.info("💡 두 방법 모두 기각 → **K-medoids 채택**")