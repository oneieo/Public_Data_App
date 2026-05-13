import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

st.set_page_config(layout="wide")
st.title("✅ K-medoids 클러스터링 최종 결과")
st.markdown("K-means, GMM 기각 후 **K-medoids 채택** — 클래스 균형 양호, 이상치에 강건")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

st.markdown("---")

# ══════════════════════════════════════════
# 데이터 로드 & 클러스터링 연산
# ══════════════════════════════════════════
@st.cache_data
def load_and_cluster():
    ground = pd.read_csv(os.path.join(DATA_DIR, 'final_playground.csv'), encoding='utf-8-sig')
    pet    = pd.read_csv(os.path.join(DATA_DIR, 'final_pet_park_.csv'),  encoding='utf-8-sig')
    zscore = pd.read_csv(os.path.join(DATA_DIR, 'zscore.csv'),           encoding='utf-8-sig')

    top23  = zscore[zscore['순위'] <= 23][['행정동', 'Z점수', '순위']].copy()
    ground = ground[ground['행정동'].isin(top23['행정동'])].copy().reset_index(drop=True)

    ground['설치연도'] = pd.to_datetime(ground['설치일자'], errors='coerce').dt.year
    ground['노후도']   = (2025 - ground['설치연도']).clip(lower=0)
    ground['노후도']   = ground['노후도'].fillna(ground['노후도'].median())
    ground = ground[ground['노후도'] > 0].copy().reset_index(drop=True)

    for col in ['위도', '경도']:
        ground[col] = pd.to_numeric(ground[col], errors='coerce')
        pet[col]    = pd.to_numeric(pet[col],    errors='coerce')
    ground = ground.dropna(subset=['위도', '경도']).copy().reset_index(drop=True)
    pet    = pet.dropna(subset=['위도', '경도']).copy()

    dist_matrix       = haversine_distances(
        np.radians(ground[['위도', '경도']].values),
        np.radians(pet[['위도', '경도']].values)
    ) * 6371
    ground['펫존거리'] = dist_matrix.min(axis=1)

    ground['공공시설여부'] = ground['민간_공공구분'].map({'공공': 1, '민간': 0})
    ground['면적'] = pd.to_numeric(ground['면적(제곱미터)'], errors='coerce')
    ground['면적'] = ground['면적'].fillna(ground['면적'].median())

    features = ['공공시설여부', '면적', '노후도', '펫존거리']
    X        = ground[features].copy().fillna(ground[features].mean(numeric_only=True))
    X_scaled = MinMaxScaler().fit_transform(X)

    pca       = PCA(n_components=2)
    X_pca     = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    # ── Elbow / Silhouette ──
    K_range     = range(2, 7)
    inertias, silhouettes = [], []
    for k in K_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl    = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_pca, lbl))

    optimal_k = list(K_range)[silhouettes.index(max(silhouettes))]

    # ── K-medoids ──
    def kmedoids(X, k, max_iter=100, random_state=42):
        np.random.seed(random_state)
        n   = len(X)
        idx = np.random.choice(n, k, replace=False)
        for _ in range(max_iter):
            dists  = np.array([[np.sum((X[i] - X[m])**2) for m in idx] for i in range(n)])
            labels = np.argmin(dists, axis=1)
            new_idx = []
            for j in range(k):
                pts = np.where(labels == j)[0]
                if len(pts) == 0:
                    new_idx.append(idx[j]); continue
                best, bc = pts[0], float('inf')
                for p in pts:
                    c = sum(np.sum((X[p] - X[q])**2) for q in pts)
                    if c < bc:
                        bc = c; best = p
                new_idx.append(best)
            if new_idx == list(idx): break
            idx = np.array(new_idx)
        return labels

    km_labels = kmedoids(X_pca, optimal_k)
    ground['클러스터'] = km_labels + 1

    # ── 입지점수 → 최적 클러스터 ──
    summary_raw  = ground.groupby('클러스터')[features].mean()
    summary_raw['입지점수'] = (
        summary_raw['면적']         / summary_raw['면적'].max() +
        summary_raw['노후도']       / summary_raw['노후도'].max() +
        summary_raw['공공시설여부'] +
        summary_raw['펫존거리']     / summary_raw['펫존거리'].max()
    )
    best_cluster = int(summary_raw['입지점수'].idxmax())

    # ── 동별 대표 + 누적합 ──
    best_df  = ground[ground['클러스터'] == best_cluster].copy()
    dong_rep = (best_df.sort_values('노후도', ascending=False)
                       .drop_duplicates('행정동').copy())
    dong_rep = dong_rep.merge(top23[['행정동', 'Z점수', '순위']], on='행정동', how='left')
    dong_rep = dong_rep.sort_values('Z점수', ascending=False).reset_index(drop=True)
    dong_rep['선택순위'] = dong_rep.index + 1
    dong_rep['누적합']   = dong_rep['Z점수'].cumsum()

    threshold    = dong_rep['Z점수'].max() * 0.05
    optimal_rows = dong_rep[dong_rep['Z점수'] < threshold]
    optimal_n    = int(optimal_rows.index[0]) if len(optimal_rows) > 0 else len(dong_rep)

    return (ground, X_pca, explained,
            list(K_range), inertias, silhouettes, optimal_k,
            km_labels, summary_raw.reset_index(),
            best_cluster, dong_rep, optimal_n, features)


(ground, X_pca, explained,
 K_range, inertias, silhouettes, optimal_k,
 km_labels, summary, best_cluster,
 dong_rep, optimal_n, features) = load_and_cluster()

COLORS = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b']
sizes  = [int(np.sum(km_labels == i)) for i in range(optimal_k)]
ratio  = max(sizes) / (min(sizes) + 1e-9)

# ══════════════════════════════════════════
# 1. Elbow + Silhouette
# ══════════════════════════════════════════
st.subheader("📐 최적 K 결정")

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
        name='Inertia',
    ))
    fig_elbow.update_layout(
        title=dict(text='Elbow Method', x=0.5, xanchor='center'),
        template='plotly_white',
        xaxis=dict(title='K (군집 수)', tickvals=K_range, gridcolor='#f1f5f9'),
        yaxis=dict(title='Inertia (SSE)', gridcolor='#f1f5f9'),
        margin=dict(t=50, b=20), height=320,
    )
    st.plotly_chart(fig_elbow, use_container_width=True, key="elbow")

with col2:
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(
        x=K_range, y=silhouettes,
        mode='lines+markers+text',
        text=[f'{v:.3f}' for v in silhouettes],
        textposition='top center',
        line=dict(color='#ef4444', width=2),
        marker=dict(size=8, color='#ef4444'),
        name='Silhouette',
    ))
    fig_sil.add_vline(
        x=optimal_k, line_dash='dash', line_color='#6366f1',
        annotation_text=f'최적 K={optimal_k}',
        annotation_position='top right',
    )
    fig_sil.update_layout(
        title=dict(text='Silhouette Score', x=0.5, xanchor='center'),
        template='plotly_white',
        xaxis=dict(title='K (군집 수)', tickvals=K_range, gridcolor='#f1f5f9'),
        yaxis=dict(title='Silhouette Score', gridcolor='#f1f5f9'),
        margin=dict(t=50, b=20), height=320,
    )
    st.plotly_chart(fig_sil, use_container_width=True, key="silhouette")

st.markdown("---")

# ══════════════════════════════════════════
# 2. K-medoids 클러스터링 산점도 (Convex Hull)
# ══════════════════════════════════════════
st.subheader("📊 K-medoids 클러스터링 결과")

# 채택/기각 판정
verdict_color = '#2563eb' if ratio <= 4 else '#dc2626'
verdict_text  = f'✅ 채택 (균형 양호)' if ratio <= 4 else f'❌ 기각 (불균형 {ratio:.1f}배)'

fig_scatter = go.Figure()

for i in range(optimal_k):
    mask  = km_labels == i
    pts_x = X_pca[mask, 0]
    pts_y = X_pca[mask, 1]
    color = COLORS[i]
    label = f'클러스터 {i+1} ({mask.sum()}개)'

    # 산점도
    fig_scatter.add_trace(go.Scatter(
        x=pts_x, y=pts_y,
        mode='markers',
        name=label,
        marker=dict(color=color, size=7, opacity=0.8,
                    line=dict(color='white', width=0.4)),
        hovertemplate=f'클러스터 {i+1}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>',
    ))

    # Convex Hull 외곽선
    if mask.sum() >= 3:
        try:
            from scipy.spatial import ConvexHull
            pts   = np.column_stack([pts_x, pts_y])
            hull  = ConvexHull(pts)
            verts = np.append(hull.vertices, hull.vertices[0])  # 닫힘
            fig_scatter.add_trace(go.Scatter(
                x=pts[verts, 0], y=pts[verts, 1],
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

    # 클러스터 중심 마커
    fig_scatter.add_trace(go.Scatter(
        x=[pts_x.mean()], y=[pts_y.mean()],
        mode='markers+text',
        marker=dict(symbol='x', size=14, color=color, line=dict(width=2)),
        text=[f'C{i+1}'], textposition='top center',
        textfont=dict(size=11, color=color),
        showlegend=False, hoverinfo='skip',
    ))

fig_scatter.update_layout(
    title=dict(
        text=f'K-medoids 클러스터링 결과 (K={optimal_k})  |  군집 크기: {sizes}  |  {verdict_text}',
        x=0.5, xanchor='center',
        font=dict(size=13, color=verdict_color),
    ),
    template='plotly_white',
    xaxis=dict(title=f'PC1 ({explained[0]*100:.0f}%)', gridcolor='#f1f5f9'),
    yaxis=dict(title=f'PC2 ({explained[1]*100:.0f}%)', gridcolor='#f1f5f9'),
    legend=dict(title='클러스터', orientation='h', y=-0.12, x=0.5, xanchor='center'),
    margin=dict(t=60, b=20), height=500,
)
st.plotly_chart(fig_scatter, use_container_width=True, key="kmedoids_scatter")

# 채택 근거 박스
with st.expander("📌 K-medoids 채택 근거"):
    st.success(f"클러스터 크기: {sizes}  |  불균형 비율 {ratio:.1f}배 → 균형 양호")
    st.markdown("""
    - ① 클래스 균형 양호 — K-means·GMM 대비 고른 군집 크기
    - ② 이상치에 강건 — 평균(centroid) 대신 실제 데이터 포인트(medoid) 기반
    - ③ 소규모 데이터에 적합 — 노이즈 영향 최소화
    """)

st.markdown("---")

# ══════════════════════════════════════════
# 3. 클러스터별 평균 특성
# ══════════════════════════════════════════
st.subheader("📋 클러스터별 평균 특성")

feat_cols = [c for c in features if c in summary.columns]
summary_disp = summary[['클러스터'] + feat_cols].copy()

# 정규화 스케일 차이가 크므로 변수별 Radar + Bar 두 가지 제공
tab1, tab2 = st.tabs(["📊 막대 차트", "🕸️ 레이더 차트"])

with tab1:
    fig_bar = go.Figure()
    bar_colors = ['#6366f1', '#f59e0b', '#ef4444', '#10b981']
    for ci, feat in enumerate(feat_cols):
        fig_bar.add_trace(go.Bar(
            name=feat,
            x=[f'클러스터 {int(r["클러스터"])}' for _, r in summary_disp.iterrows()],
            y=summary_disp[feat],
            marker=dict(color=bar_colors[ci], cornerradius=4),
            text=[f'{v:.2f}' for v in summary_disp[feat]],
            textposition='outside',
        ))
    fig_bar.update_layout(
        barmode='group',
        template='plotly_white',
        title=dict(text='클러스터별 평균 특성 비교', x=0.5, xanchor='center'),
        xaxis_title='클러스터',
        yaxis=dict(gridcolor='#f1f5f9'),
        legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
        margin=dict(t=50, b=20), height=400,
    )
    st.plotly_chart(fig_bar, use_container_width=True, key="cluster_bar")

with tab2:
    # MinMax 정규화 후 레이더
    radar_df = summary_disp.copy()
    for feat in feat_cols:
        mn, mx = radar_df[feat].min(), radar_df[feat].max()
        radar_df[feat] = (radar_df[feat] - mn) / (mx - mn + 1e-9)

    fig_radar = go.Figure()
    for _, row in radar_df.iterrows():
        cnum = int(row['클러스터'])
        vals = [row[f] for f in feat_cols] + [row[feat_cols[0]]]  # 닫힘
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=feat_cols + [feat_cols[0]],
            fill='toself',
            name=f'클러스터 {cnum}',
            line=dict(color=COLORS[cnum - 1], width=2),
            fillcolor=COLORS[cnum - 1],
            opacity=0.25,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template='plotly_white',
        title=dict(text='클러스터별 특성 레이더 (정규화)', x=0.5, xanchor='center'),
        legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
        margin=dict(t=60, b=20), height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True, key="cluster_radar")

st.dataframe(
    summary_disp.rename(columns={'클러스터': '클러스터 번호'}),
    use_container_width=True, hide_index=True,
)

st.markdown("---")

# ══════════════════════════════════════════
# 4. 동별 Z점수 누적합 (최적 개수 결정)
# ══════════════════════════════════════════
st.subheader("📈 동별 Z점수 누적합 — 최적 개수 결정")

fig_cum = go.Figure()

# 누적합 라인
fig_cum.add_trace(go.Scatter(
    x=dong_rep['선택순위'],
    y=dong_rep['누적합'],
    mode='lines+markers',
    line=dict(color='#3b82f6', width=2.5),
    marker=dict(size=8, color='#3b82f6'),
    name='Z점수 누적합',
    hovertemplate='순위: %{x}<br>누적합: %{y:.3f}<extra></extra>',
))

# 동 이름 어노테이션
for _, row in dong_rep.iterrows():
    fig_cum.add_annotation(
        x=row['선택순위'], y=row['누적합'],
        text=row['행정동'],
        showarrow=False,
        yshift=12,
        font=dict(size=9, color='#475569'),
        textangle=-40,
    )

# 최적 개수 수직선
fig_cum.add_vline(
    x=optimal_n, line_dash='dash', line_color='#ef4444', line_width=2,
    annotation_text=f'최적 {optimal_n}개',
    annotation_position='top right',
    annotation_font=dict(color='#ef4444', size=12),
)
fig_cum.add_vrect(
    x0=optimal_n - 0.4, x1=optimal_n + 0.4,
    fillcolor='#ef4444', opacity=0.08, line_width=0,
)

fig_cum.update_layout(
    template='plotly_white',
    xaxis=dict(title='동 선택 개수 (Z점수 순위 기준)',
               tickvals=dong_rep['선택순위'].tolist(),
               gridcolor='#f1f5f9'),
    yaxis=dict(title='Z점수 누적합', gridcolor='#f1f5f9'),
    legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
    margin=dict(t=40, b=60), height=420,
)
st.plotly_chart(fig_cum, use_container_width=True, key="cumsum")

st.markdown("---")