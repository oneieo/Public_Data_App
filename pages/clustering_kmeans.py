import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE = r'C:\Users\오형종\OneDrive - 전북대학교\바탕 화면\project'
os.makedirs(BASE, exist_ok=True)

def draw_convex_hull(ax, points, color, alpha_fill=0.15, alpha_line=0.6):
    if len(points) < 3:
        return
    try:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1],
                    color=color, linewidth=1.5, alpha=alpha_line)
        ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
                alpha=alpha_fill, color=color)
    except Exception:
        pass

# ============================================================
# 데이터 준비
# ============================================================
ground = pd.read_csv(f'{BASE}\\final_playground.csv', encoding='utf-8-sig')
pet    = pd.read_csv(f'{BASE}\\final_pet_park_.csv',  encoding='utf-8-sig')
zscore = pd.read_csv(f'{BASE}\\zscore.csv',           encoding='utf-8-sig')

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

ground_coords  = np.radians(ground[['위도','경도']].values)
pet_coords     = np.radians(pet[['위도','경도']].values)
dist_matrix    = haversine_distances(ground_coords, pet_coords) * 6371
ground['펫존거리'] = dist_matrix.min(axis=1)

ground['공공시설여부'] = ground['민간_공공구분'].map({'공공': 1, '민간': 0})
ground['면적'] = pd.to_numeric(ground['면적(제곱미터)'], errors='coerce')
ground['면적'] = ground['면적'].fillna(ground['면적'].median())

features = ['공공시설여부', '면적', '노후도', '펫존거리']
X = ground[features].copy().fillna(ground[features].mean(numeric_only=True))

scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

pca       = PCA(n_components=2)
X_pca     = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_

# ============================================================
# Elbow Method
# ============================================================
inertias, silhouettes = [], []
K_range = range(2, 7)

for k in K_range:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_pca, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(K_range), inertias, 'o-', color='steelblue', lw=2, ms=8)
ax1.set_xlabel('K'); ax1.set_ylabel('Inertia'); ax1.set_title('Elbow Method')
ax1.grid(alpha=0.3)
for k, v in zip(K_range, inertias):
    ax1.annotate(f'{v:.1f}', (k, v), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=9)

ax2.plot(list(K_range), silhouettes, 's-', color='tomato', lw=2, ms=8)
ax2.set_xlabel('K'); ax2.set_ylabel('Silhouette Score'); ax2.set_title('Silhouette Score')
ax2.grid(alpha=0.3)
for k, v in zip(K_range, silhouettes):
    ax2.annotate(f'{v:.3f}', (k, v), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=9)

plt.suptitle('K-means: 최적 K 결정 (Elbow + Silhouette)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}\\kmeans_01_elbow.png', dpi=150, bbox_inches='tight')
plt.show()

optimal_k = list(K_range)[silhouettes.index(max(silhouettes))]
print(f'K-means 최적 K = {optimal_k}')

# ============================================================
# K-means 클러스터링
# ============================================================
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
labels = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit_predict(X_pca)
sizes  = [int(np.sum(labels == i)) for i in range(optimal_k)]
ratio  = max(sizes) / (min(sizes) + 1e-9)

fig, ax = plt.subplots(figsize=(9, 7))
handles = []
for i in range(optimal_k):
    mask  = labels == i
    pts   = X_pca[mask]
    color = colors[i]
    ax.scatter(pts[:, 0], pts[:, 1], c=color, s=30,
               alpha=0.8, edgecolors='white', linewidths=0.3, zorder=3)
    draw_convex_hull(ax, pts, color)
    handles.append(mpatches.Patch(color=color, alpha=0.6,
                                  label=f'클러스터 {i+1} ({mask.sum()}개)'))

ax.set_xlabel(f'PC1 ({explained[0]*100:.0f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({explained[1]*100:.0f}%)', fontsize=11)
ax.set_title(
    f'K-means 클러스터링 결과 (K={optimal_k})\n'
    f'군집 크기: {sizes}  |  불균형 비율: {ratio:.1f}배 → ❌ 기각',
    fontsize=12, fontweight='bold', color='red'
)
ax.legend(handles=handles, fontsize=10)
ax.grid(alpha=0.3)
ax.text(0.02, 0.02,
        f'기각 이유: 클래스 불균형 심함\n'
        f'({min(sizes)}개 vs {max(sizes)}개, {ratio:.1f}배 차이)\n'
        f'→ 분석 신뢰성 낮음',
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{BASE}\\kmeans_02_result.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'K-means: 군집 크기={sizes}, 불균형={ratio:.1f}배 → ❌ 기각')
print('\n저장 완료:')
print('kmeans_01_elbow.png  : Elbow + Silhouette')
print('kmeans_02_result.png : K-means 클러스터링 결과')
