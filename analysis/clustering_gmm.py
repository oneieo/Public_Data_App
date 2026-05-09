import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(BASE, exist_ok=True)

OUTPUT_DIR = os.path.join(BASE, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
ground = pd.read_csv(f'{BASE}\\data\\final_playground.csv', encoding='utf-8-sig')
pet    = pd.read_csv(f'{BASE}\\data\\final_pet_park_.csv',  encoding='utf-8-sig')
zscore = pd.read_csv(f'{BASE}\\data\\zscore.csv',           encoding='utf-8-sig')

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
# GMM K 결정: PC1 밀도 그래프 봉우리 개수 (PDF 방식)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, pc_idx, pc_name in zip(axes, [0, 1], ['PC1', 'PC2']):
    data = X_pca[:, pc_idx]
    kde  = gaussian_kde(data, bw_method=0.3)
    x    = np.linspace(data.min(), data.max(), 500)
    y    = kde(x)

    ax.plot(x, y, color='steelblue', lw=2)
    ax.fill_between(x, y, alpha=0.2, color='steelblue')

    peaks = [(x[i], y[i]) for i in range(1, len(y)-1)
             if y[i] > y[i-1] and y[i] > y[i+1]]

    for px, py in peaks:
        ax.axvline(x=px, color='red', linestyle='--', alpha=0.7)
        ax.annotate(f'봉우리\n({px:.2f})', (px, py),
                    textcoords='offset points', xytext=(8, 0),
                    color='red', fontsize=9)

    ax.set_xlabel(pc_name); ax.set_ylabel('밀도')
    ax.set_title(f'{pc_name} 밀도 분포 (봉우리 {len(peaks)}개 → GMM K={len(peaks)})')
    ax.grid(alpha=0.3)
    print(f'{pc_name} 봉우리 개수: {len(peaks)}개')

plt.suptitle('GMM: 최적 K 결정 (밀도 그래프 봉우리 개수)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'gmm_01_density.png'), dpi=150, bbox_inches='tight')
plt.close()

# PC1 봉우리 개수를 GMM K로 사용
pc1_data  = X_pca[:, 0]
kde_pc1   = gaussian_kde(pc1_data, bw_method=0.3)
x_pc1     = np.linspace(pc1_data.min(), pc1_data.max(), 500)
y_pc1     = kde_pc1(x_pc1)
n_peaks   = sum(1 for i in range(1, len(y_pc1)-1)
                if y_pc1[i] > y_pc1[i-1] and y_pc1[i] > y_pc1[i+1])
optimal_k = max(2, n_peaks)
print(f'\nGMM 최적 K = {optimal_k} (PC1 봉우리 기준)')

# ============================================================
# GMM 클러스터링
# ============================================================
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
labels = GaussianMixture(n_components=optimal_k, random_state=42).fit_predict(X_pca)
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
    f'GMM 클러스터링 결과 (K={optimal_k})\n'
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
plt.savefig(os.path.join(OUTPUT_DIR, 'gmm_02_result.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f'GMM: 군집 크기={sizes}, 불균형={ratio:.1f}배 → ❌ 기각')
print('\n저장 완료:')
print('gmm_01_density.png : GMM K 결정 밀도 그래프')
print('gmm_02_result.png  : GMM 클러스터링 결과')
