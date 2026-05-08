import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE = r'C:\Users\user\Desktop\project'
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

def kmedoids(X, k, max_iter=100, random_state=42):
    np.random.seed(random_state)
    n   = len(X)
    idx = np.random.choice(n, k, replace=False)
    for _ in range(max_iter):
        dists  = np.array([[np.sum((X[i]-X[m])**2) for m in idx] for i in range(n)])
        labels = np.argmin(dists, axis=1)
        new_idx = []
        for j in range(k):
            pts = np.where(labels == j)[0]
            if len(pts) == 0:
                new_idx.append(idx[j]); continue
            best, bc = pts[0], float('inf')
            for p in pts:
                c = sum(np.sum((X[p]-X[q])**2) for q in pts)
                if c < bc: bc = c; best = p
            new_idx.append(best)
        if new_idx == list(idx): break
        idx = np.array(new_idx)
    return labels

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
# Elbow Method → 최적 K 결정
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

plt.suptitle('K-medoids: 최적 K 결정 (Elbow + Silhouette)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}\\kmedoids_01_elbow.png', dpi=150, bbox_inches='tight')
plt.show()

optimal_k = list(K_range)[silhouettes.index(max(silhouettes))]
print(f'K-medoids 최적 K = {optimal_k}')

# ============================================================
# K-medoids 클러스터링
# ============================================================
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
labels = kmedoids(X_pca, optimal_k)
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
    f'K-medoids 클러스터링 결과 (K={optimal_k})\n'
    f'군집 크기: {sizes}  |  불균형 비율: {ratio:.1f}배 → ✅ 채택',
    fontsize=12, fontweight='bold', color='blue'
)
ax.legend(handles=handles, fontsize=10)
ax.grid(alpha=0.3)
ax.text(0.02, 0.02,
        f'채택 이유:\n'
        f'① 클래스 균형 양호 ({min(sizes)}개 vs {max(sizes)}개)\n'
        f'② 이상치에 강건 (중앙점 기반)\n'
        f'③ 소규모 데이터에 적합',
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{BASE}\\kmedoids_02_result.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 전환 최적 클러스터 선정 및 최종 결과
# ============================================================
ground['클러스터'] = labels + 1
summary = ground.groupby('클러스터')[features].mean().round(2)
summary['입지점수'] = (
    summary['면적']         / summary['면적'].max() +
    summary['노후도']       / summary['노후도'].max() +
    summary['공공시설여부'] +
    summary['펫존거리']     / summary['펫존거리'].max()
)
best_cluster = int(summary['입지점수'].idxmax())

print('\n=== 클러스터별 평균 특성 ===')
print(summary)
print(f'\n✅ 전환 최적 클러스터: {best_cluster}번')

# 동별 대표 선정 (노후도 기준)
best_df  = ground[ground['클러스터'] == best_cluster].copy()
dong_rep = best_df.sort_values('노후도', ascending=False)\
                  .drop_duplicates('행정동').copy()
dong_rep = dong_rep.merge(top23[['행정동','Z점수','순위']], on='행정동', how='left')
dong_rep = dong_rep.sort_values('Z점수', ascending=False).reset_index(drop=True)
dong_rep['선택순위'] = dong_rep.index + 1
dong_rep['누적합']   = dong_rep['Z점수'].cumsum()

print(f'\n전환 최적 클러스터 내 동별 대표: {len(dong_rep)}개')

# 누적합 기울기 → 최적 개수
threshold    = dong_rep['Z점수'].max() * 0.05
optimal_rows = dong_rep[dong_rep['Z점수'] < threshold]
optimal_n    = int(optimal_rows.index[0]) if len(optimal_rows) > 0 else len(dong_rep)

# 누적합 그래프
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dong_rep['선택순위'], dong_rep['누적합'],
        'o-', color='steelblue', markersize=7, linewidth=2)
ax.axvline(x=optimal_n, color='red', linestyle='--', linewidth=2,
           label=f'최적 개수: {optimal_n}개')
ax.axvspan(optimal_n - 0.5, optimal_n + 0.5, color='gray', alpha=0.2)
for _, row in dong_rep.iterrows():
    ax.annotate(row['행정동'], (row['선택순위'], row['누적합']),
                textcoords='offset points', xytext=(0, 8),
                ha='center', fontsize=7, rotation=45)
ax.set_xlabel('동 선택 개수 (Z점수 순위 기준)')
ax.set_ylabel('Z점수 누적합')
ax.set_title('동별 Z점수 누적합 기울기 (최적 최종 개수 결정)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{BASE}\\kmedoids_03_optimal_n.png', dpi=150, bbox_inches='tight')
plt.show()

# 최종 선정
final_selected = dong_rep.head(optimal_n).copy()
print(f'\n=== 🏆 최종 선정 {optimal_n}곳 ===')
print(final_selected[['선택순위','행정동','시설명','면적','노후도',
                       '공공시설여부','Z점수','소재지주소']].to_string())

# 저장
ground.to_csv(f'{BASE}\\07_클러스터링_전체결과.csv',  index=False, encoding='utf-8-sig')
best_df.to_csv(f'{BASE}\\07_전환최적_클러스터.csv',   index=False, encoding='utf-8-sig')
dong_rep.to_csv(f'{BASE}\\07_동별대표_후보.csv',      index=False, encoding='utf-8-sig')
final_selected.to_csv(f'{BASE}\\07_최종선정.csv',     index=False, encoding='utf-8-sig')

print('\n저장 완료:')
print('kmedoids_01_elbow.png    : Elbow + Silhouette')
print('kmedoids_02_result.png   : K-medoids 클러스터링 결과')
print('kmedoids_03_optimal_n.png: 동별 Z점수 누적합')
print('07_클러스터링_전체결과.csv')
print('07_전환최적_클러스터.csv')
print('07_동별대표_후보.csv')
print(f'07_최종선정.csv ({optimal_n}곳)')
