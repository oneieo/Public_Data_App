import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# BASE = r'C:\Users\오형종\OneDrive - 전북대학교\바탕 화면\project'/
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(BASE, exist_ok=True)

# ============================================================
# 공통 함수: Convex Hull 경계선 (PDF 스타일)
# ============================================================
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
# 1. 파일 불러오기
# ============================================================
ground = pd.read_csv(f'{BASE}\\data\\final_playground.csv', encoding='utf-8-sig')
pet    = pd.read_csv(f'{BASE}\\data\\final_pet_park_.csv',  encoding='utf-8-sig')
zscore = pd.read_csv(f'{BASE}\\data\\zscore.csv',           encoding='utf-8-sig')

# ============================================================
# [1단계] Z점수 역할: 23개 동 선별 (필터링)
# → 순위가 23위까지 포함되는지만 결정
# → Z점수는 여기서만 필터로 사용
# ============================================================
top23 = zscore[zscore['순위'] <= 23][['행정동', 'Z점수', '순위']].copy()
ground = ground[ground['행정동'].isin(top23['행정동'])].copy().reset_index(drop=True)
print(f'[1단계] 상위 23개 동 선별 완료 → 놀이터: {len(ground)}개')

# ============================================================
# 2. 기본 변수 생성
# ============================================================
ground['설치연도'] = pd.to_datetime(ground['설치일자'], errors='coerce').dt.year
ground['노후도']   = (2025 - ground['설치연도']).clip(lower=0)
ground['노후도']   = ground['노후도'].fillna(ground['노후도'].median())

before = len(ground)
ground = ground[ground['노후도'] > 0].copy().reset_index(drop=True)
print(f'노후도 0 제외: {before - len(ground)}개 → {len(ground)}개')

ground['위도'] = pd.to_numeric(ground['위도'], errors='coerce')
ground['경도'] = pd.to_numeric(ground['경도'], errors='coerce')
pet['위도']    = pd.to_numeric(pet['위도'], errors='coerce')
pet['경도']    = pd.to_numeric(pet['경도'], errors='coerce')

ground = ground.dropna(subset=['위도', '경도']).copy().reset_index(drop=True)
pet    = pet.dropna(subset=['위도', '경도']).copy()

# 펫존거리 계산
ground_coords  = np.radians(ground[['위도', '경도']].values)
pet_coords     = np.radians(pet[['위도', '경도']].values)
dist_matrix    = haversine_distances(ground_coords, pet_coords) * 6371
ground['펫존거리'] = dist_matrix.min(axis=1)

ground['공공시설여부'] = ground['민간_공공구분'].map({'공공': 1, '민간': 0})
ground['면적'] = pd.to_numeric(ground['면적(제곱미터)'], errors='coerce')
ground['면적'] = ground['면적'].fillna(ground['면적'].median())

# ============================================================
# [2단계] 클러스터링: Z점수 없이 놀이터 자체 특성만 사용
# → "어떤 특성의 놀이터가 전환에 적합한가?"만 판단
# → Z점수(동 수준 변수)는 개별 놀이터 특성과 무관
# ============================================================
print('\n[2단계] 클러스터링 시작 (놀이터 자체 특성만 사용)')
features = ['공공시설여부', '면적', '노후도', '펫존거리']
X = ground[features].copy().fillna(ground[features].mean(numeric_only=True))

print('\n=== PCA 변수 기초통계 ===')
print(X.describe().round(2))

# MinMaxScaler 정규화
scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# PCA (4차원 → 2차원)
pca        = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
explained  = pca.explained_variance_ratio_

ground['PC1'] = pca_result[:, 0]
ground['PC2'] = pca_result[:, 1]

loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=features
).round(3)

print('\n=== PCA 결과 ===')
print(f'PC1 설명력: {explained[0]*100:.1f}%')
print(f'PC2 설명력: {explained[1]*100:.1f}%')
print(f'누적 설명력: {explained.sum()*100:.1f}%')
print('\n=== 주성분 구성 ===')
print(loadings)

# Elbow Method
X_pca = pca_result
inertias, silhouettes = [], []
K_range = range(2, 7)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_pca, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(K_range), inertias, 'o-', color='steelblue', lw=2, ms=8)
ax1.set_xlabel('K (군집 수)'); ax1.set_ylabel('Inertia (SSE)')
ax1.set_title('Elbow Method'); ax1.grid(alpha=0.3)
for k, v in zip(K_range, inertias):
    ax1.annotate(f'{v:.1f}', (k, v), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=9)

ax2.plot(list(K_range), silhouettes, 's-', color='tomato', lw=2, ms=8)
ax2.set_xlabel('K (군집 수)'); ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score'); ax2.grid(alpha=0.3)
for k, v in zip(K_range, silhouettes):
    ax2.annotate(f'{v:.3f}', (k, v), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=9)

plt.suptitle('최적 군집 수 결정 (Elbow + Silhouette)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}\\clustering_01_elbow.png', dpi=150, bbox_inches='tight')
plt.show()

optimal_k = list(K_range)[silhouettes.index(max(silhouettes))]
print(f'\n최적 K = {optimal_k}')

# K-medoids 구현
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

# 3가지 방법 비교
colors        = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
fig, axes     = plt.subplots(1, 3, figsize=(20, 6))
method_labels = {}
method_verdict= {}

for ax, method in zip(axes, ['K-means', 'GMM', 'K-medoids']):
    if method == 'K-means':
        labels = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit_predict(X_pca)
    elif method == 'GMM':
        labels = GaussianMixture(n_components=optimal_k, random_state=42).fit_predict(X_pca)
    else:
        labels = kmedoids(X_pca, optimal_k)

    method_labels[method] = labels
    sizes   = [int(np.sum(labels == i)) for i in range(optimal_k)]
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

    ratio = max(sizes) / (min(sizes) + 1e-9)
    if ratio > 4:
        verdict     = f'기각 (불균형 {ratio:.1f}배)'
        title_color = 'red'
        method_verdict[method] = '❌ 기각'
    else:
        verdict     = '채택 (균형 양호)'
        title_color = 'blue'
        method_verdict[method] = '✅ 채택'

    ax.set_xlabel(f'PC1 ({explained[0]*100:.0f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({explained[1]*100:.0f}%)', fontsize=10)
    ax.set_title(f'{method}  [{verdict}]\n군집 크기: {sizes}',
                 color=title_color, fontweight='bold')
    ax.legend(handles=handles, fontsize=8); ax.grid(alpha=0.3)
    print(f'{method}: {method_verdict[method]}, 크기={sizes}')

plt.suptitle(f'클러스터링 방법 비교 (K={optimal_k})',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}\\clustering_02_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 전환 최적 클러스터 선정
final_labels       = method_labels['K-medoids']
ground['클러스터'] = final_labels + 1

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

# 최종 시각화
fig, ax = plt.subplots(figsize=(10, 8))
handles = []
for i in range(optimal_k):
    mask        = final_labels == i
    cluster_num = i + 1
    pts         = X_pca[mask]
    color       = colors[i]
    if cluster_num == best_cluster:
        label = f'클러스터 {cluster_num} ({mask.sum()}개)  ← 전환 최적'
        s, ew = 60, 0.5
        draw_convex_hull(ax, pts, color, alpha_fill=0.25, alpha_line=0.9)
    else:
        label = f'클러스터 {cluster_num} ({mask.sum()}개)'
        s, ew = 25, 0.3
        draw_convex_hull(ax, pts, color, alpha_fill=0.10, alpha_line=0.5)
    ax.scatter(pts[:, 0], pts[:, 1], c=color, s=s,
               alpha=0.85, edgecolors='white', linewidths=ew, zorder=3)
    handles.append(mpatches.Patch(color=color, alpha=0.7, label=label))

best_mask = final_labels == best_cluster - 1
cx, cy = X_pca[best_mask, 0].mean(), X_pca[best_mask, 1].mean()
ax.scatter(cx, cy, s=300, marker='*', c='gold',
           edgecolors='black', linewidths=1.5, zorder=10)
handles.append(mpatches.Patch(color='gold', label='전환 최적 클러스터 중심 ★'))
ax.set_xlabel(f'PC1 ({explained[0]*100:.0f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({explained[1]*100:.0f}%)', fontsize=11)
ax.set_title('K-medoids 클러스터링 최종 결과 (Convex Hull)',
             fontsize=13, fontweight='bold')
ax.legend(handles=handles, fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{BASE}\\clustering_03_final.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# [3단계] Z점수 재활용: 최종 개수 결정
# → 전환 최적 클러스터에서 동별 대표 1개 추출
# → 동별 Z점수 순으로 정렬
# → 누적합 기울기로 최적 개수 결정
# ============================================================
print('\n[3단계] Z점수로 최종 개수 결정 시작')

best_df = ground[ground['클러스터'] == best_cluster].copy()

# 동별 대표 1개 추출 (면적 최대)
dong_rep = best_df.sort_values('면적', ascending=False)\
                  .drop_duplicates('행정동').copy()

# 동별 Z점수 병합
dong_rep = dong_rep.merge(top23[['행정동', 'Z점수', '순위']],
                          on='행정동', how='left')

# Z점수 순으로 정렬 (1위 동부터)
dong_rep = dong_rep.sort_values('Z점수', ascending=False).reset_index(drop=True)
dong_rep['선택순위'] = dong_rep.index + 1
dong_rep['누적합']   = dong_rep['Z점수'].cumsum()
dong_rep['기울기']   = dong_rep['Z점수']  # 각 순위에서의 Z점수 증가량

print(f'\n전환 최적 클러스터 내 동별 대표: {len(dong_rep)}개')
print(dong_rep[['행정동','순위','Z점수','면적','노후도','공공시설여부']].to_string())

# 누적합 기울기로 최적 개수 결정
threshold    = dong_rep['기울기'].max() * 0.05
optimal_rows = dong_rep[dong_rep['기울기'] < threshold]
if len(optimal_rows) > 0:
    optimal_n = int(optimal_rows.index[0])
else:
    optimal_n = len(dong_rep)

print(f'\n=== 누적합 기울기 기준 최적 개수: {optimal_n}개 ===')

# 누적합 그래프
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dong_rep['선택순위'], dong_rep['누적합'],
        'o-', color='steelblue', markersize=7, linewidth=2)
ax.axvline(x=optimal_n, color='red', linestyle='--', linewidth=2,
           label=f'최적 개수: {optimal_n}개')
ax.axvspan(optimal_n - 0.5, optimal_n + 0.5, color='gray', alpha=0.2)

# 각 점에 동 이름 표시
for _, row in dong_rep.iterrows():
    ax.annotate(row['행정동'],
                (row['선택순위'], row['누적합']),
                textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=7, rotation=45)

ax.set_xlabel('동 선택 개수 (Z점수 순위 기준)')
ax.set_ylabel('Z점수 누적합')
ax.set_title('동별 Z점수 누적합 기울기\n(최적 최종 개수 결정)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{BASE}\\clustering_04_optimal_n.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 최종 선정
# ============================================================
final_selected = dong_rep.head(optimal_n).copy()

print(f'\n=== 🏆 최종 선정 {optimal_n}곳 ===')
print(final_selected[['선택순위', '행정동', '시설명', '면적',
                       '노후도', '공공시설여부', 'Z점수', '소재지주소']].to_string())

# ============================================================
# 저장
# ============================================================
ground.to_csv(f'{BASE}\\07_클러스터링_결과.csv',         index=False, encoding='utf-8-sig')
best_df.to_csv(f'{BASE}\\07_전환최적_클러스터.csv',      index=False, encoding='utf-8-sig')
dong_rep.to_csv(f'{BASE}\\07_동별대표_후보.csv',         index=False, encoding='utf-8-sig')
final_selected.to_csv(f'{BASE}\\07_최종선정.csv',        index=False, encoding='utf-8-sig')
loadings.to_csv(f'{BASE}\\07_PCA_주성분구성.csv',        encoding='utf-8-sig')

print('\n' + '='*60)
print('✅ 저장 완료!')
print('='*60)
print('07_클러스터링_결과.csv       : 639개 전체 클러스터 배정')
print('07_전환최적_클러스터.csv     : 전환 최적 클러스터 시설 목록')
print('07_동별대표_후보.csv         : 동별 대표 후보 + Z점수 순위')
print(f'07_최종선정.csv              : 최종 선정 {optimal_n}곳')
print('07_PCA_주성분구성.csv        : PC1/PC2 변수 기여도')
print()
print('clustering_01_elbow.png      : Elbow + Silhouette')
print('clustering_02_comparison.png : K-means/GMM/K-medoids 비교')
print('clustering_03_final.png      : K-medoids 최종 결과')
print('clustering_04_optimal_n.png  : 동별 Z점수 누적합 그래프')
