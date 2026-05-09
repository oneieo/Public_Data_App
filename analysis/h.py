import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 0. 파일 로드 함수
# ============================================================
def read_csv_safe(path):
    for enc in ['utf-8-sig', 'cp949', 'euc-kr', 'utf-8']:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue
    raise ValueError(f"인코딩 확인 필요: {path}")

import os
os.makedirs('output', exist_ok=True)


# ============================================================
# 1. 데이터 로드
# ============================================================
population = read_csv_safe("data/final_population.csv")
move_pop   = read_csv_safe("data/move_population.csv")
hospital   = read_csv_safe("data/final_animal_hospital.csv")
pharmacy   = read_csv_safe("data/final_animal_permercy.csv")
pet_shop   = read_csv_safe("data/pet_shop.csv")
bus        = read_csv_safe("data/bus_count.csv")
subway     = read_csv_safe("data/subway_station_count.csv")
money      = read_csv_safe("data/money.csv")
playground = read_csv_safe("data/final_playground_count.csv")
pet_reg    = read_csv_safe("data/final_pet_registration.csv")
pet_park   = read_csv_safe("data/final_pet_park_count.csv")

# ============================================================
# 2. 전처리 및 집계
# ============================================================
hospital_cnt = hospital.groupby('행정동').size().reset_index(name='동물병원수')
pharmacy_cnt = pharmacy.groupby('행정동').size().reset_index(name='동물약국수')

move_col  = [c for c in move_pop.columns if c != '행정동'][0]
move_pop  = move_pop.rename(columns={move_col: '유동인구'})
shop_col  = [c for c in pet_shop.columns if c not in ['행정동']][0]
pet_shop  = pet_shop[['행정동', shop_col]].rename(columns={shop_col: '펫샵수'})
bus_col   = [c for c in bus.columns if c != '행정동'][0]
bus       = bus.rename(columns={bus_col: '버스정류장수'})
sub_col   = [c for c in subway.columns if c != '행정동'][0]
subway    = subway.rename(columns={sub_col: '역개수'})
play_col  = [c for c in playground.columns if c != '행정동'][0]
playground = playground.rename(columns={play_col: '놀이터개수'})
park_col  = [c for c in pet_park.columns if c != '행정동'][0]
pet_park  = pet_park.rename(columns={park_col: '펫파크수'})
pet_reg   = pet_reg[['행정동', '합계']].rename(columns={'합계': '반려동물등록수'})
money     = money[['행정동', '공시지가_평균']].rename(columns={'공시지가_평균': '공시지가'})

# ============================================================
# 3. 데이터 병합
# ============================================================
df = population[['행정동', '인구수']].copy()
df = df.merge(move_pop[['행정동', '유동인구']],    on='행정동', how='left')
df = df.merge(hospital_cnt,                        on='행정동', how='left')
df = df.merge(pharmacy_cnt,                        on='행정동', how='left')
df = df.merge(pet_shop,                            on='행정동', how='left')
df = df.merge(bus[['행정동', '버스정류장수']],      on='행정동', how='left')
df = df.merge(subway[['행정동', '역개수']],         on='행정동', how='left')
df = df.merge(money,                               on='행정동', how='left')
df = df.merge(playground[['행정동', '놀이터개수']], on='행정동', how='left')
df = df.merge(pet_reg,                             on='행정동', how='left')
df = df.merge(pet_park,                            on='행정동', how='left')
df = df.fillna(0)

for col in df.columns:
    if col != '행정동':
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

# ============================================================
# 4. 파생변수 생성
# ============================================================
df['상권활성도'] = df['동물병원수'] + df['동물약국수'] + df['펫샵수']
df['교통편의도'] = df['버스정류장수'] + df['역개수']
df['Y'] = df['반려동물등록수'] / (df['펫파크수'] + 1)

X_vars = ['인구수', '유동인구', '상권활성도', '교통편의도', '공시지가', '놀이터개수']
Y_var  = 'Y'

print(f"✅ 데이터셋 완성: {len(df)}개 행정동")

# ============================================================
# [CSV 1] Z-score 정규화 결과
# ============================================================
print('\n' + '='*60)
print('【 CSV 1: Z-score 정규화 결과 저장 】')
print('='*60)

data_zscore = df[['행정동'] + X_vars + [Y_var]].copy()
X_mean = data_zscore[X_vars].mean()
X_std  = data_zscore[X_vars].std()
data_zscore[X_vars] = (data_zscore[X_vars] - X_mean) / X_std

# 컬럼명에 _Z 붙여서 구분
zscore_cols = {v: f'{v}_Z' for v in X_vars}
csv1 = data_zscore.rename(columns=zscore_cols)
csv1.to_csv('output/01_Z-score_정규화.csv', index=False, encoding='utf-8-sig')
print('저장 완료: output/01_Z-score_정규화.csv')
print(csv1.head(3).to_string())

# ============================================================
# [CSV 3] 다중회귀분석 결과
# ============================================================
print('\n' + '='*60)
print('【 CSV 3: 다중회귀분석 결과 저장 】')
print('='*60)

data_reg = df[X_vars + [Y_var]].copy()
data_reg[X_vars] = (data_reg[X_vars] - X_mean) / X_std  # Z-score 정규화 적용

n, k = len(data_reg), len(X_vars)
X    = data_reg[X_vars].values
y    = data_reg[Y_var].values
X_b  = np.column_stack([np.ones(n), X])

beta      = np.linalg.lstsq(X_b, y, rcond=None)[0]
y_hat     = X_b @ beta
residuals = y - y_hat

ss_tot = np.sum((y - np.mean(y))**2)
ss_res = np.sum(residuals**2)
ss_reg = ss_tot - ss_res
r2     = 1 - ss_res/ss_tot
adj_r2 = 1 - (1-r2)*(n-1)/(n-k-1)
F      = (ss_reg/k) / (ss_res/(n-k-1))
p_F    = 1 - stats.f.cdf(F, k, n-k-1)

mse      = ss_res / (n-k-1)
cov_beta = mse * np.linalg.inv(X_b.T @ X_b)
se       = np.sqrt(np.diag(cov_beta))
t_stats  = beta / se
p_vals   = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k-1))

# 모델 요약 출력
print(f'R²: {r2:.4f} | 수정R²: {adj_r2:.4f} | F: {F:.4f} | F p-value: {p_F:.6f} {"✅" if p_F < 0.05 else "❌"}')

# 회귀분석 결과 CSV
reg_rows = []
for label, b, s, t, p in zip(['절편'] + X_vars, beta, se, t_stats, p_vals):
    sig  = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ''))
    reg_rows.append({
        '변수': label,
        '비표준화_회귀계수': round(b, 4),
        '표준오차': round(s, 4),
        't통계량': round(t, 4),
        'p_value': round(p, 4),
        '유의성': sig,
        '유의여부': 'O' if p < 0.05 else 'X'
    })

# 모델 요약 행 추가
summary_row = {
    '변수': '[ 모델 요약 ]',
    '비표준화_회귀계수': '',
    '표준오차': '',
    't통계량': '',
    'p_value': round(p_F, 6),
    '유의성': f'R²={r2:.4f}',
    '유의여부': f'수정R²={adj_r2:.4f} | F={F:.4f}'
}

csv3 = pd.DataFrame(reg_rows)
csv3.to_csv('output/03_다중회귀분석_결과.csv', index=False, encoding='utf-8-sig')
print('저장 완료: output/03_다중회귀분석_결과.csv')
print(csv3.to_string())

# ============================================================
# [CSV 4] 비표준화 → 표준화 회귀계수 → 가중치 비율화
# ============================================================
print('\n' + '='*60)
print('【 CSV 4: 표준화 회귀계수 → 가중치 비율화 저장 】')
print('='*60)

std_beta   = beta[1:]                              # 절편 제외
abs_sum    = np.sum(np.abs(std_beta))
weights    = std_beta / abs_sum                    # 비율 정규화
weights_pos = np.abs(std_beta) / np.sum(np.abs(std_beta))  # 절댓값 기준 비율

csv4_rows = []
for v, raw_b, std_b, w, wp in zip(X_vars,
                                    beta[1:],      # 비표준화 (Z-score 정규화 후라 곧 표준화)
                                    std_beta,
                                    weights,
                                    weights_pos):
    csv4_rows.append({
        '변수': v,
        '비표준화_회귀계수': round(raw_b, 4),
        '표준화_회귀계수': round(std_b, 4),
        '가중치_부호포함(%)': round(w * 100, 2),
        '가중치_절댓값비율(%)': round(wp * 100, 2),
        '방향': '↑ 높을수록 유리' if w > 0 else '↓ 낮을수록 유리'
    })

csv4 = pd.DataFrame(csv4_rows)
csv4.to_csv('output/04_가중치_비율화.csv', index=False, encoding='utf-8-sig')
print('저장 완료: output/04_가중치_비율화.csv')
print(csv4.to_string())

# ============================================================
# [CSV 5] 최종 Z점수 (가중치 × Z-score 합산) 결과
# ============================================================
print('\n' + '='*60)
print('【 CSV 5: 행정동별 최종 Z점수 및 순위 저장 】')
print('='*60)

# Z-score 정규화된 X값에 표준화 회귀계수(가중치) 곱해서 Z점수 산출
X_norm   = data_reg[X_vars].values  # 이미 Z-score 정규화됨
Z_scores = X_norm @ std_beta

# 각 변수별 기여값 계산
contrib_df = pd.DataFrame(X_norm * std_beta,
                           columns=[f'{v}_기여값' for v in X_vars])

csv5 = df[['행정동', 'Y', '반려동물등록수', '펫파크수']].copy()

# 원본값도 포함
for v in X_vars:
    csv5[f'{v}_원본'] = df[v].values

# Z-score 정규화값 포함
z_df = pd.DataFrame(X_norm, columns=[f'{v}_Z값' for v in X_vars])
csv5 = pd.concat([csv5.reset_index(drop=True), z_df], axis=1)

# 각 변수 기여값 포함
csv5 = pd.concat([csv5, contrib_df], axis=1)

# 최종 Z점수
csv5['Z점수'] = Z_scores
csv5['순위'] = csv5['Z점수'].rank(ascending=False).astype(int)
csv5 = csv5.sort_values('순위').reset_index(drop=True)

csv5.to_csv('output/05_최종_Z점수_순위.csv', index=False, encoding='utf-8-sig')
print('저장 완료: output/05_최종_Z점수_순위.csv')
print()
print('=== 행정동별 최종 순위 ===')
print(csv5[['순위', '행정동', 'Y', '반려동물등록수', 'Z점수']].to_string())
print()
print(f'📍 상위 10개 행정동:')
print(csv5[['순위', '행정동', 'Z점수']].head(10).to_string())

print('\n' + '='*60)
print('✅ 전체 CSV 저장 완료!')
print('='*60)
print('01_Z-score_정규화.csv      : Z-score 정규화된 변수값')
print('02_상관분석_행렬.csv        : 변수 간 상관계수 행렬')
print('02_상관분석_쌍목록.csv      : 변수 쌍별 상관계수 + 판단')
print('02_상관분석_히트맵.png      : 히트맵 이미지')
print('03_다중회귀분석_결과.csv    : 회귀계수, t값, p-value')
print('04_가중치_비율화.csv        : 표준화계수 → 가중치 변환')
print('05_최종_Z점수_순위.csv      : 행정동별 Z점수 + 변수별 기여값 + 순위')


# ============================================================
# [시각화] 정규화 + 파생변수 + 가중치 + 최종 Z점수
# ============================================================
X_vars = [
    '인구수',
    '유동인구',
    '상권활성도',
    '교통편의도',
    '놀이터개수'
]

import os

os.makedirs('output', exist_ok=True)

# ============================================================
# 1. 정규화 전/후 분포 비교
# ============================================================

print('\n' + '='*60)
print('【 시각화 1: 정규화 전후 비교 】')
print('='*60)

for v in X_vars:

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ---------------------------------
    # 정규화 전
    # ---------------------------------

    sns.histplot(
        df[v],
        kde=True,
        ax=axes[0]
    )

    axes[0].set_title(f'{v} - 정규화 전')
    axes[0].set_xlabel(v)

    # ---------------------------------
    # 정규화 후
    # ---------------------------------

    sns.histplot(
        data_zscore[v],
        kde=True,
        ax=axes[1]
    )

    axes[1].set_title(f'{v} - Z-score 정규화 후')
    axes[1].set_xlabel(f'{v}_Z')

    plt.tight_layout()

    plt.savefig(
        f'output/시각화_정규화전후_{v}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

print('✅ 정규화 전후 시각화 저장 완료')


# ============================================================
# 2. 상권활성도 생성 구조 시각화
# ============================================================

print('\n' + '='*60)
print('【 시각화 2: 상권활성도 생성 구조 】')
print('='*60)

derived_df1 = pd.DataFrame({
    '변수': [
        '동물병원수',
        '동물약국수',
        '펫샵수',
        '상권활성도'
    ],
    '평균값': [
        df['동물병원수'].mean(),
        df['동물약국수'].mean(),
        df['펫샵수'].mean(),
        df['상권활성도'].mean()
    ]
})

plt.figure(figsize=(8, 5))

sns.barplot(
    data=derived_df1,
    x='변수',
    y='평균값'
)

plt.title('상권활성도 파생변수 생성 구조')

plt.tight_layout()

plt.savefig(
    'output/시각화_상권활성도_생성구조.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print('✅ 상권활성도 생성 구조 저장 완료')


# ============================================================
# 3. 교통편의도 생성 구조 시각화
# ============================================================

print('\n' + '='*60)
print('【 시각화 3: 교통편의도 생성 구조 】')
print('='*60)

derived_df2 = pd.DataFrame({
    '변수': [
        '버스정류장수',
        '역개수',
        '교통편의도'
    ],
    '평균값': [
        df['버스정류장수'].mean(),
        df['역개수'].mean(),
        df['교통편의도'].mean()
    ]
})

plt.figure(figsize=(8, 5))

sns.barplot(
    data=derived_df2,
    x='변수',
    y='평균값'
)

plt.title('교통편의도 파생변수 생성 구조')

plt.tight_layout()

plt.savefig(
    'output/시각화_교통편의도_생성구조.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print('✅ 교통편의도 생성 구조 저장 완료')


# ============================================================
# 4. 변수별 가중치 시각화
# ============================================================

print('\n' + '='*60)
print('【 시각화 4: 변수별 가중치 】')
print('='*60)

plt.figure(figsize=(10, 5))

sns.barplot(
    data=csv4,
    x='변수',
    y='가중치_절댓값비율(%)'
)

plt.title('변수별 최종 가중치 비율')

plt.ylabel('가중치 (%)')

plt.tight_layout()

plt.savefig(
    'output/시각화_변수별_가중치.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print('✅ 변수별 가중치 시각화 저장 완료')


# ============================================================
# 5. 상위 10개 행정동 Z점수 시각화
# ============================================================

print('\n' + '='*60)
print('【 시각화 5: 상위 10개 행정동 Z점수 】')
print('='*60)

top10 = csv5.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top10,
    x='Z점수',
    y='행정동'
)

plt.title('상위 10개 행정동 최종 Z점수')

plt.tight_layout()

plt.savefig(
    'output/시각화_상위10개_Z점수.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print('✅ 상위 10개 행정동 Z점수 저장 완료')


# ============================================================
# 6. 변수별 기여값 히트맵
# ============================================================

print('\n' + '='*60)
print('【 시각화 6: 변수별 기여값 히트맵 】')
print('='*60)

contrib_cols = [f'{v}_기여값' for v in X_vars]

heatmap_df = csv5[
    ['행정동'] + contrib_cols
].head(10)

heatmap_data = heatmap_df.set_index('행정동')

plt.figure(figsize=(10, 6))

sns.heatmap(
    heatmap_data,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('상위 10개 행정동 변수별 기여값 히트맵')

plt.tight_layout()

plt.savefig(
    'output/시각화_변수별_기여값_히트맵.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print('✅ 변수별 기여값 히트맵 저장 완료')


# ============================================================
# 완료 메시지
# ============================================================

print('\n' + '='*60)
print('✅ 전체 시각화 완료')
print('='*60)

print('저장 파일 목록')
print('1. 시각화_정규화전후_변수명.png')
print('2. 시각화_상권활성도_생성구조.png')
print('3. 시각화_교통편의도_생성구조.png')
print('4. 시각화_변수별_가중치.png')
print('5. 시각화_상위10개_Z점수.png')
print('6. 시각화_변수별_기여값_히트맵.png')

print('\n📁 저장 위치: output 폴더')