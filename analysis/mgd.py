import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
os.makedirs('output', exist_ok=True)

# ============================================================
# 0. 기본 설정 (회귀분석 결과에서 도출된 가중치)
# ============================================================

# 아래 값은 regression_analysis_full.py 실행 결과의 가중치
# 실제 돌린 결과값으로 교체하세요
BASE_WEIGHTS = {
    '인구수':    0.545,   # 표준화 회귀계수 비율화 값
    '유동인구':  -0.142,
    '상권활성도': 0.084,
    '교통편의도': 0.084,
    '공시지가':   0.033,
    '놀이터개수': -0.112,
}

X_vars = list(BASE_WEIGHTS.keys())

# ============================================================
# 1. 데이터 로드 및 Z-score 정규화 (회귀분석 코드와 동일)
# ============================================================
def read_csv_safe(path):
    for enc in ['utf-8-sig', 'cp949', 'euc-kr', 'utf-8']:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue

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

df['상권활성도'] = df['동물병원수'] + df['동물약국수'] + df['펫샵수']
df['교통편의도'] = df['버스정류장수'] + df['역개수']

# Z-score 정규화
X_mean = df[X_vars].mean()
X_std  = df[X_vars].std()
df_z   = df.copy()
df_z[X_vars] = (df[X_vars] - X_mean) / X_std
X_norm = df_z[X_vars].values

# ============================================================
# 2. Z점수 계산 함수
# ============================================================
def calc_zscore(X_norm, weights_dict, X_vars):
    w = np.array([weights_dict[v] for v in X_vars])
    return X_norm @ w

# ============================================================
# 3. 기본 시나리오 (회귀결과 가중치)
# ============================================================
base_scores = calc_zscore(X_norm, BASE_WEIGHTS, X_vars)
df['기본_Z점수'] = base_scores
df['기본_순위']  = pd.Series(base_scores).rank(ascending=False).astype(int).values

# ============================================================
# 4. 민감도 분석 시나리오 설정
# MFC 자료와 동일하게 단독 변화 방식
# 각 가중치를 하나씩 ±10%, ±20% 조정, 나머지는 고정
# ============================================================

# 조정 비율 설정
adjustments = [+0.10, +0.20, -0.10, -0.20]

scenarios = []

# 기본 시나리오 먼저 추가
scenarios.append({
    'scenario_id': 'BASE',
    'scenario_name': '기본 (회귀결과)',
    '조정변수': '-',
    '조정방향': '-',
    '조정비율': '0%',
    **{f'{v}_가중치': round(BASE_WEIGHTS[v], 4) for v in X_vars}
})

# 각 변수별 ±10%, ±20% 단독 변화
for var in X_vars:
    for adj in adjustments:
        new_weights = BASE_WEIGHTS.copy()
        new_weights[var] = round(BASE_WEIGHTS[var] * (1 + adj), 4)

        direction = '+' if adj > 0 else '-'
        pct       = f'{direction}{int(abs(adj)*100)}%'
        sid       = f'{var}_{pct}'

        scenarios.append({
            'scenario_id': sid,
            'scenario_name': f'{var} {pct} 조정',
            '조정변수': var,
            '조정방향': direction,
            '조정비율': pct,
            **{f'{v}_가중치': round(new_weights[v], 4) for v in X_vars}
        })

print(f'총 시나리오 수: {len(scenarios)}개 (기본 1 + 변수별 4 × {len(X_vars)}개)')

# ============================================================
# 5. 시나리오별 Z점수 및 순위 계산
# ============================================================

# 결과 저장용
all_results  = []   # 시나리오별 행정동 Z점수 + 순위
rank_changes = []   # 순위 변동 요약

base_rank = df[['행정동', '기본_순위']].set_index('행정동')['기본_순위'].to_dict()

for sc in scenarios:
    w_dict = {v: sc[f'{v}_가중치'] for v in X_vars}
    scores = calc_zscore(X_norm, w_dict, X_vars)
    ranks  = pd.Series(scores).rank(ascending=False).astype(int).values

    for i, (dong, score, rank) in enumerate(zip(df['행정동'], scores, ranks)):
        all_results.append({
            'scenario_id':   sc['scenario_id'],
            'scenario_name': sc['scenario_name'],
            '조정변수':      sc['조정변수'],
            '조정비율':      sc['조정비율'],
            '행정동':        dong,
            'Z점수':         round(score, 4),
            '순위':          rank,
            '기본순위':      base_rank[dong],
            '순위변동':      base_rank[dong] - rank   # 양수 = 상승, 음수 = 하락
        })

results_df = pd.DataFrame(all_results)

# ============================================================
# 6. 상위 10개 동 순위 변동 분석
# ============================================================
top10_dongs = df.nsmallest(10, '기본_순위')['행정동'].tolist()

# 상위 10개 동의 시나리오별 순위만 추출
top10_pivot = results_df[results_df['행정동'].isin(top10_dongs)].pivot_table(
    index='scenario_id',
    columns='행정동',
    values='순위',
    aggfunc='first'
)
top10_pivot = top10_pivot[top10_dongs]   # 기본순위대로 컬럼 정렬

# ============================================================
# 7. 순위 안정성 요약
# ============================================================
stability_rows = []

for dong in top10_dongs:
    dong_data  = results_df[results_df['행정동'] == dong]
    base_r     = base_rank[dong]
    ranks_all  = dong_data['순위'].values
    max_change = int(np.max(np.abs(ranks_all - base_r)))
    stable     = '✅ 안정' if max_change <= 2 else ('⚠️ 주의' if max_change <= 5 else '❌ 불안정')

    stability_rows.append({
        '행정동':        dong,
        '기본순위':      base_r,
        '최대순위변동':  max_change,
        '안정성':        stable,
        '최고순위':      int(ranks_all.min()),
        '최저순위':      int(ranks_all.max()),
    })

stability_df = pd.DataFrame(stability_rows).sort_values('기본순위')

# ============================================================
# 8. CSV 저장
# ============================================================

# CSV 6-1: 시나리오 설정표
sc_df = pd.DataFrame(scenarios)
sc_df.to_csv('output/06_민감도_시나리오설정.csv', index=False, encoding='utf-8-sig')
print('\n저장: output/06_민감도_시나리오설정.csv')

# CSV 6-2: 전체 시나리오별 Z점수 + 순위
results_df.to_csv('output/06_민감도_전체결과.csv', index=False, encoding='utf-8-sig')
print('저장: output/06_민감도_전체결과.csv')

# CSV 6-3: 상위 10개 동 피벗 (시나리오 × 행정동 순위표)
top10_pivot.to_csv('output/06_민감도_상위10동_순위변동.csv', encoding='utf-8-sig')
print('저장: output/06_민감도_상위10동_순위변동.csv')

# CSV 6-4: 안정성 요약
stability_df.to_csv('output/06_민감도_안정성요약.csv', index=False, encoding='utf-8-sig')
print('저장: output/06_민감도_안정성요약.csv')

# ============================================================
# 9. 결과 출력
# ============================================================
print('\n' + '='*65)
print('【 민감도 분석 결과 】')
print('='*65)
print(f'총 시나리오: {len(scenarios)}개')
print(f'(기본 1개 + 변수 {len(X_vars)}개 × 조정 4회)')
print()

print('=== 상위 10개 행정동 순위 안정성 ===')
print(stability_df.to_string(index=False))

print()
print('=== 시나리오별 상위 10동 순위표 ===')
print(top10_pivot.to_string())




# ============================================================
# 10. 시각화: 상위 3개 동 순위 변동 그래프
# ============================================================

# 상위 3개 동 선택
top3_dongs = top10_dongs[:3]

fig, axes = plt.subplots(1, len(top3_dongs), figsize=(15, 5), sharey=True)

# axes가 하나일 경우 대비
if len(top3_dongs) == 1:
    axes = [axes]

for ax, dong in zip(axes, top3_dongs):
    dong_data = results_df[results_df['행정동'] == dong].copy()

    # 기본 시나리오 제외
    adj_data = dong_data[dong_data['scenario_id'] != 'BASE']

    # 기본 순위 (빨간 점선)
    ax.axhline(
        y=base_rank[dong],
        color='red',
        linestyle='--',
        linewidth=2,
        label='기본순위'
    )

    # 시나리오별 순위 점
    ax.scatter(
        range(len(adj_data)),
        adj_data['순위'],
        s=40,
        zorder=5
    )

    ax.set_title(dong, fontsize=12, fontweight='bold')
    ax.set_xlabel('시나리오', fontsize=9)

    # 첫 번째 그래프만 y축 표시
    if dong == top3_dongs[0]:
        ax.set_ylabel('순위', fontsize=10)

    ax.invert_yaxis()  # 순위 낮을수록 위
    ax.set_xticks([])
    ax.legend(fontsize=8)

plt.suptitle(
    '상위 3개 행정동 민감도 분석 - 순위 변동',
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()
plt.savefig('output/06_민감도_순위변동_그래프_top3.png', dpi=150)
plt.close()

print('\n그래프 저장: output/06_민감도_순위변동_그래프_top3.png')