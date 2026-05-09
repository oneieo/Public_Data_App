import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ============================================================
# 1. 데이터 로드
# ============================================================
population = read_csv_safe(os.path.join(DATA_DIR, 'final_population.csv'))
move_pop   = read_csv_safe(os.path.join(DATA_DIR, 'move_population.csv'))
hospital   = read_csv_safe(os.path.join(DATA_DIR, 'final_animal_hospital.csv'))
pharmacy   = read_csv_safe(os.path.join(DATA_DIR, 'final_animal_permercy.csv'))
pet_shop   = read_csv_safe(os.path.join(DATA_DIR, 'pet_shop.csv'))
bus        = read_csv_safe(os.path.join(DATA_DIR, 'bus_count.csv'))
subway     = read_csv_safe(os.path.join(DATA_DIR, 'subway_station_count.csv'))
money      = read_csv_safe(os.path.join(DATA_DIR, 'money.csv'))
playground = read_csv_safe(os.path.join(DATA_DIR, 'final_playground_count.csv'))
pet_reg    = read_csv_safe(os.path.join(DATA_DIR, 'final_pet_registration.csv'))
pet_park   = read_csv_safe(os.path.join(DATA_DIR, 'final_pet_park_count.csv'))

# ============================================================
# 2. 전처리 및 집계
# ============================================================
hospital_cnt = hospital.groupby('행정동').size().reset_index(name='동물병원수')
pharmacy_cnt = pharmacy.groupby('행정동').size().reset_index(name='동물약국수')

move_col   = [c for c in move_pop.columns   if c != '행정동'][0]
move_pop   = move_pop.rename(columns={move_col: '유동인구'})

shop_col   = [c for c in pet_shop.columns   if c != '행정동'][0]
pet_shop   = pet_shop[['행정동', shop_col]].rename(columns={shop_col: '펫샵수'})

bus_col    = [c for c in bus.columns        if c != '행정동'][0]
bus        = bus.rename(columns={bus_col: '버스정류장수'})

sub_col    = [c for c in subway.columns     if c != '행정동'][0]
subway     = subway.rename(columns={sub_col: '역개수'})

play_col   = [c for c in playground.columns if c != '행정동'][0]
playground = playground.rename(columns={play_col: '놀이터개수'})

park_col   = [c for c in pet_park.columns   if c != '행정동'][0]
pet_park   = pet_park.rename(columns={park_col: '펫파크수'})

pet_reg    = pet_reg[['행정동', '합계']].rename(columns={'합계': '반려동물등록수'})
money      = money[['행정동', '공시지가_평균']].rename(columns={'공시지가_평균': '공시지가'})

# ============================================================
# 3. 데이터 병합
# ============================================================
df = population[['행정동', '인구수']].copy()
df = df.merge(move_pop[['행정동', '유동인구']],     on='행정동', how='left')
df = df.merge(hospital_cnt,                         on='행정동', how='left')
df = df.merge(pharmacy_cnt,                         on='행정동', how='left')
df = df.merge(pet_shop,                             on='행정동', how='left')
df = df.merge(bus[['행정동', '버스정류장수']],       on='행정동', how='left')
df = df.merge(subway[['행정동', '역개수']],          on='행정동', how='left')
df = df.merge(money,                                on='행정동', how='left')
df = df.merge(playground[['행정동', '놀이터개수']], on='행정동', how='left')
df = df.merge(pet_reg,                              on='행정동', how='left')
df = df.merge(pet_park,                             on='행정동', how='left')
df = df.fillna(0)

# 콤마 제거 및 숫자 변환
for col in df.columns:
    if col != '행정동':
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

# ============================================================
# 4. 파생변수 생성
# ============================================================
df['상권활성도'] = df['동물병원수'] + df['동물약국수'] + df['펫샵수']
df['교통편의도'] = df['버스정류장수'] + df['역개수']
df['Y']         = df['반려동물등록수'] / (df['펫파크수'] + 1)

X_vars = ['인구수', '유동인구', '상권활성도', '교통편의도', '공시지가', '놀이터개수']
Y_var  = 'Y'

print(f"데이터셋 완성: {len(df)}개 행정동, 결측치: {df.isnull().sum().sum()}")

# ============================================================
# 5. 상관분석
# ============================================================
print('\n' + '='*60)
print('【 1단계: X 변수 간 상관분석 】')
print('='*60)

corr = df[X_vars].corr().round(3)
print(corr.to_string())

print('\n⚠️  상관계수 0.8 이상 (다중공선성 의심):')
found = False
for i in range(len(X_vars)):
    for j in range(i+1, len(X_vars)):
        val = abs(corr.iloc[i, j])
        if val >= 0.8:
            print(f'  {X_vars[i]} ↔ {X_vars[j]}: {corr.iloc[i,j]}')
            found = True
if not found:
    print('  → 없음! 모든 변수 사용 가능 ✅')

# 히트맵 저장
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('변수 간 상관분석 히트맵')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '상관분석_히트맵.png'), dpi=150, bbox_inches='tight')
plt.close()  # ← plt.show() 대신 close()로 바꾸기
print('히트맵 저장 완료')

# ============================================================
# 6. VIF (분산팽창계수)
# ============================================================
print('\n' + '='*60)
print('【 2단계: VIF (다중공선성 정밀 검사) 】')
print('='*60)

def calc_vif(df, vars):
    vifs = []
    for v in vars:
        y_v    = df[v].values
        X_v    = df[[x for x in vars if x != v]].values
        X_v    = np.column_stack([np.ones(len(X_v)), X_v])
        beta   = np.linalg.lstsq(X_v, y_v, rcond=None)[0]
        y_hat  = X_v @ beta
        ss_res = np.sum((y_v - y_hat) ** 2)
        ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif    = 1 / (1 - r2) if r2 < 1 else float('inf')
        vifs.append(round(vif, 2))
    return vifs

vifs   = calc_vif(df, X_vars)
vif_df = pd.DataFrame({'변수': X_vars, 'VIF': vifs})
vif_df['판단'] = vif_df['VIF'].apply(
    lambda v: '✅ 정상' if v < 5 else ('⚠️ 주의' if v < 10 else '❌ 제거 필요')
)

print(f'{"변수":<12} {"VIF":>8}  판단')
print('-'*35)
for _, row in vif_df.iterrows():
    print(f'{row["변수"]:<12} {row["VIF"]:>8}  {row["판단"]}')

# ============================================================
# 7. 결과 저장
# ============================================================
corr.to_csv(os.path.join(OUTPUT_DIR, '상관분석_결과.csv'), encoding='utf-8-sig')
vif_df.to_csv(os.path.join(OUTPUT_DIR, 'VIF_결과.csv'), index=False, encoding='utf-8-sig')

print('\n' + '='*60)
print('✅ 저장 완료!')
print('  상관분석_결과.csv   → output/')
print('  VIF_결과.csv        → output/')
print('  상관분석_히트맵.png → output/')
print('='*60)