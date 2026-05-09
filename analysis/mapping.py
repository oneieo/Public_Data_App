import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')


# ============================================================
# 한글 폰트 설정
# ============================================================

plt.rcParams['axes.unicode_minus'] = False

try:
    plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows용
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'


# ============================================================
# 0. 데이터 로드
# ============================================================

PATH = './data/'

print("현재 작업 폴더:", os.getcwd())
print("현재 폴더 파일 목록:")
print(os.listdir(PATH))


def read_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")

    for enc in ['utf-8-sig', 'cp949', 'euc-kr']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"✅ 로드 성공: {file_path} / encoding={enc}")
            return df
        except Exception as e:
            print(f"❌ 로드 실패: {file_path} / encoding={enc}")

    raise ValueError(f"모든 인코딩으로 읽기 실패: {file_path}")


# 파일 불러오기
df_hospital = read_csv(PATH + 'animal_hospital.csv')
df_pharmacy = read_csv(PATH + 'animal_permarcy.csv')
df_pet_reg = read_csv(PATH + 'pet_registration.csv')
df_playground = read_csv(PATH + 'playground.csv')
df_petpark = read_csv(PATH + 'pet_park.csv')
df_pop = read_csv(PATH + 'population.csv')


# 확인 출력
print("\n✅ 데이터 로드 완료")
print(f"동물병원: {len(df_hospital)}개")
print(f"동물약국: {len(df_pharmacy)}개")
print(f"반려동물 등록: {len(df_pet_reg)}개 동")
print(f"어린이 놀이시설: {len(df_playground)}개")
print(f"반려견 놀이터: {len(df_petpark)}개")
print(f"인구분포: {len(df_pop)}개 동")

print("인구분포 컬럼:")
print(df_pop.columns)

print("\n반려동물 등록현황 컬럼:")
print(df_pet_reg.columns)

pop_dongs = set(df_pop['행정동'].dropna().unique())
pet_dongs = set(df_pet_reg['읍면동(법정동)'].dropna().unique())

missing_in_pet_reg = sorted(pop_dongs - pet_dongs)
extra_in_pet_reg = sorted(pet_dongs - pop_dongs)

print("인구분포에는 있는데 반려동물 등록현황에는 없는 동:")
print(missing_in_pet_reg)
print("개수:", len(missing_in_pet_reg))

print("\n반려동물 등록현황에는 있는데 인구분포에는 없는 동:")
print(extra_in_pet_reg)
print("개수:", len(extra_in_pet_reg))

#매핑 후

import pandas as pd

# ============================================================
# 1. 파일 불러오기
# ============================================================

df_pop = pd.read_csv("data/population.csv", encoding="utf-8-sig")
df_pet = pd.read_csv("data/pet_registration.csv", encoding="utf-8-sig")


# ============================================================
# 2. 컬럼명 확인
# (실제 컬럼명 보고 필요하면 수정)
# ============================================================

print("population.csv 컬럼명")
print(df_pop.columns)

print("\npet_registration.csv 컬럼명")
print(df_pet.columns)


# ============================================================
# 3. 사용할 컬럼명 지정
# 반드시 실제 컬럼명과 맞춰줘
# ============================================================
pop_dong_col = "행정동"
pop_count_col = "인구수"

pet_dong_col = "읍면동(법정동)"
pet_count_col = "합계"

# ============================================================
# 4. 공백 제거
# ============================================================

df_pop[pop_dong_col] = (
    df_pop[pop_dong_col]
    .astype(str)
    .str.strip()
)

df_pet[pet_dong_col] = (
    df_pet[pet_dong_col]
    .astype(str)
    .str.strip()
)


# ============================================================
# 5. 행정동 비교
# 서로 없는 동 찾기
# ============================================================

pop_dongs = set(df_pop[pop_dong_col].dropna().unique())
pet_dongs = set(df_pet[pet_dong_col].dropna().unique())

only_in_pop = sorted(pop_dongs - pet_dongs)
only_in_pet = sorted(pet_dongs - pop_dongs)

print("\n==============================")
print("population.csv에만 있는 행정동")

for i, dong in enumerate(only_in_pop, 1):
    print(f"{i}. {dong}")

print("개수:", len(only_in_pop))


print("\n==============================")
print("pet_registration.csv에만 있는 행정동")

for i, dong in enumerate(only_in_pet, 1):
    print(f"{i}. {dong}")

print("개수:", len(only_in_pet))

# ============================================================
# 행정동-법정동 매핑 전/후 불일치 개수 시각화
# ============================================================

import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs("output", exist_ok=True)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------
# 1. 매핑 전 불일치 개수
# ------------------------------------------------------------

before_pop_only = len(missing_in_pet_reg)   # 인구분포에는 있는데 등록현황에는 없는 동
before_pet_only = len(extra_in_pet_reg)     # 등록현황에는 있는데 인구분포에는 없는 동

# ------------------------------------------------------------
# 2. 매핑 후 불일치 개수
# 위 코드에서 only_in_pop, only_in_pet이 매핑 후 결과라고 가정
# ------------------------------------------------------------

after_pop_only = len(only_in_pop)
after_pet_only = len(only_in_pet)

# ------------------------------------------------------------
# 3. 시각화용 데이터프레임
# ------------------------------------------------------------

compare_df = pd.DataFrame({
    '구분': [
        '인구분포에만 존재',
        '반려동물 등록현황에만 존재',
        '인구분포에만 존재',
        '반려동물 등록현황에만 존재'
    ],
    '단계': [
        '매핑 전',
        '매핑 전',
        '매핑 후',
        '매핑 후'
    ],
    '불일치 개수': [
        before_pop_only,
        before_pet_only,
        after_pop_only,
        after_pet_only
    ]
})

print(compare_df)

# ------------------------------------------------------------
# 4. 막대그래프
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.barplot(
    data=compare_df,
    x='구분',
    y='불일치 개수',
    hue='단계'
)

plt.title('행정동-법정동 매핑 전후 불일치 개수 비교', fontsize=15, fontweight='bold')
plt.xlabel('')
plt.ylabel('불일치 행정동 개수')
plt.xticks(rotation=0)

# 막대 위 숫자 표시
ax = plt.gca()

for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=3)

plt.tight_layout()

plt.savefig(
    'output/행정동_법정동_매핑_전후_불일치_비교.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print('\n그래프 저장 완료: output/행정동_법정동_매핑_전후_불일치_비교.png')