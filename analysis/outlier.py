import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# 기본 설정
# ============================================================

BASE = os.getcwd()

OUTPUT_DIR = os.path.join(BASE, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 한글 깨짐 방지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# CSV 파일 불러오기
# ============================================================

df = pd.read_csv("data/pgd.csv", encoding="utf-8-sig")

# 컬럼 확인
print("컬럼 목록")
print(df.columns)

# ============================================================
# 면적 컬럼 설정
# ============================================================

area_col = "면적(제곱미터)"

# 숫자형 변환
df[area_col] = pd.to_numeric(
    df[area_col],
    errors="coerce"
)

# 결측 제거
area = df[area_col].dropna()

# 데이터 개수 확인
print("\n면적 유효 데이터 개수:", len(area))

# ============================================================
# 1. 히스토그램
# ============================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    area,
    bins=30,
    kde=True
)

plt.title("면적 분포 히스토그램")
plt.xlabel("면적(㎡)")
plt.ylabel("개수")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "01_면적_히스토그램.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# 2. 박스플롯
# ============================================================

plt.figure(figsize=(10, 5))

sns.boxplot(x=area)

plt.title("면적 박스플롯")
plt.xlabel("면적(㎡)")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "02_면적_박스플롯.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# 3. 산점도
# ============================================================

plt.figure(figsize=(10, 5))

plt.scatter(
    range(len(area)),
    area,
    alpha=0.6
)

plt.title("면적 산점도")
plt.xlabel("Index")
plt.ylabel("면적(㎡)")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "03_면적_산점도.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# 4. IQR 방식 이상치 계산
# ============================================================

Q1 = df[area_col].quantile(0.25)
Q3 = df[area_col].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\n==============================")
print("IQR 이상치 기준")
print("==============================")

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

print("하한값:", lower_bound)
print("상한값:", upper_bound)

# ============================================================
# 5. 이상치 추출
# ============================================================

outliers = df[
    (df[area_col] < lower_bound) |
    (df[area_col] > upper_bound)
]

print("\n이상치 개수:", len(outliers))

print("\n이상치 목록")

print(
    outliers[
        [
            "행정동",
            "시설명",
            "소재지주소",
            area_col
        ]
    ]
    .sort_values(
        by=area_col,
        ascending=False
    )
)

# ============================================================
# 6. 이상치 CSV 저장
# ============================================================

outliers.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "04_면적_이상치_목록.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\nCSV 저장 완료")
print("output/04_면적_이상치_목록.csv")

# ============================================================
# 7. 이상치 포함 박스플롯 시각화
# ============================================================

plt.figure(figsize=(10, 5))

sns.boxplot(x=df[area_col])

plt.axvline(
    lower_bound,
    color='red',
    linestyle='--',
    label='하한값'
)

plt.axvline(
    upper_bound,
    color='blue',
    linestyle='--',
    label='상한값'
)

plt.title("IQR 기준 면적 이상치 시각화")
plt.xlabel("면적(㎡)")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_IQR_이상치_시각화.png"
    ),
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ============================================================
# 완료 메시지
# ============================================================

print("\n==============================")
print("✅ 분석 완료")
print("==============================")

print("저장된 파일 목록")
print("1. 01_면적_히스토그램.png")
print("2. 02_면적_박스플롯.png")
print("3. 03_면적_산점도.png")
print("4. 04_면적_이상치_목록.csv")
print("5. 05_IQR_이상치_시각화.png")
print("\n저장 위치: output 폴더")