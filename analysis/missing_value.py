import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# 0. 기본 설정
# ============================================================

os.makedirs("output", exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 데이터 로드
# ============================================================

def read_csv_safe(path):

    for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:

        try:
            df = pd.read_csv(path, encoding=enc)
            print(f'로드 성공: {enc}')
            return df

        except Exception:
            continue

    raise ValueError(f'모든 인코딩 실패: {path}')


df_raw = read_csv_safe("data/playground.csv")

area_col = "면적(제곱미터)"

print("컬럼 목록")
print(df_raw.columns)

# 숫자형 변환
df_raw[area_col] = pd.to_numeric(df_raw[area_col], errors="coerce")

# ============================================================
# 2. 최초 결측치 현황
# ============================================================

initial_total = len(df_raw)
initial_missing = df_raw[area_col].isnull().sum()
initial_missing_ratio = initial_missing / initial_total * 100

print("\n[초기 결측치 현황]")
print(f"전체 행 수: {initial_total}")
print(f"면적 결측치 수: {initial_missing}")
print(f"면적 결측치 비율: {initial_missing_ratio:.2f}%")

# ============================================================
# 3. 시설유형 제거
# ============================================================

remove_types = [
    "놀이제공영업소",
    "식품접객업소",
    "아동복지시설",
    "어린이집",
    "육아종합지원센터",
    "의료기관",
    "종교시설"
]

before_type_remove = len(df_raw)

df_clean = df_raw.copy()
df_clean["시설유형"] = df_clean["시설유형"].astype(str).str.strip()

df_clean = df_clean[~df_clean["시설유형"].isin(remove_types)].copy()

after_type_remove = len(df_clean)
type_removed_count = before_type_remove - after_type_remove

print("\n[시설유형 제거]")
print(f"제거 전 행 수: {before_type_remove}")
print(f"제거 후 행 수: {after_type_remove}")
print(f"제거된 행 수: {type_removed_count}")

# ============================================================
# 4. 실내 시설 제거
# ============================================================

before_indoor_remove = len(df_clean)

df_clean = df_clean[df_clean["실내_실외"] != "실내"].copy()
df_clean = df_clean.rename(columns={"실내_실외": "실외"})

after_indoor_remove = len(df_clean)
indoor_removed_count = before_indoor_remove - after_indoor_remove

print("\n[실내 시설 제거]")
print(f"제거 전 행 수: {before_indoor_remove}")
print(f"제거 후 행 수: {after_indoor_remove}")
print(f"제거된 행 수: {indoor_removed_count}")

# ============================================================
# 5. 전처리 후 결측치 현황
# ============================================================

before_fill_missing = df_clean[area_col].isnull().sum()

print("\n[중앙값 대체 전 결측치]")
print(f"면적 결측치 수: {before_fill_missing}")

# ============================================================
# 6. 시설유형별 중앙값 계산
# ============================================================

median_by_type = (
    df_clean
    .groupby("시설유형")[area_col]
    .median()
    .reset_index()
    .rename(columns={area_col: "시설유형별_면적_중앙값"})
)

print("\n[시설유형별 면적 중앙값]")
print(median_by_type.to_string(index=False))

median_by_type.to_csv(
    "output/시설유형별_면적_중앙값.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 7. 시설유형별 중앙값으로 결측치 대체
# ============================================================

df_clean[area_col] = df_clean.groupby("시설유형")[area_col].transform(
    lambda x: x.fillna(x.median())
)

# 혹시 시설유형 전체가 결측이라 중앙값 계산이 안 된 경우 전체 중앙값으로 한 번 더 대체
df_clean[area_col] = df_clean[area_col].fillna(df_clean[area_col].median())

after_fill_missing = df_clean[area_col].isnull().sum()

print("\n[중앙값 대체 후 결측치]")
print(f"면적 결측치 수: {after_fill_missing}")

# ============================================================
# 8. 최종 저장
# ============================================================

df_clean.to_csv(
    "output/playground_cleaned_final.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료: output/playground_cleaned_final.csv")

# ============================================================
# 9. 시각화 1: 단계별 행 수 변화
# ============================================================

step_df = pd.DataFrame({
    "단계": ["원본", "시설유형 제거 후", "실내 제거 후", "최종"],
    "행 수": [
        initial_total,
        after_type_remove,
        after_indoor_remove,
        len(df_clean)
    ]
})

plt.figure(figsize=(8, 5))
sns.barplot(data=step_df, x="단계", y="행 수")

plt.title("전처리 단계별 데이터 행 수 변화", fontsize=14, fontweight="bold")
plt.xlabel("")
plt.ylabel("행 수")

ax = plt.gca()
for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=3)

plt.tight_layout()
plt.savefig(
    "output/01_전처리_단계별_행수변화.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# 10. 시각화 2: 면적 결측치 변화
# ============================================================

missing_df = pd.DataFrame({
    "단계": ["초기", "행 제거 후", "중앙값 대체 후"],
    "면적 결측치 수": [
        initial_missing,
        before_fill_missing,
        after_fill_missing
    ]
})

plt.figure(figsize=(7, 5))
sns.barplot(data=missing_df, x="단계", y="면적 결측치 수")

plt.title("면적 결측치 처리 전후 비교", fontsize=14, fontweight="bold")
plt.xlabel("")
plt.ylabel("결측치 수")

ax = plt.gca()
for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=3)

plt.tight_layout()
plt.savefig(
    "output/02_면적_결측치_처리전후.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# 11. 시각화 3: 시설유형별 면적 중앙값
# ============================================================

median_plot_df = median_by_type.sort_values(
    "시설유형별_면적_중앙값",
    ascending=False
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=median_plot_df,
    x="시설유형별_면적_중앙값",
    y="시설유형"
)

plt.title("시설유형별 면적 중앙값", fontsize=14, fontweight="bold")
plt.xlabel("면적 중앙값(㎡)")
plt.ylabel("시설유형")

plt.tight_layout()
plt.savefig(
    "output/03_시설유형별_면적_중앙값.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# 12. 시각화 4: 최종 결측치 검증
# ============================================================

final_missing_df = (
    df_clean
    .isnull()
    .sum()
    .reset_index()
)

final_missing_df.columns = ["컬럼", "결측치 수"]

final_missing_df.to_csv(
    "output/최종_컬럼별_결측치_검증.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=final_missing_df,
    x="결측치 수",
    y="컬럼"
)

plt.title("최종 데이터 컬럼별 결측치 검증", fontsize=14, fontweight="bold")
plt.xlabel("결측치 수")
plt.ylabel("컬럼")

plt.tight_layout()
plt.savefig(
    "output/04_최종_컬럼별_결측치_검증.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("\n✅ 전체 전처리 및 시각화 완료")
print("저장 위치: output 폴더")