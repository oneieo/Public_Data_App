import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv("output/05_최종_Z점수_순위.csv", encoding="utf-8-sig")

# 컬럼명 확인
print(df.columns)

# 👉 입지 점수 컬럼명 맞게 수정
score_col = 'Z점수'   # 필요하면 바꿔라

# 점수 기준 내림차순 정렬
df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

# 순위 생성
df['순위'] = df.index + 1

# 누적합
df['누적합'] = df[score_col].cumsum()

# 기울기 (증가량)
df['기울기'] = df['누적합'].diff()

# 첫 값 보정
df.loc[0, '기울기'] = df.loc[0, score_col]

# ---------------------------
# 🔥 최적 지점 찾기 (핵심)
# ---------------------------

# 기울기 변화율 (감소율)
df['기울기변화'] = df['기울기'].diff()

# 기준: 증가량이 거의 평평해지는 지점
threshold = df['기울기'].max() * 0.05  # 5% 기준

optimal_idx = df[df['기울기'] < threshold].index[0]
optimal_k = df.loc[optimal_idx, '순위']

print("최적 거점 개수:", optimal_k)

# ---------------------------
# 🔥 그래프
# ---------------------------

plt.figure(figsize=(10,6))

# 누적합 곡선
plt.plot(df['순위'], df['누적합'], marker='o')

# 최적 지점 표시
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'최적 개수: {optimal_k}')

# 음영 처리 (PPT 느낌)
plt.axvspan(optimal_k-1, optimal_k+1, color='gray', alpha=0.2)

plt.title('최적 거점 개수 (누적합 기반)')
plt.xlabel('거점 개수 (순위)')
plt.ylabel('누적 입지 점수')
plt.legend()

plt.grid()
plt.show()