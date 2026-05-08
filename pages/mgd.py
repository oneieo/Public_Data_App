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
plt.show()

print('\n그래프 저장: output/06_민감도_순위변동_그래프_top3.png')