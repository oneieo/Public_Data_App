import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(layout="wide", page_title="데이터 전처리 결과")
st.title("🗂️ 데이터 전처리 결과")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

AREA_COL = "면적(제곱미터)"

# ══════════════════════════════════════════
# 공통: 최종 정제 데이터 로드
# ══════════════════════════════════════════
@st.cache_data
def load_cleaned():
    return pd.read_csv(
        os.path.join(OUTPUT_DIR, 'playground_cleaned_final.csv'),
        encoding='utf-8-sig'
    )

@st.cache_data
def load_median():
    return pd.read_csv(
        os.path.join(OUTPUT_DIR, '시설유형별_면적_중앙값.csv'),
        encoding='utf-8-sig'
    )

@st.cache_data
def load_final_missing():
    return pd.read_csv(
        os.path.join(OUTPUT_DIR, '최종_컬럼별_결측치_검증.csv'),
        encoding='utf-8-sig'
    )

@st.cache_data
def load_outliers():
    path = os.path.join(OUTPUT_DIR, '04_면적_이상치_목록.csv')
    return pd.read_csv(path, encoding='utf-8-sig') if os.path.exists(path) else pd.DataFrame()

df        = load_cleaned()
median_df = load_median()
missing_df = load_final_missing()
outlier_df = load_outliers()

# ══════════════════════════════════════════
# 섹션 1. 전처리 단계별 행 수 변화
# ── 분석 코드 결과값을 직접 입력 (고정값)
# ══════════════════════════════════════════
st.subheader("📊 전처리 단계별 데이터 변화")

step_df = pd.DataFrame({
    "단계": ["원본", "시설유형 제거 후", "실내 제거 후", "최종"],
    "행 수": [1144, 977, 963, len(df)],
    "색상": ["#e2e8f0", "#cbd5e1", "#94a3b8", "#16a34a"]  # 앞 3개 회색, 최종 진한 초록
})

missing_step_df = pd.DataFrame({
    "단계": ["초기", "행 제거 후", "중앙값 대체 후"],
    "면적 결측치 수": [567, 469, 0],
    "색상": ["#e2e8f0", "#cbd5e1", "#16a34a"]  # 초기/행제거 회색, 완료 초록
})

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure(go.Bar(
        x=step_df["단계"],
        y=step_df["행 수"],
        text=step_df["행 수"],
        textposition="outside",
        marker_color=step_df["색상"].tolist(),
        marker_line=dict(color="white", width=1.5),
    ))
    fig.update_layout(
        title=dict(text="전처리 단계별 데이터 행 수 변화", x=0.5, xanchor="center"),
        template="plotly_white",
        showlegend=False,
        yaxis=dict(title="행 수", gridcolor="#f1f5f9"),
        xaxis=dict(title=""),
        margin=dict(t=50, b=20),
        height=360
    )
    st.plotly_chart(fig, use_container_width=True, key="bar_step")

with col2:
    fig2 = go.Figure(go.Bar(
        x=missing_step_df["단계"],
        y=missing_step_df["면적 결측치 수"],
        text=missing_step_df["면적 결측치 수"],
        textposition="outside",
        marker_color=missing_step_df["색상"].tolist(),
        marker_line=dict(color="white", width=1.5),
    ))
    fig2.update_layout(
        title=dict(text="면적 결측치 처리 전후 비교", x=0.5, xanchor="center"),
        template="plotly_white",
        showlegend=False,
        yaxis=dict(title="결측치 수", gridcolor="#f1f5f9"),
        xaxis=dict(title=""),
        margin=dict(t=50, b=20),
        height=360
    )
    st.plotly_chart(fig2, use_container_width=True, key="bar_missing")

st.markdown("---")

# ══════════════════════════════════════════
# 섹션 2. 시설유형별 면적 중앙값
# ══════════════════════════════════════════
st.subheader("📐 시설유형별 면적 중앙값")

col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(
        median_df.sort_values("시설유형별_면적_중앙값", ascending=False),
        use_container_width=True,
        hide_index=True
    )

with col2:
    median_sorted = median_df.sort_values("시설유형별_면적_중앙값").reset_index(drop=True)
    median_sorted["색순위"] = range(len(median_sorted))  # 0, 1, 2, 3...

    fig3 = px.bar(
        median_sorted,
        x="시설유형별_면적_중앙값",
        y="시설유형",
        orientation="h",
        text="시설유형별_면적_중앙값",
        color="색순위",                          
        color_continuous_scale="Greens",
        title="시설유형별 면적 중앙값 (㎡)"
    )
    fig3.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig3.update_layout(coloraxis_showscale=False, xaxis_title="면적 중앙값(㎡)", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════
# 섹션 3. 최종 결측치 검증
# ══════════════════════════════════════════
st.subheader("✅ 최종 컬럼별 결측치 검증")

col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

with col2:
    has_any_missing = missing_df["결측치 수"].sum() > 0

    # 결측치 있는 컬럼 강조색, 없는 컬럼 회색
    colors = [
        "#ff621e" if v > 0 else "#e2e8f0"
        for v in missing_df["결측치 수"]
    ]

    fig4 = go.Figure(go.Bar(
        x=missing_df["결측치 수"],
        y=missing_df["컬럼"],
        orientation="h",
        text=missing_df["결측치 수"].apply(lambda v: str("") if v > 0 else ""),
        textposition="outside",
        marker_color=colors,
        marker_line=dict(color="white", width=1),
    ))

    fig4.update_layout(
        title=dict(
            text="컬럼별 결측치 현황",
            x=0.5, xanchor="center",
            font=dict(size=16, color="#000000" if has_any_missing else "#16a34a")
        ),
        template="plotly_white",
        xaxis=dict(title="결측치 수", gridcolor="#f1f5f9"),
        yaxis=dict(title="", autorange="reversed"),  # 위에서부터 순서대로
        margin=dict(t=50, r=60, b=20),
        height=max(300, len(missing_df) * 32),
        showlegend=False
    )

    # 결측치 있는 컬럼에 어노테이션
    for _, row in missing_df[missing_df["결측치 수"] > 0].iterrows():
        fig4.add_annotation(
            x=row["결측치 수"],
            y=row["컬럼"],
            text=f"  ← {row['결측치 수']}개 결측",
            showarrow=False,
            xanchor="left",
            font=dict(color="#ff621e", size=15)
        )

    st.plotly_chart(fig4, use_container_width=True, key="bar_missing_check")

    if not has_any_missing:
        st.success("🎉 모든 컬럼의 결측치가 0입니다!")
    else:
        missing_cols = missing_df[missing_df["결측치 수"] > 0]["컬럼"].tolist()
        st.warning(f"⚠️ 결측치 있는 컬럼: {', '.join(missing_cols)}")

st.markdown("---")

# # ══════════════════════════════════════════
# # 섹션 4. 면적 이상치 분석 (IQR)
# # ══════════════════════════════════════════

st.subheader("📦 면적 이상치 분석 (IQR)")

@st.cache_data
def load_pgd():
    return pd.read_csv(
        os.path.join(OUTPUT_DIR, '..', 'data', 'pgd.csv'),
        encoding='utf-8-sig'
    )

pgd_df = load_pgd()
pgd_df[AREA_COL] = pd.to_numeric(pgd_df[AREA_COL], errors='coerce')
area = pgd_df[AREA_COL].dropna()

Q1    = area.quantile(0.25)
Q3    = area.quantile(0.75)
IQR   = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

col1, col2, col3, col4 = st.columns(4)
col1.metric("Q1",   f"{Q1:,.1f} ㎡")
col2.metric("Q3",   f"{Q3:,.1f} ㎡")
col3.metric("하한값", f"{lower:,.1f} ㎡")
col4.metric("상한값", f"{upper:,.1f} ㎡")

col_a, col_b = st.columns(2)

with col_a:
    # 구간별 색상 분리
    counts, bins = pd.cut(area, bins=40, retbins=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_counts = counts.value_counts(sort=False).values
    bar_colors = [
        "#F97316" if (b < lower or b > upper) else "#16A34A"
        for b in bin_centers
    ]

    fig5 = go.Figure(go.Bar(
        x=bin_centers,
        y=bin_counts,
        marker_color=bar_colors,
        name="면적"
    ))
    fig5.add_vline(x=lower, line_dash="dash", line_color="orange", annotation_text="하한값")
    fig5.add_vline(x=upper, line_dash="dash", line_color="green", annotation_text="상한값")
    fig5.update_layout(
        title="면적 분포 히스토그램",
        xaxis_title="면적(㎡)",
        yaxis_title="count",
        bargap=0.05,
        showlegend=False,
        template="plotly_white"
    )
    st.plotly_chart(fig5, use_container_width=True, key="hist_area_iqr")

with col_b:
    # 이상치 따로 분리
    outliers_pts = area[(area < lower) | (area > upper)]
    normal_pts   = area[(area >= lower) & (area <= upper)]

    fig6 = go.Figure()

    # 박스플롯 (이상치 점 숨김)
    fig6.add_trace(go.Box(
        x=area,
        name="면적",
        marker_color="#16A34A",
        boxmean=True,
        boxpoints=False  # 기본 이상치 점 숨기기
    ))

    # 이상치 점 별도로 주황색으로 추가
    fig6.add_trace(go.Scatter(
        x=outliers_pts,
        y=["면적"] * len(outliers_pts),
        mode="markers",
        marker=dict(color="#F97316", size=8),
        name="이상치"
    ))

    fig6.update_layout(
        title="면적 박스플롯",
        xaxis_title="면적(㎡)",
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig6, use_container_width=True, key="box_area_iqr")

# 산점도 (이상치 색상 구분)
df_scatter = pgd_df[[AREA_COL]].dropna().reset_index()
df_scatter["이상치"] = df_scatter[AREA_COL].apply(
    lambda v: "이상치" if v < lower or v > upper else "정상"
)

outlier_count = (df_scatter["이상치"] == "이상치").sum()
st.caption(f"IQR 기준 이상치: {outlier_count}개 / 전체 {len(df_scatter)}개")

fig7 = px.scatter(
    df_scatter,
    x="index", y=AREA_COL,
    color="이상치",
    color_discrete_map={"정상": "#16A34A", "이상치": "#ff621e"},
    opacity=0.6,
    title="면적 산점도 (이상치 구분)",
    labels={"index": "Index", AREA_COL: "면적(㎡)"},
    template="plotly_white"
)
fig7.add_hline(y=lower, line_dash="dash", line_color="orange",  annotation_text="하한값")
fig7.add_hline(y=upper, line_dash="dash", line_color="green", annotation_text="상한값")
st.plotly_chart(fig7, use_container_width=True, key="scatter_area_iqr")

st.markdown("---")

# ══════════════════════════════════════════
# 섹션 5. 이상치 목록
# ══════════════════════════════════════════
st.subheader("🚨 이상치 목록")

if not outlier_df.empty:
    st.metric("이상치 개수", f"{len(outlier_df)}개")
    st.dataframe(
        outlier_df.sort_values(AREA_COL, ascending=False),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("이상치 목록 파일이 없습니다.")