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

# 분석 스크립트 실행 결과값 그대로 하드코딩
# (분석 코드를 바꾸면 여기도 업데이트)
step_df = pd.DataFrame({
    "단계": ["원본", "시설유형 제거 후", "실내 제거 후", "최종"],
    "행 수": [1144, 977, 963, len(df)]  # ← 실제 숫자로 교체
    # 예) [3241, 2800, 2400, len(df)]
})

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        step_df, x="단계", y="행 수",
        text="행 수",
        color="단계",
        color_discrete_sequence=px.colors.sequential.Blues_r,
        title="전처리 단계별 데이터 행 수 변화"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="행 수", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # 결측치 변화도 같은 방식
    # 분석 코드 실행 결과 숫자를 아래에 입력
    missing_step_df = pd.DataFrame({
        "단계": ["초기", "행 제거 후", "중앙값 대체 후"],
        "면적 결측치 수": [567, 469, 0]  # ← 실제 숫자로 교체
        # 예) [312, 120, 0]
    })
    fig2 = px.bar(
        missing_step_df, x="단계", y="면적 결측치 수",
        text="면적 결측치 수",
        color="단계",
        color_discrete_sequence=["#ef4444", "#f97316", "#22c55e"],
        title="면적 결측치 처리 전후 비교"
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(showlegend=False, xaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)

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
    fig3 = px.bar(
        median_df.sort_values("시설유형별_면적_중앙값"),
        x="시설유형별_면적_중앙값",
        y="시설유형",
        orientation="h",
        text="시설유형별_면적_중앙값",
        color="시설유형별_면적_중앙값",
        color_continuous_scale="Blues",
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
    # 결측치 0인 컬럼은 숨기고 있는 것만 표시
    has_missing = missing_df[missing_df["결측치 수"] > 0]
    if has_missing.empty:
        st.success("🎉 모든 컬럼의 결측치가 0입니다!")
    else:
        fig4 = px.bar(
            has_missing.sort_values("결측치 수"),
            x="결측치 수", y="컬럼",
            orientation="h",
            text="결측치 수",
            color="결측치 수",
            color_continuous_scale="Reds",
            title="결측치가 남아있는 컬럼"
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

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
    fig5 = px.histogram(
        area, nbins=40,
        title="면적 분포 히스토그램",
        labels={"value": "면적(㎡)", "count": "개수"},
        color_discrete_sequence=["#3b82f6"],
        template="plotly_white"
    )
    fig5.update_layout(bargap=0.05, showlegend=False)
    fig5.add_vline(x=lower, line_dash="dash", line_color="red",  annotation_text="하한값")
    fig5.add_vline(x=upper, line_dash="dash", line_color="blue", annotation_text="상한값")
    st.plotly_chart(fig5, use_container_width=True, key="hist_area_iqr")

with col_b:
    fig6 = go.Figure()
    fig6.add_trace(go.Box(
        x=area,
        name="면적",
        marker_color="#3b82f6",
        boxmean=True
    ))
    fig6.update_layout(
        title="면적 박스플롯",
        xaxis_title="면적(㎡)",
        template="plotly_white"
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
    color_discrete_map={"정상": "#3b82f6", "이상치": "#ef4444"},
    opacity=0.6,
    title="면적 산점도 (이상치 구분)",
    labels={"index": "Index", AREA_COL: "면적(㎡)"},
    template="plotly_white"
)
fig7.add_hline(y=lower, line_dash="dash", line_color="red",  annotation_text="하한값")
fig7.add_hline(y=upper, line_dash="dash", line_color="blue", annotation_text="상한값")
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