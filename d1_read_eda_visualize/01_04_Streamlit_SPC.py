import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.errors import StreamlitAPIException

"""
Streamlit 기반 X-bar / R 관리도(SPC) 앱

핵심 흐름:
1) CSV 로딩 (경로 입력 또는 파일 업로드)
2) 숫자형 변수 선택
3) 선택한 변수 데이터를 subgroup(n) 단위로 재구성
4) X-bar, R, UCL/LCL 계산
5) 관리도 시각화 및 이탈 점(Out of Control) 표시
"""


# n(부분군 크기)에 따른 표준 SPC 상수
# - A2: X-bar 관리도 한계 계산 상수
# - D3, D4: R 관리도 한계 계산 상수
SPC_CONSTANTS = {
    2: {"A2": 1.880, "D3": 0.000, "D4": 3.267},
    3: {"A2": 1.023, "D3": 0.000, "D4": 2.574},
    4: {"A2": 0.729, "D3": 0.000, "D4": 2.282},
    5: {"A2": 0.577, "D3": 0.000, "D4": 2.114},
    6: {"A2": 0.483, "D3": 0.000, "D4": 2.004},
    7: {"A2": 0.419, "D3": 0.076, "D4": 1.924},
    8: {"A2": 0.373, "D3": 0.136, "D4": 1.864},
    9: {"A2": 0.337, "D3": 0.184, "D4": 1.816},
    10: {"A2": 0.308, "D3": 0.223, "D4": 1.777},
    11: {"A2": 0.285, "D3": 0.256, "D4": 1.744},
    12: {"A2": 0.266, "D3": 0.283, "D4": 1.717},
    13: {"A2": 0.249, "D3": 0.307, "D4": 1.693},
    14: {"A2": 0.235, "D3": 0.328, "D4": 1.672},
    15: {"A2": 0.223, "D3": 0.347, "D4": 1.653},
    16: {"A2": 0.212, "D3": 0.363, "D4": 1.637},
    17: {"A2": 0.203, "D3": 0.378, "D4": 1.622},
    18: {"A2": 0.194, "D3": 0.391, "D4": 1.608},
    19: {"A2": 0.187, "D3": 0.403, "D4": 1.597},
    20: {"A2": 0.180, "D3": 0.415, "D4": 1.585},
    21: {"A2": 0.173, "D3": 0.425, "D4": 1.575},
    22: {"A2": 0.167, "D3": 0.434, "D4": 1.566},
    23: {"A2": 0.162, "D3": 0.443, "D4": 1.557},
    24: {"A2": 0.157, "D3": 0.451, "D4": 1.548},
    25: {"A2": 0.153, "D3": 0.459, "D4": 1.541},
}


def load_dataframe(file_path: str, uploaded_file):
    """
    데이터 로딩 함수.

    uploaded_file이 있으면 업로드 파일을 우선 사용하고,
    없으면 텍스트 입력 경로(file_path)에서 CSV를 읽는다.
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(file_path)


def build_subgroups(series: pd.Series, subgroup_size: int):
    """
    1개 변수(series)를 subgroup_size 단위 2차원 배열로 변환한다.

    - 숫자 변환 불가 값은 NaN으로 처리 후 제거
    - subgroup을 완성하지 못하는 마지막 잔여 데이터는 제외
    - SPC 계산 최소 조건(2개 subgroup 이상)을 만족하지 못하면 None 반환
    """
    # 문자열/혼합형 입력 방어: 숫자 이외 값은 NaN으로 강제
    clean = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    subgroup_count = len(clean) // subgroup_size
    if subgroup_count < 2:
        return None, clean

    # reshape를 위해 subgroup_size * subgroup_count 만큼만 사용
    used = subgroup_count * subgroup_size
    data = clean.iloc[:used].to_numpy().reshape(subgroup_count, subgroup_size)
    return data, clean


def control_limits(subgroup_data: np.ndarray, subgroup_size: int):
    """
    X-bar / R 관리도의 중심선(CL), 관리상한(UCL), 관리하한(LCL)을 계산한다.

    공식:
    - Xbarbar = subgroup 평균들의 평균
    - Rbar = subgroup 범위(max-min)들의 평균
    - X-chart: UCL/LCL = Xbarbar ± A2 * Rbar
    - R-chart: UCL/LCL = D4 * Rbar, D3 * Rbar
    """
    c = SPC_CONSTANTS[subgroup_size]
    # 각 subgroup의 평균(X-bar_i)과 범위(R_i)
    subgroup_means = subgroup_data.mean(axis=1)
    subgroup_ranges = subgroup_data.max(axis=1) - subgroup_data.min(axis=1)

    # 관리도의 중심선(CL)
    xbarbar = subgroup_means.mean()
    rbar = subgroup_ranges.mean()

    # X-bar 관리도 한계
    x_ucl = xbarbar + c["A2"] * rbar
    x_lcl = xbarbar - c["A2"] * rbar

    # R 관리도 한계
    r_ucl = c["D4"] * rbar
    r_lcl = c["D3"] * rbar

    return {
        "subgroup_means": subgroup_means,
        "subgroup_ranges": subgroup_ranges,
        "xbarbar": xbarbar,
        "rbar": rbar,
        "x_ucl": x_ucl,
        "x_lcl": x_lcl,
        "r_ucl": r_ucl,
        "r_lcl": r_lcl,
    }


def draw_chart(values, cl, ucl, lcl, title, ylabel):
    """
    단일 관리도(X-bar 또는 R)를 그리는 공용 함수.
    values는 subgroup 순서값이며, CL/UCL/LCL 기준선을 함께 표시한다.
    """
    x = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))

    # subgroup 개수가 많을수록 마커를 더 작게 하여 겹침을 줄인다.
    marker_size = 5 if len(values) <= 40 else 3.5

    ax.plot(
        x,
        values,
        marker="o",
        markersize=marker_size,
        markeredgewidth=0.0,
        linewidth=0.8,
        alpha=0.9,
        label="Value",
    )
    ax.axhline(cl, color="green", linestyle="-", linewidth=1.0, label="CL")
    ax.axhline(ucl, color="red", linestyle="--", linewidth=0.9, label="UCL")
    ax.axhline(lcl, color="red", linestyle="--", linewidth=0.9, label="LCL")

    # 관리한계를 벗어난 점을 별도로 강조 표시
    out_of_control = (values > ucl) | (values < lcl)
    if out_of_control.any():
        ax.scatter(
            x[out_of_control],
            values[out_of_control],
            color="red",
            s=24,
            zorder=3,
            label="Out of Control",
        )

    ax.set_title(title)
    ax.set_xlabel("Subgroup")
    ax.set_ylabel(ylabel)
    # 범례는 플롯 바깥으로 배치해 데이터와 겹치지 않게 한다.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=9)
    ax.grid(alpha=0.3)
    return fig


# 일부 실행 환경(재실행/멀티페이지)에서 set_page_config 중복 호출 예외가 발생할 수 있어 보호 처리.
try:
    st.set_page_config(page_title="X-bar / R Chart", layout="wide")
except StreamlitAPIException:
    pass
st.title("SPC: X-bar / R 관리도")
st.caption("df.csv를 읽어 변수별 X-bar / R 관리도를 생성합니다.")

with st.sidebar:
    # 사용자 입력 영역: 파일/경로, subgroup 크기
    st.header("입력 설정")
    csv_path = st.text_input("CSV 경로", value="df.csv")
    uploaded = st.file_uploader("또는 CSV 업로드", type=["csv"])
    subgroup_size = st.number_input("Subgroup 크기 (n)", min_value=2, max_value=25, value=5, step=1)

try:
    # CSV 로딩 실패 시 즉시 종료(경로/파일 형식 오류 대응)
    df = load_dataframe(csv_path, uploaded)
except Exception as e:
    st.error(f"CSV를 읽지 못했습니다: {e}")
    st.stop()

st.success(f"데이터 로드 완료: {df.shape[0]}행 x {df.shape[1]}열")
with st.expander("데이터 미리보기"):
    st.dataframe(df.head(), use_container_width=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("숫자형 컬럼이 없어 X-bar / R 관리도를 만들 수 없습니다.")
    st.stop()

default_cols = [c for c in ["x_1", "target"] if c in numeric_cols]
if not default_cols:
    default_cols = [numeric_cols[0]]

selected_cols = st.multiselect(
    "관리도를 그릴 변수 선택(숫자형)",
    options=numeric_cols,
    default=default_cols,
)

if not selected_cols:
    st.info("변수를 하나 이상 선택해 주세요.")
    st.stop()

# 선택한 각 변수별로 독립적인 관리도를 계산/표시
for col in selected_cols:
    st.markdown(f"---\n### 변수: `{col}`")
    subgroup_data, clean_series = build_subgroups(df[col], subgroup_size)

    if subgroup_data is None:
        st.warning(
            f"`{col}`는 유효 데이터가 부족합니다. "
            f"(유효 샘플: {len(clean_series)}, 필요 최소: {subgroup_size * 2})"
        )
        continue

    subgroup_count = subgroup_data.shape[0]
    dropped_count = len(clean_series) - subgroup_count * subgroup_size
    if dropped_count > 0:
        st.info(
            f"`{col}`: 마지막 {dropped_count}개 샘플은 subgroup에 맞지 않아 제외했습니다."
        )

    result = control_limits(subgroup_data, subgroup_size)

    # 한계값/중심선 수치 요약
    c1, c2 = st.columns(2)
    c1.metric("Subgroup 개수", subgroup_count)
    c1.metric("X-bar 중심선 (CL)", f"{result['xbarbar']:.4f}")
    c1.metric("R 중심선 (CL)", f"{result['rbar']:.4f}")
    c2.metric("X-bar UCL / LCL", f"{result['x_ucl']:.4f} / {result['x_lcl']:.4f}")
    c2.metric("R UCL / LCL", f"{result['r_ucl']:.4f} / {result['r_lcl']:.4f}")

    fig_x = draw_chart(
        values=result["subgroup_means"],
        cl=result["xbarbar"],
        ucl=result["x_ucl"],
        lcl=result["x_lcl"],
        title=f"X-bar Chart - {col}",
        ylabel="Subgroup Mean",
    )
    st.pyplot(fig_x, use_container_width=True)
    plt.close(fig_x)

    fig_r = draw_chart(
        values=result["subgroup_ranges"],
        cl=result["rbar"],
        ucl=result["r_ucl"],
        lcl=result["r_lcl"],
        title=f"R Chart - {col}",
        ylabel="Subgroup Range",
    )
    st.pyplot(fig_r, use_container_width=True)
    plt.close(fig_r)

