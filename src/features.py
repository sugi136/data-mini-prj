# 배터리 수명 예측용 피처 생성 모듈
# EDA에서 선정한 피처의 계산 기능 구현

import numpy as np
import pandas as pd

# 기본 설정
CELL_ID = "cell_id"      # 셀 식별자 (Key)
TARGET = "cycle_life"    # 예측 대상 (Target)
FEATURE_COLUMNS = ["log_delta_q_variance", "mean_chargetime"]  # 최종 피처 (순서 = 출력 순서)

def compute_log_delta_q_variance(qd10, qd100) -> float:
    """사이클 10과 100의 Qdlin 차이로부터 로그 분산을 계산합니다.
    Qdlin은 전압 축에 대해 선형 보간된 방전 용량 곡선입니다.
    EDA와 동일하게 표본 분산(ddof=1)을 사용하고,
    분산이 0이면 아주 작은 양수로 치환한 뒤 로그10을 적용합니다.
    """
    qd10 = np.asarray(qd10, dtype=float).reshape(-1)
    qd100 = np.asarray(qd100, dtype=float).reshape(-1)

    # Qdlin 배열의 길이가 동일하고, 최소 2개 이상의 데이터가 있는지 확인합니다.
    if qd10.shape != qd100.shape or qd10.size < 2:
        raise ValueError(
            f"Qdlin 길이를 확인해 주세요. (10: {qd10.size}, 100: {qd100.size})"
        )

    # Qdlin 차이 계산 및 로그 분산 계산
    delta_q = qd100 - qd10
    delta_variance = float(np.var(delta_q, ddof=1))
    return float(np.log10(max(delta_variance, np.finfo(float).tiny)))


def _pair_qdlin(qd10_df, qd100_df):
    """사이클 10과 100의 곡선을 셀 식별자 기준으로 연결합니다."""
    paired = qd10_df[[CELL_ID, "Qdlin"]].merge(
        qd100_df[[CELL_ID, "Qdlin"]],
        on=CELL_ID,
        how="outer",
        suffixes=("_10", "_100"),
        validate="one_to_one",
        indicator=True,
    )

    # 두 데이터프레임의 셀 목록이 일치하는지 확인합니다.
    if not paired["_merge"].eq("both").all():
        raise ValueError("Cycle 10·100의 셀 목록이 일치하지 않습니다.")

    return paired[[CELL_ID, "Qdlin_10", "Qdlin_100"]]


def build_delta_q_features(qd10_df, qd100_df):
    """곡선 연결과 로그 분산 계산 함수를 호출해 셀별 피처 표를 만듭니다."""
    paired = _pair_qdlin(qd10_df, qd100_df)
    records = []

    # 각 셀별로 Qdlin 차이의 로그 분산을 계산합니다.
    for _, row in paired.iterrows():
        try:
            log_variance = compute_log_delta_q_variance(
                row["Qdlin_10"], row["Qdlin_100"]
            )
        except ValueError as error:
            # 계산에 문제가 발생한 셀을 오류 메시지에 포함합니다.
            raise ValueError(f"셀 {row[CELL_ID]}: {error}") from error

        records.append({
            CELL_ID: row[CELL_ID],
            "log_delta_q_variance": log_variance,
        })

    return pd.DataFrame(
        records,
        columns=[CELL_ID, "log_delta_q_variance"],
    )

def build_chargetime_features(summary_df):
    """초기 100사이클의 충전 시간 평균을 계산합니다.
    summary_df는 extract_summary()로 생성된 DataFrame이어야 합니다.
    """

    # EDA와 동일하게 사이클 1의 충전 시간 0도 평균에 포함합니다.
    early = summary_df[summary_df["cycle"] <= 100]

    return (
        early.groupby(CELL_ID, as_index=False)
        .agg(mean_chargetime=("chargetime", "mean"))
    )

def _build_targets(summary_df):
    """셀별 Target의 일관성을 검사하고 셀당 한 행으로 정리합니다."""
    # 셀별 Target이 유일한지 확인합니다.
    target_counts = (
        summary_df.groupby(CELL_ID)[TARGET]
        .nunique(dropna=False)
    )
    if target_counts.gt(1).any():
        raise ValueError("동일한 셀에 서로 다른 cycle_life가 있습니다.")

    # Target 표를 생성합니다.
    return (
        summary_df[[CELL_ID, TARGET]]
        .drop_duplicates(subset=CELL_ID)
    )


def _validate_finite(feature_table):
    """피처의 결측값과 무한대를 검사하고 해당 셀을 보고합니다."""
    # 피처에 결측값 또는 무한대가 있는 셀을 확인합니다.
    valid_rows = np.isfinite(
        feature_table[FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all(axis=1)

    if not valid_rows.all():
        invalid_cells = feature_table.loc[
            ~valid_rows, CELL_ID
        ].tolist()
        raise ValueError(
            f"피처에 결측값 또는 무한대가 있는 셀: {invalid_cells}"
        )


def build_feature_table(summary_df, qd10_df, qd100_df):
    """EDA에서 선정한 피처를 계산하여 셀별 표로 정리합니다.
    summary_df는 extract_summary()로 생성된 DataFrame이어야 합니다.
    qd10_df와 qd100_df는 extract_cycle_qdlin()로 생성된 DataFrame이어야 합니다.
    """
    targets = _build_targets(summary_df)
    delta_features = build_delta_q_features(qd10_df, qd100_df)
    charge_features = build_chargetime_features(summary_df)

    # Target 표의 셀을 기준으로 연결합니다.
    # 피처가 없는 셀도 남겨 두어 아래 검사에서 확인할 수 있도록 합니다.
    feature_table = (
        targets.merge(
            delta_features,
            on=CELL_ID,
            how="left",
            validate="one_to_one",
        )
        .merge(
            charge_features,
            on=CELL_ID,
            how="left",
            validate="one_to_one",
        )
    )

    # Target이 없는 셀은 제거합니다. (EDA에서 제외된 셀)
    feature_table = feature_table.dropna(subset=[TARGET]).copy()

    _validate_finite(feature_table)

    return (
        feature_table[
            [CELL_ID, *FEATURE_COLUMNS, TARGET]
        ]
        .sort_values(CELL_ID)
        .reset_index(drop=True)
    )
