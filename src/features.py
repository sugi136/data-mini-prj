# 배터리 수명 예측용 피처 생성 모듈
# EDA에서 선정한 피처의 계산 기능 구현

import numpy as np
import pandas as pd

def build_delta_q_features(qd10_df, qd100_df):
    """사이클 10과 100의 Qdlin 차이로부터 로그 분산을 계산합니다.
    Qdlin은 1차 방전 용량을 의미하며, 사이클 10과 100의 Qdlin 차이를 계산하여
    표본 분산을 구하고, 로그10을 취한 값을 반환합니다.
    """

    paired = qd10_df[["cell_id", "Qdlin"]].merge(
        qd100_df[["cell_id", "Qdlin"]],
        on="cell_id",
        how="outer",
        suffixes=("_10", "_100"),
        validate="one_to_one",
        indicator=True,
    )

    # 두 데이터프레임의 셀 목록이 일치하는지 확인합니다.
    if not paired["_merge"].eq("both").all():
        raise ValueError("Cycle 10·100의 셀 목록이 일치하지 않습니다.")

    records = []

    # 각 셀별로 Qdlin 차이의 로그 분산을 계산합니다.
    for _, row in paired.iterrows():
        qd10 = np.asarray(row["Qdlin_10"], dtype=float).reshape(-1)
        qd100 = np.asarray(row["Qdlin_100"], dtype=float).reshape(-1)

        # Qdlin 배열의 길이가 동일하고, 최소 2개 이상의 데이터가 있는지 확인합니다.
        if qd10.shape != qd100.shape or qd10.size < 2:
            raise ValueError(
                f"셀 {row['cell_id']}: Qdlin 길이를 확인해 주세요."
            )

        # Qdlin 차이 계산 및 로그 분산 계산
        delta_q = qd100 - qd10
        delta_variance = float(np.var(delta_q, ddof=1))
        log_variance = np.log10(
            max(delta_variance, np.finfo(float).tiny)
        )

        records.append({
            "cell_id": row["cell_id"],
            "log_delta_q_variance": log_variance,
        })

    return pd.DataFrame(
        records,
        columns=["cell_id", "log_delta_q_variance"],
    )

def build_chargetime_features(summary_df):
    """사이클 10과 100의 충전 시간 평균을 계산합니다.
    summary_df는 extract_summary()로 생성된 DataFrame이어야 합니다.
    """

    early = summary_df[summary_df["cycle"] <= 100]

    return (
        early.groupby("cell_id", as_index=False)
        .agg(mean_chargetime=("chargetime", "mean"))
    )

def build_feature_table(summary_df, qd10_df, qd100_df):
    """EDA에서 선정한 피처를 계산하여 셀별 표로 정리합니다.
    summary_df는 extract_summary()로 생성된 DataFrame이어야 합니다.
    qd10_df와 qd100_df는 extract_cycle_qdlin()로 생성된 DataFrame이어야 합니다.
    """

    # 셀별 Target이 유일한지 확인합니다.
    target_counts = (
        summary_df.groupby("cell_id")["cycle_life"]
        .nunique(dropna=False)
    )
    if target_counts.gt(1).any():
        raise ValueError("동일한 셀에 서로 다른 cycle_life가 있습니다.")

    # Target 표를 생성합니다.
    targets = (
        summary_df[["cell_id", "cycle_life"]]
        .drop_duplicates(subset="cell_id")
    )

    delta_features = build_delta_q_features(qd10_df, qd100_df)
    charge_features = build_chargetime_features(summary_df)

    # Target 표의 셀을 기준으로 연결합니다.
    # 피처가 없는 셀도 남겨 두어 아래 검사에서 확인할 수 있도록 합니다.
    feature_table = (
        targets.merge(
            delta_features,
            on="cell_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            charge_features,
            on="cell_id",
            how="left",
            validate="one_to_one",
        )
    )

    # Target이 없는 셀은 제거합니다. (EDA에서 제외된 셀)
    feature_table = feature_table.dropna(subset=["cycle_life"]).copy()

    feature_columns = [
        "log_delta_q_variance",
        "mean_chargetime",
    ]

    # 피처에 결측값 또는 무한대가 있는 셀을 확인합니다.
    valid_rows = np.isfinite(
        feature_table[feature_columns].to_numpy(dtype=float)
    ).all(axis=1)

    if not valid_rows.all():
        invalid_cells = feature_table.loc[
            ~valid_rows, "cell_id"
        ].tolist()
        raise ValueError(
            f"피처에 결측값 또는 무한대가 있는 셀: {invalid_cells}"
        )

    return (
        feature_table[
            ["cell_id", *feature_columns, "cycle_life"]
        ]
        .sort_values("cell_id")
        .reset_index(drop=True)
    )