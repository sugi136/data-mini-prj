"""초기 100사이클 원본값을 셀 단위 모델 피처로 변환합니다."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import CellEarlyCycleData


EARLY_CYCLE_LIMIT = 100
FEATURE_COLUMNS = ("log_delta_q_variance", "mean_chargetime")
IDENTIFIER_COLUMNS = ("batch", "cell_id", "cell_key")
TARGET_COLUMN = "cycle_life"


def _calculate_log_delta_q_variance(cell: CellEarlyCycleData) -> float:
    """ΔQ100-10(V)의 표본분산을 log10 변환해 반환합니다."""
    common_length = min(len(cell.qd_cycle_10), len(cell.qd_cycle_100))
    if common_length < 2:
        return np.nan

    delta_q = (
        cell.qd_cycle_100[:common_length]
        - cell.qd_cycle_10[:common_length]
    )
    finite_delta_q = delta_q[np.isfinite(delta_q)]
    if len(finite_delta_q) < 2:
        return np.nan

    delta_q_variance = float(np.var(finite_delta_q, ddof=1))
    if not np.isfinite(delta_q_variance) or delta_q_variance <= 0:
        return np.nan

    return float(np.log10(delta_q_variance))


def _calculate_mean_charge_time(cell: CellEarlyCycleData) -> float:
    """초기 100사이클의 평균 충전 시간을 계산합니다."""
    common_length = min(len(cell.cycles), len(cell.charge_times))
    if common_length == 0:
        return np.nan

    cycles = cell.cycles[:common_length]
    charge_times = cell.charge_times[:common_length]
    valid = (
        np.isfinite(cycles)
        & np.isfinite(charge_times)
        & (cycles <= EARLY_CYCLE_LIMIT)
    )
    if not np.any(valid):
        return np.nan

    return float(np.mean(charge_times[valid]))


def build_feature_table(
    cells: Iterable[CellEarlyCycleData],
) -> pd.DataFrame:
    """셀 목록을 식별자·Target·두 입력 피처로 구성된 표로 변환합니다."""
    records = []

    for cell in cells:
        # Target이 없는 셀은 지도학습 데이터로 사용할 수 없습니다.
        if not np.isfinite(cell.cycle_life):
            continue

        records.append(
            {
                "batch": cell.batch,
                "cell_id": cell.cell_id,
                "cell_key": cell.cell_key,
                TARGET_COLUMN: cell.cycle_life,
                "log_delta_q_variance": _calculate_log_delta_q_variance(cell),
                "mean_chargetime": _calculate_mean_charge_time(cell),
            }
        )

    feature_table = pd.DataFrame.from_records(records)
    validate_feature_table(feature_table)
    return feature_table.sort_values(["batch", "cell_id"]).reset_index(drop=True)


def validate_feature_table(feature_table: pd.DataFrame) -> None:
    """모델 학습 전에 피처 테이블의 구조와 값 범위를 검사합니다."""
    required_columns = {
        *IDENTIFIER_COLUMNS,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }
    missing_columns = required_columns - set(feature_table.columns)
    if missing_columns:
        raise ValueError(f"필수 열이 없습니다: {sorted(missing_columns)}")

    if feature_table.empty:
        raise ValueError("생성된 피처 데이터가 없습니다.")

    duplicate_mask = feature_table.duplicated(["batch", "cell_key"])
    if duplicate_mask.any():
        duplicate_keys = feature_table.loc[duplicate_mask, "cell_key"].tolist()
        raise ValueError(f"중복 셀이 있습니다: {duplicate_keys}")

    numeric_columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
    infinite_mask = np.isinf(feature_table[numeric_columns].to_numpy(dtype=float))
    if infinite_mask.any():
        raise ValueError("Target 또는 피처에 무한대 값이 있습니다.")


def save_feature_table(feature_table: pd.DataFrame, output_path: Path) -> None:
    """피처 테이블을 CSV로 저장하며 필요한 상위 디렉터리를 생성합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(output_path, index=False)


def split_by_batch(
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """학습 Batch 1과 외부 평가 Batch 2·3을 분리합니다."""
    splits = []
    for batch_name in ("Batch 1", "Batch 2", "Batch 3"):
        batch_data = feature_table.loc[
            feature_table["batch"] == batch_name
        ].copy()
        if batch_data.empty:
            raise ValueError(f"{batch_name} 데이터가 없습니다.")
        splits.append(batch_data.reset_index(drop=True))

    return tuple(splits)
