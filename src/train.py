"""Ridge 수명 예측 모델의 교차검증, 학습, 평가 기능을 제공합니다."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, TARGET_COLUMN


RANDOM_STATE = 42
DEFAULT_ALPHA_GRID = np.logspace(-4, 4, 17)


def build_ridge_pipeline() -> Pipeline:
    """결측값 처리·표준화·Ridge를 하나로 묶어 누수를 방지합니다."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]
    )


def _build_cv(n_splits: int, random_state: int) -> KFold:
    """모든 모델 선택 단계에서 재현 가능한 셀 단위 K-Fold를 생성합니다."""
    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def _split_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """피처 표에서 모델 입력 X와 Target y를 분리합니다."""
    return data.loc[:, FEATURE_COLUMNS], data.loc[:, TARGET_COLUMN]


def build_alpha_search(
    *,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> GridSearchCV:
    """MAE를 기준으로 Ridge alpha를 선택하는 GridSearchCV를 생성합니다."""
    return GridSearchCV(
        estimator=build_ridge_pipeline(),
        param_grid={"model__alpha": list(alpha_grid)},
        scoring="neg_mean_absolute_error",
        cv=_build_cv(n_splits, random_state),
        n_jobs=-1,
        refit=True,
    )


def generate_nested_cv_predictions(
    train_data: pd.DataFrame,
    *,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    outer_splits: int = 5,
    inner_splits: int = 4,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Batch 1에서 alpha 선택까지 포함한 Nested CV 예측값을 생성합니다.

    각 outer fold의 검증 셀은 해당 fold의 alpha 선택과 전처리 fit에 사용되지
    않으므로, 단일 CV로 튜닝과 성능 측정을 함께 하는 낙관 편향을 줄입니다.
    """
    x_train, y_train = _split_xy(train_data)
    inner_search = build_alpha_search(
        alpha_grid=alpha_grid,
        n_splits=inner_splits,
        random_state=random_state + 1,
    )
    outer_cv = _build_cv(outer_splits, random_state)
    return cross_val_predict(
        inner_search,
        x_train,
        y_train,
        cv=outer_cv,
        n_jobs=-1,
    )


def fit_final_model(
    train_data: pd.DataFrame,
    *,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> GridSearchCV:
    """Batch 1 전체에서 최종 alpha를 선택하고 Ridge Pipeline을 학습합니다."""
    x_train, y_train = _split_xy(train_data)
    search = build_alpha_search(
        alpha_grid=alpha_grid,
        n_splits=n_splits,
        random_state=random_state,
    )
    search.fit(x_train, y_train)
    return search


def calculate_regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """수명 예측 결과를 네 가지 회귀 지표로 요약합니다."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    return {
        "MAE": float(mean_absolute_error(actual_values, predicted_values)),
        "RMSE": float(root_mean_squared_error(actual_values, predicted_values)),
        "MAPE_pct": float(
            mean_absolute_percentage_error(actual_values, predicted_values) * 100
        ),
        "R2": float(r2_score(actual_values, predicted_values)),
    }


def predict_dataset(
    fitted_model: GridSearchCV,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """셀별 실제값·예측값·오차를 포함한 표를 생성합니다."""
    x_data, y_data = _split_xy(data)
    predicted = fitted_model.predict(x_data)

    predictions = data.loc[:, ["batch", "cell_key"]].copy()
    predictions["actual"] = y_data.to_numpy(dtype=float)
    predictions["predicted"] = predicted
    predictions["error"] = predictions["predicted"] - predictions["actual"]
    predictions["absolute_error"] = predictions["error"].abs()
    return predictions
