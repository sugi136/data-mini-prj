"""배터리 수명 예측 모델의 학습·평가 함수.

- 피처와 Target 분리 및 입력 검사
- Train/Valid 데이터 분할
- StandardScaler와 Ridge Pipeline 구성
- 교차검증을 통한 alpha 선택 및 모델 학습
- Hold-out 평가 및 Nested CV의 MAPE(%) 계산
- 평균 예측 기준 모델 구성 및 교차검증 평가
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from src.features import CELL_ID, FEATURE_COLUMNS, TARGET

# 모델 학습과 평가에 사용할 상수들을 정의합니다.
RANDOM_STATE = 42

# Hold-out 검증 세트 비율을 정의합니다.
VALID_SIZE = 0.2

# Ridge 모델 학습 시 사용할 교차 검증 분할 수와 alpha 후보군을 정의합니다.
CV_SPLITS = 5
ALPHA_CANDIDATES = np.logspace(-4, 4, 25)

# 프로젝트 루트와 성능 기록 파일 경로를 정의합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERFORMANCE_PATH = PROJECT_ROOT / "results" / "model_performance.csv"


def _make_kfold():
    """고정 시드로 셀 단위 교차검증 분할기를 생성합니다."""
    # 셀 단위로 교차검증하며, 분할 결과를 재현할 수 있도록 고정합니다.
    return KFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def _mape_percent(y_true, y_pred):
    """실제값과 예측값으로 MAPE(%)를 계산합니다."""
    # scikit-learn의 MAPE는 비율로 반환되므로 백분율로 변환합니다.
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100)


def _validate_feature_table(feature_table):
    """피처 표의 필수 열, 빈 표 여부 및 셀 식별자를 검사합니다."""

    required_columns = [*FEATURE_COLUMNS, TARGET]

    # 필요한 열이 모두 있는지 확인합니다.
    missing_columns = [
        column
        for column in required_columns
        if column not in feature_table.columns
    ]

    if missing_columns:
        raise ValueError(f"필요한 열이 없습니다: {missing_columns}")

    # feature_table이 비어 있는지 확인합니다.
    if feature_table.empty:
        raise ValueError("피처 표가 비어 있습니다.")

    # 피처 표에 cell_id 열이 있는지 확인합니다.
    if CELL_ID not in feature_table.columns:
        raise ValueError("셀 식별에 필요한 cell_id 열이 없습니다.")

    if (
        feature_table[CELL_ID].isna().any()
        or feature_table[CELL_ID].duplicated().any()
    ):
        raise ValueError("cell_id에 결측값 또는 중복이 있습니다.")


def _validate_xy(X, y):
    """피처와 Target의 유한값 여부 및 Target의 양수 조건을 검사합니다."""
    # 피처와 Target에 결측값이나 무한대가 있는지 확인합니다.
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("피처에 결측값 또는 무한대가 있습니다.")

    if not np.isfinite(y.to_numpy()).all():
        raise ValueError("Target에 결측값 또는 무한대가 있습니다.")

    # Cycle Life는 양수여야 하며 MAPE 계산에서도 0을 허용하지 않습니다.
    if y.le(0).any():
        raise ValueError("cycle_life는 0보다 커야 합니다.")


def split_features_target(feature_table):
    """피처 표를 학습용 피처와 Target으로 분리합니다."""
    _validate_feature_table(feature_table)

    # 선정한 두 피처만 입력으로 사용합니다.
    X = feature_table[FEATURE_COLUMNS].astype(float).copy()
    y = feature_table[TARGET].astype(float).copy()

    _validate_xy(X, y)
    return X, y


def split_train_valid(X, y):
    """피처와 Target을 학습용과 검증용으로 분리합니다."""

    # 각 행은 서로 다른 셀입니다.
    # random_state를 고정해 실행할 때마다 같은 셀이 분할되도록 합니다.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return X_train, X_valid, y_train, y_valid

def build_pipeline():
    """Ridge 모델 학습을 위한 파이프라인을 생성합니다."""

    return Pipeline([
        # 학습 데이터의 평균과 표준편차를 기준으로 피처를 표준화합니다.
        ("scaler", StandardScaler()),

        # L2 규제로 회귀계수의 크기를 제한합니다.
        # alpha는 이후 Train 내부 교차검증에서 선택합니다.
        ("ridge", Ridge()),
    ])

def build_baseline():
    """학습 Target의 평균을 예측하는 기준 모델을 생성합니다."""
    return DummyRegressor(strategy="mean")


def evaluate_baseline_cv(X_train, y_train):
    """Ridge의 바깥 CV와 같은 분할로 기준 모델의 MAPE(%)를 계산합니다."""
    # fold별 학습 평균 예측
    values = cross_val_predict(
        build_baseline(),
        X_train,
        y_train,
        cv=_make_kfold(),
    )

    # 셀별 검증 예측
    predictions = pd.Series(
        values,
        index=y_train.index,
        name="baseline_cv_prediction",
    )
    return predictions, _mape_percent(y_train, predictions)


def select_alpha(X_train, y_train):
    """학습용 데이터에서 교차검증을 통해 Ridge 모델의 alpha를 선택합니다."""

    search = GridSearchCV(
        estimator=build_pipeline(),

        # Pipeline의 ridge 단계에 전달할 alpha 후보입니다.
        param_grid={"ridge__alpha": ALPHA_CANDIDATES},

        # scikit-learn은 큰 점수를 선호하므로 음수 MAPE를 사용합니다.
        # 가장 높은 점수는 실제 MAPE가 가장 작은 후보입니다.
        scoring="neg_mean_absolute_percentage_error",
        cv=_make_kfold(),

        # 선택한 alpha로 전체 Train 데이터를 다시 학습합니다.
        refit=True,
    )

    # 각 fold의 학습 부분에서만 스케일러와 Ridge를 학습합니다.
    search.fit(X_train, y_train)

    return search

def evaluate_model(model, X, y):
    """학습된 Pipeline으로 예측하고 예측값과 MAPE(%)를 반환합니다."""

    # Pipeline이 학습 때 구한 기준으로 표준화한 뒤 예측합니다.
    predictions = model.predict(X)

    return predictions, _mape_percent(y, predictions)

def evaluate_train_cv(X_train, y_train):
    """Nested CV로 Train의 예측값과 MAPE(%)를 계산합니다."""

    outer_cv = _make_kfold()

    # 각 셀이 바깥 검증 fold에 속했을 때의 예측값을 저장합니다.
    predictions = pd.Series(
        np.nan,
        index=y_train.index,
        name="cv_prediction",
        dtype=float,
    )

    for fit_indices, valid_indices in outer_cv.split(X_train):
        # KFold가 반환하는 값은 행 위치이므로 iloc으로 선택합니다.
        X_fold_train = X_train.iloc[fit_indices]
        y_fold_train = y_train.iloc[fit_indices]
        X_fold_valid = X_train.iloc[valid_indices]

        # 바깥 학습 fold 안에서만 alpha를 선택하고 Pipeline을 학습합니다.
        # 바깥 검증 fold는 스케일링 학습과 alpha 선택에 사용되지 않습니다.
        search = select_alpha(X_fold_train, y_fold_train)

        predictions.iloc[valid_indices] = (
            search.best_estimator_.predict(X_fold_valid)
        )

    # 모든 셀의 바깥 검증 예측을 모아 MAPE를 계산합니다.
    return predictions, _mape_percent(y_train, predictions)
