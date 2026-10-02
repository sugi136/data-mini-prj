# 배터리 수명 예측 모델의 학습과 평가 모듈

# 선정한 두 피처로 Ridge를 학습 
# -> Train CV, Hold-out Valid, Batch 2 Test 성능과 Gap 정리


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# features.py에서 정의한 모델 입력 열을 재사용합니다.
# notebooks에서 src를 import할 수 있도록 경로를 설정한 뒤 사용합니다.
from src.features import FEATURE_COLUMNS, TARGET


# 무작위 분할과 CV에서 동일한 값을 사용합니다.
RANDOM_STATE = 42

# Batch 1의 20%를 별도 검증용으로 보존하는 제안값입니다.
VALID_SIZE = 0.2

# Train 내부 교차검증 설정과 Ridge 규제 강도 후보입니다.
CV_SPLITS = 5
ALPHA_CANDIDATES = np.logspace(-4, 4, 25)

# 실행 위치와 관계없이 프로젝트의 results 폴더를 사용합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERFORMANCE_PATH = PROJECT_ROOT / "results" / "model_performance.csv"