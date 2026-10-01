# ESS 배터리 수명 예측

배터리의 초기 100사이클 데이터만으로 EOL(초기 용량의 80% 도달)까지의 총 Cycle Life를 예측하는 프로젝트입니다. 장기간 수명시험이 끝나기 전에 배터리 수명을 추정하여 셀 조기 선별, ESS 운영 계획 및 유지보수 의사결정을 지원하는 것을 목적으로 합니다.

## 프로젝트 개요

- 데이터셋: MIT–Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터: Batch 1 (`2017-05-12`)
- 평가 데이터: Batch 2 (`2018-02-20`), Batch 3 (`2018-04-12`)
- 태스크: Regression
- Target: `cycle_life`
- 입력 범위: 각 셀의 초기 100사이클
- 최종 입력 피처: `log_delta_q_variance`, `mean_chargetime`
- 선정 모델: Ridge Regression

Batch 1에는 과제 기준의 단수명 셀(`<500`사이클)이 존재하지 않아 장·단수명 이진 분류 대신 Cycle Life 회귀 문제로 설계했습니다.

## 파일 구조

```text
.
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── features.py
│   └── train.py
├── results/
│   ├── README.md
│   ├── model_performance.csv
│   ├── predictions.csv
│   └── error_analysis.csv
├── requirements.txt
└── README.md
```

## 환경 설정

Python 3.14.6 환경을 기준으로 작성했습니다.

```bash
git clone https://github.com/sugi136/data-mini-prj.git
cd data-mini-prj

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python -m ipykernel install --user \
  --name data-mini-prj \
  --display-name "Python (data-mini-prj)"
```

원본 데이터는 저장소에 포함되지 않습니다. 필요한 파일과 경로는 [data/README.md](data/README.md)를 참고합니다.

## EDA

### Cycle Life 분포

- Batch 1: 46개, 534~1,227사이클, 평균 844.7사이클
- Batch 2: 유효 Target 39개, 392~1,186사이클, 평균 565.7사이클
- Batch 3: 유효 Target 44개, 541~1,935사이클, 평균 1,059.7사이클
- 핵심 발견: Batch별 Target 분포 차이가 커서 데이터를 합친 무작위 분할보다 Batch 단위 외부 평가가 필요합니다.

### 열화 곡선 분석

- 방전 용량은 일정하게 감소하지 않고 수명 후반부에 급격히 감소합니다.
- Knee point는 평균적으로 전체 수명의 약 75~80% 지점에서 관찰됩니다.
- 핵심 발견: Knee 이후 열화 속도가 크게 증가하지만, 전체 수명으로 계산한 Knee 값은 미래 정보이므로 모델 입력에서 제외합니다.

### ΔQ(V) 곡선 분석

- `ΔQ100–10(V) = Q100(V) - Q10(V)`로 계산했습니다.
- 단수명 셀은 장수명 셀보다 음의 변화 폭과 분산이 크게 나타났습니다.
- `log10(var(ΔQ100–10))`와 Cycle Life의 상관계수는 Batch별 `-0.702~-0.902`입니다.
- 핵심 발견: `log_delta_q_variance`를 핵심 수명 예측 피처로 선정했습니다.

### 충전 속도(C-rate)와 수명의 관계

- 높은 C-rate가 항상 단수명을 의미하지는 않았습니다.
- 최대 C-rate뿐 아니라 해당 전류가 적용된 충전 구간도 함께 고려해야 합니다.
- 핵심 발견: 충전 조건의 영향을 반영하는 초기 평균 충전 시간 `mean_chargetime`을 두 번째 피처로 선정했습니다.

### 초기 피처 상관관계 및 다중공선성

- `log_delta_q_variance`가 모든 Batch에서 Cycle Life와 가장 강하고 일관된 음의 관계를 보였습니다.
- `mean_Tavg`와 `mean_Tmax`는 높은 상관관계를 보여 중복 정보 가능성이 확인되었습니다.
- 핵심 발견: 소표본에서 불필요한 복잡도를 줄이기 위해 최종 입력을 두 피처로 제한했습니다.

## Modeling

### 피처 엔지니어링 전략

각 셀의 초기 100사이클을 하나의 행으로 요약합니다.

| 피처 | 의미 | 선정 근거 |
|---|---|---|
| `log_delta_q_variance` | 10사이클과 100사이클 사이 ΔQ(V) 분산의 로그값 | Cycle Life와 강하고 일관된 음의 관계 |
| `mean_chargetime` | 초기 100사이클 평균 충전 시간 | 충전 속도와 프로토콜 영향을 반영 |

다음 값은 모델 입력에서 제외합니다.

- Knee point 및 Knee 이후 기울기: 전체 수명 정보를 사용하는 미래 정보
- `cycle_life`에서 만든 장·단수명 변수: Target Leakage
- `batch`, `cell_id`: 데이터 식별자
- `mean_Tmax`: 다른 온도 피처와 높은 다중공선성

### 모델 선택 및 근거

- 최종 모델: Ridge Regression
- 학습 표본이 46개이므로 복잡한 비선형 모델보다 단순한 선형 모델을 선택했습니다.
- 두 피처와 Cycle Life 사이에서 선형 관계가 관찰되었습니다.
- 두 입력 피처의 Batch 1 상관계수는 약 `-0.66`으로, L2 규제를 통해 회귀계수의 변동을 완화합니다.
- 두 피처를 모두 유지하므로 피처 제거를 수행하는 ElasticNet보다 Ridge가 단순합니다.
- 트리 모델과 달리 학습 Target 범위를 넘어서는 값에 대한 외삽이 가능합니다.
- `StandardScaler`와 Ridge를 하나의 Pipeline으로 구성하고, Batch 1 교차검증에서 `alpha`를 결정합니다.

### 학습 및 평가 원칙

- Batch 1: 모델 학습과 교차검증
- Batch 2: 1차 외부 평가
- Batch 3: 2차 외부 평가
- 결측값 처리와 스케일링은 각 교차검증 학습 fold에서만 `fit`
- Batch 2·3은 최종 평가에만 사용하며 모델이나 하이퍼파라미터 선택에 사용하지 않음
- 모든 무작위 단계에 `random_state` 고정

## 성능 결과

| 평가 데이터 | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| Batch 1 Nested CV | 75.02 | 92.97 | 8.79% | 0.741 |
| Batch 2 Test | 145.81 | 163.61 | 29.43% | 0.444 |
| Batch 3 Test | 156.71 | 258.08 | 12.54% | 0.308 |

Batch 1의 Nested CV에서 전처리와 `alpha` 선택을 각 학습 fold 안에서 수행했습니다. Batch 1 전체로 최종 모델을 학습할 때 선택된 Ridge `alpha`는 `0.0001`입니다.

## 오류 분석

- 가장 큰 오차는 Batch 3의 장수명 셀 `Batch 3-C38`에서 발생했습니다.
- 실제 수명은 1,935사이클이지만 1,028사이클로 예측해 약 907사이클 과소예측했습니다.
- Batch 3의 1,600사이클 이상 장수명 셀들이 주요 과소예측 사례로 나타났습니다.
- Batch 2의 400사이클 전후 단수명 셀 일부는 약 270~303사이클 과대예측했습니다.
- Batch 1의 Target 범위가 534~1,227사이클이므로 외부 Batch의 극단적인 장·단수명을 충분히 학습하지 못한 것으로 해석할 수 있습니다.
- 선택된 `alpha`가 매우 작고 `mean_chargetime`의 표준화 계수도 작아, 현재 모델은 `log_delta_q_variance`에 대부분 의존합니다.

## ESS 도메인 해석

### 활용 가능성

- 장기 수명시험 이전의 셀 조기 선별
- ESS 셀 배치 및 품질관리 우선순위 결정
- 예상 수명을 고려한 점검·교체 계획 수립

### 한계 및 추가 요구사항

- 소수의 실험실 셀로 학습되어 실제 ESS 운전 환경을 충분히 대표하지 못합니다.
- 온도, SOC 범위, 휴지 시간, 부하 패턴 등 실제 운전 조건이 추가로 필요합니다.
- 다른 제조사와 화학계 셀에 대한 외부 검증이 필요합니다.
- 운영 중 Data Drift와 Model Drift 감시 및 재학습 기준이 필요합니다.
- 본 모델의 결과는 안전 진단을 단독으로 대체할 수 없습니다.

## 참고문헌

- Severson, K. A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391. https://doi.org/10.1038/s41560-019-0356-8

## 팀 구성

- 팀원명 및 역할: 작성 예정
