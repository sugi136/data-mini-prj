# 데이터 준비

이 프로젝트는 Severson et al. (2019)의 MIT–Stanford Battery Dataset을 사용합니다.
원본 `.mat` 파일은 수 GB이므로 Git 저장소에 포함하지 않습니다.

## 필요한 파일

다음 파일을 이 `data/` 디렉터리에 배치합니다.

```text
data/
├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
├── 2018-02-20_batchdata_updated_struct_errorcorrect.mat
└── 2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

| 파일 | 역할 |
|---|---|
| `2017-05-12...mat` | Batch 1: 학습 및 교차검증 |
| `2018-02-20...mat` | Batch 2: 1차 외부 평가 |
| `2018-04-12...mat` | Batch 3: 2차 외부 평가 |

데이터 설명과 원 논문의 코드는 [공식 GitHub 저장소](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)를 참고합니다.

## 주의사항

- 원본 `.mat` 파일은 `.gitignore`에 의해 제외됩니다.
- 공개 가능한 소규모 샘플이 필요하면 `data/sample/` 아래에 저장합니다.
- 파일명이나 경로를 변경하면 노트북의 `BATCH_FILES` 설정도 함께 수정해야 합니다.
