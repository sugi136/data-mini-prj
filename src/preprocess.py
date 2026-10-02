# 배터리 원본 데이터 전처리 모듈
# 데이터 로드와 셀 단위 데이터 정리 기능 구현

from pathlib import Path

import mat73
import numpy as np
import pandas as pd

# 파일 경로 정의
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

BATCH_1_PATH = DATA_DIR / "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
BATCH_2_PATH = DATA_DIR / "2018-02-20_batchdata_updated_struct_errorcorrect.mat"
BATCH_3_PATH = DATA_DIR / "2018-04-12_batchdata_updated_struct_errorcorrect.mat"

# 셀 공통 정보의 열 순서를 유지합니다.
META_COLUMNS = ["cell_id", "cycle_life", "charging_policy"]

# 원본 필드 이름을 30번에서 사용한 표의 열 이름에 대응시킵니다.
SUMMARY_FIELDS = {
    "cycle": "cycle",
    "QDischarge": "QD",
    "QCharge": "QC",
    "IR": "IR",
    "Tmax": "Tmax",
    "Tavg": "Tavg",
    "Tmin": "Tmin",
    "chargetime": "chargetime",
}


# 데이터셋 불러오기
def load_mat(file_path):
    """MAT 파일을 로드합니다. mat73 형식의 경우 mat73 라이브러리를 사용합니다."""
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    # mat73에는 경로를 문자열로 전달합니다.
    return mat73.loadmat(str(file_path))


def to_list_of_dicts(batch):
    """배치 데이터를 셀 단위의 딕셔너리 목록으로 변환합니다.
    배치 데이터가 딕셔너리 형태일 경우, 각 셀의 데이터를 딕셔너리로 묶어 리스트로 반환합니다.
    이미 셀 단위의 목록이면 그대로 반환합니다.
    """

    if isinstance(batch, dict):
        keys = list(batch.keys())
        cell_count = len(batch[keys[0]])

        return [
            {key: batch[key][index] for key in keys}
            for index in range(cell_count)
        ]

    return batch

def _to_vector(value):
    """MATLAB에서 읽은 값을 1차원 숫자 배열로 변환합니다."""
    # MATLAB 배열의 불필요한 차원을 제거하고 숫자 배열로 통일합니다.
    return np.asarray(value, dtype=float).reshape(-1)


def _read_cell_meta(cell_id, cell):
    """Summary와 Qdlin 표에서 공통으로 사용하는 셀 정보를 추출합니다."""
    # Target이 없거나 빈 배열이면 결측값으로 남깁니다.
    life_values = _to_vector(cell.get("cycle_life"))
    cycle_life = float(life_values[0]) if life_values.size else np.nan

    # 읽기 쉬운 충전 정책을 우선 사용하고, 없으면 원본 정책을 사용합니다.
    policy = cell.get("policy_readable")
    if policy is None:
        policy = cell.get("policy")
    charging_policy = str(policy) if policy is not None else "unknown"

    return {
        "cell_id": cell_id,
        "cycle_life": cycle_life,
        "charging_policy": charging_policy,
    }


def _read_summary_arrays(cell_id, summary):
    """Summary를 열별 배열로 변환하고 사이클 정렬에 필요한 길이를 검사합니다."""
    arrays = {
        column_name: _to_vector(summary[source_name])
        for source_name, column_name in SUMMARY_FIELDS.items()
    }

    # 같은 위치의 값들이 같은 사이클을 가리키려면 길이가 같아야 합니다.
    lengths = {
        column_name: len(values)
        for column_name, values in arrays.items()
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"셀 {cell_id}의 summary 필드 길이가 다릅니다: {lengths}"
        )
    return arrays


def extract_summary(batch):
    """배치 데이터를 셀·사이클별 DataFrame으로 반환합니다.
    각 셀의 summary 필드에서 필요한 열을 추출하고,
    cycle_life와 charging_policy를 포함한 표 형태로 정리합니다.
    """
    records = []

    for cell_id, cell in enumerate(batch):
        metadata = _read_cell_meta(cell_id, cell)
        arrays = _read_summary_arrays(cell_id, cell["summary"])

        for row_index in range(len(arrays["cycle"])):
            records.append({
                **metadata,
                **{
                    column_name: values[row_index]
                    for column_name, values in arrays.items()
                },
            })

    columns = [
        *META_COLUMNS, *SUMMARY_FIELDS.values(),
    ]
    return pd.DataFrame(records, columns=columns)

def _validate_cycle_number(cycle_number):
    """추출할 사이클 번호가 1 이상의 정수인지 검사합니다."""
    # 사이클 번호 유효성 검사
    if isinstance(cycle_number, bool) or not isinstance(cycle_number, int):
        raise TypeError("cycle_number는 정수여야 합니다.")

    if cycle_number < 1:
        raise ValueError("cycle_number는 1 이상이어야 합니다.")


def _read_qdlin(cell_id, cell, cycle_number):
    """한 셀에서 지정한 위치의 Qdlin을 추출하고 배열의 유효성을 검사합니다."""
    # dict 형태의 cycles를 사이클별 딕셔너리 목록으로 변환합니다.
    cycles = to_list_of_dicts(cell["cycles"])

    # cycle_number가 유효한지 확인합니다.
    if len(cycles) < cycle_number:
        raise ValueError(
            f"셀 {cell_id}: 요청한 사이클은 {cycle_number}이지만 "
            f"데이터는 {len(cycles)}개입니다."
        )

    # cycle_number에 해당하는 Qdlin 배열을 추출합니다.
    # 기존과 동일하게 배열의 위치를 사용합니다. (10번째는 인덱스 9)
    qdlin_values = _to_vector(cycles[cycle_number - 1]["Qdlin"])

    # Qdlin 배열이 비어있거나 NaN/무한대가 있는지 확인합니다.
    if qdlin_values.size == 0:
        raise ValueError(
            f"셀 {cell_id}: Cycle {cycle_number}의 Qdlin이 비어 있습니다."
        )

    if not np.isfinite(qdlin_values).all():
        raise ValueError(
            f"셀 {cell_id}: Cycle {cycle_number}의 Qdlin에 "
            "NaN 또는 무한대가 있습니다."
        )
    return qdlin_values


# Cycle 10·100 Qdlin 배열 추출
def extract_cycle_qdlin(batch, cycle_number):
    """각 셀에서 지정한 사이클의 Qdlin 곡선을 DataFrame으로 반환합니다.
    Qdlin은 전압 축에 대해 선형 보간된 방전 용량 곡선이며,
    반환하는 표는 셀당 한 행이고 Qdlin 열에 1차원 배열을 저장합니다.
    """
    _validate_cycle_number(cycle_number)
    records = []

    for cell_id, cell in enumerate(batch):
        qdlin_values = _read_qdlin(cell_id, cell, cycle_number)
        records.append({
            **_read_cell_meta(cell_id, cell),
            "Qdlin": qdlin_values,
        })

    columns = [*META_COLUMNS, "Qdlin"]
    return pd.DataFrame(records, columns=columns)









# 테스트 코드
def _inspect_batch(file_path):
    """배치의 로딩·표 생성·데이터 품질 점검 결과를 출력합니다."""
    print(f"로딩 중: {Path(file_path).name}")

    mat = load_mat(file_path)
    batch = to_list_of_dicts(mat["batch"])

    print(f"셀 개수: {len(batch)}")
    print(f"첫 번째 셀의 필드: {list(batch[0].keys())}")

    # 모델링에 필요한 데이터가 읽혔는지 첫 번째 셀로 확인합니다.
    cell = batch[0]
    print("Cycle Life:", cell["cycle_life"])
    print("Summary 필드:", list(cell["summary"].keys()))
    print("Cycles 타입:", type(cell["cycles"]).__name__)

    # 셀별 요약 데이터를 하나의 표로 정리합니다.
    df = extract_summary(batch)

    print("DataFrame 크기:", df.shape)
    print(df.head())

    # 사이클별 결측 개수와 Target이 없는 셀을 확인합니다.
    print("\n열별 결측 개수:")
    print(df.isna().sum())

    missing_target_cells = df.loc[
        df["cycle_life"].isna(), "cell_id"
    ].unique()
    print("\nTarget 결측 셀:", missing_target_cells)

    qd10_df = extract_cycle_qdlin(batch, 10)
    qd100_df = extract_cycle_qdlin(batch, 100)

    print("Cycle 10 추출 셀 수:", len(qd10_df))
    print("Cycle 100 추출 셀 수:", len(qd100_df))
    print("Cycle 10 첫 셀 배열 길이:", qd10_df.iloc[0]["Qdlin"].size)
    print("Cycle 100 첫 셀 배열 길이:", qd100_df.iloc[0]["Qdlin"].size)

    # cell_id로 연결하여 같은 셀의 두 사이클을 비교합니다.
    qd_pairs = qd10_df[["cell_id", "Qdlin"]].merge(
        qd100_df[["cell_id", "Qdlin"]],
        on="cell_id",
        suffixes=("_10", "_100"),
        validate="one_to_one",
    )
    
    length_matches = (
        qd_pairs["Qdlin_10"].apply(len)
        == qd_pairs["Qdlin_100"].apply(len)
    )
    
    print("배열 길이가 다른 셀:",
          qd_pairs.loc[~length_matches, "cell_id"].tolist())

    # 모델 입력 범위인 초기 100사이클만 점검합니다.
    early_df = df[df["cycle"].between(1, 100)].copy()
    
    # 평균 충전 시간 계산에 영향을 주는 0 이하의 값을 찾습니다.
    invalid_charge_rows = early_df[
        early_df["chargetime"] <= 0
    ]
    
    print("\n초기 100사이클의 충전 시간 0 이하 행:")
    print(
        invalid_charge_rows[
            ["cell_id", "cycle", "chargetime", "QD", "QC", "Tavg", "Tmax"]
        ].to_string(index=False)
    )
    
    print("\n사이클별 발생 건수:")
    print(
        invalid_charge_rows.groupby("cycle")
        .size()
        .rename("count")
    )
    
    print("\n해당 값이 있는 셀 수:",
          invalid_charge_rows["cell_id"].nunique())


if __name__ == "__main__":
    _inspect_batch(BATCH_1_PATH)
