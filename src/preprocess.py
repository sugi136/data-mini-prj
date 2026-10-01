"""MIT-Stanford 배터리 MAT 파일에서 모델 입력에 필요한 값만 읽습니다.

원본 파일은 수 GB이므로 전체 구조를 메모리에 올리지 않습니다. 각 셀의
Target, 초기 사이클 충전 시간, 10·100사이클 Qdlin만 선택적으로 읽습니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BATCH_FILENAMES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}


@dataclass(slots=True)
class CellEarlyCycleData:
    """셀 하나에서 초기 피처 계산에 필요한 최소 데이터입니다."""

    batch: str
    cell_id: int
    cell_key: str
    cycle_life: float
    cycles: np.ndarray
    charge_times: np.ndarray
    qd_cycle_10: np.ndarray
    qd_cycle_100: np.ndarray


def get_batch_files(data_dir: Path | None = None) -> dict[str, Path]:
    """Batch 이름과 원본 MAT 파일 경로의 매핑을 반환합니다."""
    resolved_data_dir = (data_dir or PROJECT_ROOT / "data").resolve()
    batch_files = {
        batch_name: resolved_data_dir / filename
        for batch_name, filename in BATCH_FILENAMES.items()
    }
    validate_batch_files(batch_files)
    return batch_files


def validate_batch_files(batch_files: Mapping[str, Path]) -> None:
    """필요한 원본 파일이 모두 존재하는지 확인합니다."""
    missing_files = [
        f"{batch_name}: {file_path}"
        for batch_name, file_path in batch_files.items()
        if not file_path.is_file()
    ]
    if missing_files:
        formatted = "\n".join(f"- {item}" for item in missing_files)
        raise FileNotFoundError(f"필요한 MAT 파일을 찾을 수 없습니다.\n{formatted}")


def _read_vector(hdf_object: h5py.Dataset) -> np.ndarray:
    """HDF5 데이터셋을 1차원 NumPy 배열로 읽습니다."""
    return np.asarray(hdf_object[()]).reshape(-1)


def _dereference(
    mat_file: h5py.File,
    reference_dataset: h5py.Dataset,
    index: int,
) -> h5py.Dataset | h5py.Group:
    """MATLAB struct의 객체 참조를 실제 HDF5 객체로 변환합니다."""
    return mat_file[reference_dataset[index, 0]]


def _read_scalar_reference(
    mat_file: h5py.File,
    reference_dataset: h5py.Dataset,
    index: int,
) -> float:
    """객체 참조가 가리키는 첫 번째 값을 실수로 읽습니다."""
    values = _read_vector(_dereference(mat_file, reference_dataset, index))
    return float(values[0]) if len(values) else np.nan


def _read_qd_cycle(
    mat_file: h5py.File,
    qdlin_references: h5py.Dataset,
    zero_based_cycle_index: int,
) -> np.ndarray:
    """지정한 사이클의 선형 보간 방전 용량 Qdlin을 읽습니다."""
    reference = qdlin_references[zero_based_cycle_index, 0]
    return _read_vector(mat_file[reference]).astype(float)


def load_batch(
    file_path: Path,
    batch_name: str,
) -> list[CellEarlyCycleData]:
    """한 Batch에서 초기 피처 생성에 필요한 데이터만 선택적으로 읽습니다."""
    cells: list[CellEarlyCycleData] = []

    with h5py.File(file_path, "r") as mat_file:
        batch_group = mat_file["batch"]
        cell_count = batch_group["cycle_life"].shape[0]

        for cell_id in range(cell_count):
            cell_key = f"{batch_name}-C{cell_id:02d}"
            cycle_life = _read_scalar_reference(
                mat_file,
                batch_group["cycle_life"],
                cell_id,
            )

            summary_group = _dereference(
                mat_file,
                batch_group["summary"],
                cell_id,
            )
            cycles = _read_vector(summary_group["cycle"]).astype(float)
            charge_times = _read_vector(summary_group["chargetime"]).astype(float)

            # Qdlin의 행 하나가 사이클 하나를 가리키는 객체 참조입니다.
            cycles_group = _dereference(
                mat_file,
                batch_group["cycles"],
                cell_id,
            )
            qdlin_references = cycles_group["Qdlin"]

            if qdlin_references.shape[0] >= 100:
                qd_cycle_10 = _read_qd_cycle(mat_file, qdlin_references, 9)
                qd_cycle_100 = _read_qd_cycle(mat_file, qdlin_references, 99)
            else:
                # 피처 생성 단계에서 제외 사유를 확인할 수 있도록 빈 배열로 둡니다.
                qd_cycle_10 = np.array([], dtype=float)
                qd_cycle_100 = np.array([], dtype=float)

            cells.append(
                CellEarlyCycleData(
                    batch=batch_name,
                    cell_id=cell_id,
                    cell_key=cell_key,
                    cycle_life=cycle_life,
                    cycles=cycles,
                    charge_times=charge_times,
                    qd_cycle_10=qd_cycle_10,
                    qd_cycle_100=qd_cycle_100,
                )
            )

    return cells


def load_all_batches(
    batch_files: Mapping[str, Path] | None = None,
) -> list[CellEarlyCycleData]:
    """Batch 1·2·3의 초기 피처 입력 데이터를 한 목록으로 읽습니다."""
    resolved_files = dict(batch_files or get_batch_files())
    validate_batch_files(resolved_files)

    all_cells: list[CellEarlyCycleData] = []
    for batch_name, file_path in resolved_files.items():
        batch_cells = load_batch(file_path, batch_name)
        all_cells.extend(batch_cells)
        print(f"{batch_name} 로딩 완료: {len(batch_cells)} cells")

    return all_cells
