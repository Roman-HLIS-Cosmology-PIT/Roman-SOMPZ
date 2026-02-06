import pytest
from pathlib import Path


# Paths for HDF5 Files


@pytest.fixture
def h5_data_file(test_dir) -> Path:
    """Path to test HDF5 File"""

    return test_dir / "data/pandas_test_hdf5.h5"


@pytest.fixture
def h5_no_group_file(test_dir) -> Path:
    """Path to test HDF5 File with no group"""

    return test_dir / "data/no_groupname_test.hdf5"


@pytest.fixture
def h5_test_outfile(test_dir) -> Path:
    """Path to test HDF5 File for testing writing"""

    return test_dir / "test_out.h5"


# Paths for Parquet Files


@pytest.fixture
def parquet_data_file(test_dir) -> Path:
    """Path to test Parquet File"""

    return test_dir / "data/parquet_test.parquet"


# Paths for CSV Files


@pytest.fixture
def csv_data_file(test_dir) -> Path:
    """Path to test CSV File"""

    return test_dir / "data/csv_test.csv"
