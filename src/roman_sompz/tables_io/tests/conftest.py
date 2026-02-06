import pytest
import numpy as np
from tables_io.lazy_modules import tables, apTable, apDiffUtils, fits, h5py, pd, pq, jnp
from pathlib import Path


@pytest.fixture
def test_dir() -> Path:
    """Return path to test directory

    Returns
    -------
    Path
        Path to test directory
    """
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session", name="data_tables")
def data_tables():
    """Make and return some test data"""
    nrow = 1000
    vect_size = 20
    mat_size = 5
    scalar = np.random.uniform(size=nrow)
    vect = np.random.uniform(size=nrow * vect_size).reshape(nrow, vect_size)
    matrix = np.random.uniform(size=nrow * mat_size * mat_size).reshape(
        nrow, mat_size, mat_size
    )
    data = dict(scalar=scalar, vect=vect, matrix=matrix)
    table = apTable.Table(data)
    table.meta["a"] = 1
    table.meta["b"] = None
    table.meta["c"] = [3, 4, 5]
    small_table = apTable.Table(dict(a=np.ones(21), b=np.zeros(21)))
    small_table.meta["small"] = True
    out_tables = dict(data=table, md=small_table)
    return out_tables


@pytest.fixture
def data_keys(data_tables):
    """Make and return some test data"""
    return data_tables.keys()


@pytest.fixture
def data_table(data_tables):
    """Make and return some test data as an astropy table"""
    return data_tables["data"]
