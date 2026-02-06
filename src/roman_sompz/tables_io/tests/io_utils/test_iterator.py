"""Tests for Iterator Functions"""

import pytest
import numpy as np

from tables_io import io_utils
from tests.helpers.utilities import check_deps
from tables_io.lazy_modules import h5py, pq, pd


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_get_input_data_length_hdf5(h5_no_group_file, h5_data_file):
    """Test the get_input_data_size_hdf5 function"""
    assert io_utils.iterator.get_input_data_length_HDF5(h5_no_group_file) == 10
    try:
        _ = io_utils.iterator.get_input_data_length_HDF5(h5_data_file, "df")
    except ValueError:
        pass
    else:
        raise ValueError("Failed to catch ValueError for mismatched column lengths")


@pytest.mark.skipif(
    not check_deps([h5py, pq, pd]), reason="Missing HDF5, parquet or csv"
)
def test_get_input_data_length(parquet_data_file, h5_no_group_file, csv_data_file):
    """Test the get_input_data_size function"""
    assert io_utils.iterator.get_input_data_length(parquet_data_file) == 10
    assert io_utils.iterator.get_input_data_length(h5_no_group_file) == 10
    assert io_utils.iterator.get_input_data_length(csv_data_file) == 10
    with pytest.raises(NotImplementedError):
        io_utils.iterator.get_input_data_length("dummy.fits")


@pytest.mark.skipif(not check_deps([pq]), reason="Missing parquet")
def test_get_input_data_length_pq(parquet_data_file):
    """Test the get_input_data_size_pq function"""
    assert io_utils.iterator.get_input_data_length_pq(parquet_data_file) == 10


@pytest.mark.skipif(not check_deps([pd]), reason="Missing csv")
def test_get_input_data_length_csv(csv_data_file):
    """Test the get_input_data_size_csv function"""
    assert io_utils.iterator.get_input_data_length_csv(csv_data_file) == 10


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_iter_chunk_hdf5_data(h5_no_group_file):
    """Test the hdf5 data chunk iterator"""
    for itr in io_utils.iterator.iter_HDF5_to_dict(h5_no_group_file, chunk_size=2):
        for val in itr[2].values():
            assert np.size(val) == 2

    for itr in io_utils.iterator.iter_HDF5_to_dict(h5_no_group_file, chunk_size=3):
        for val in itr[2].values():
            assert np.size(val) <= 3
