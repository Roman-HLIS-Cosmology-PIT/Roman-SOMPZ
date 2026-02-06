"""Tests for Table Slicing Utilities"""

import pytest
from tables_io import types, convert
from tables_io.utils.slice_utils import slice_table, slice_tabledict
from tests.helpers.utilities import compare_tables, check_deps
from tables_io.lazy_modules import tables, apTable, apDiffUtils, fits, h5py, pd, pq, jnp


@pytest.mark.skipif(
    not check_deps([apTable, pd]), reason="Missing panda or astropy.table"
)
@pytest.mark.parametrize(
    "tType",
    [
        types.AP_TABLE,
        types.NUMPY_DICT,
        types.NUMPY_RECARRAY,
        types.PA_TABLE,
        types.PD_DATAFRAME,
    ],
)
def test_slice(data_tables, tType):
    """Testing Slicing of Tables and Tabledicts"""
    odict_1 = convert(data_tables, tType)
    single_table = odict_1["data"]

    check_data_1 = slice_table(single_table, slice(0, 50))

    t_dict_1 = convert(check_data_1, types.AP_TABLE)

    test_dict = slice_table(data_tables["data"], slice(0, 50))
    test_dict2 = slice_tabledict(data_tables, slice(0, 10))

    assert compare_tables(test_dict, t_dict_1)
    assert compare_tables(test_dict2["data"], data_tables["data"][0:10])
