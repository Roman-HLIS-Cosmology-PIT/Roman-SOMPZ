"""Tests for Converting Tabledicts"""

import pytest

from tables_io import types
from tables_io.conv.conv_tabledict import convert
from ..helpers.utilities import compare_table_dicts, check_deps
from tables_io.lazy_modules import tables, apTable, apDiffUtils, fits, h5py, pd, pq, jnp


@pytest.mark.parametrize(
    "mod",
    [tables, apTable, apDiffUtils, fits, h5py, pd, pq],
)
def test_deps(mod):
    assert check_deps([mod])


@pytest.mark.skipif(
    not check_deps([apTable, pd]), reason="Missing panda or astropy.table"
)
@pytest.mark.parametrize(
    "tType1",
    [
        types.AP_TABLE,
        types.NUMPY_DICT,
        types.NUMPY_RECARRAY,
        types.PA_TABLE,
        types.PD_DATAFRAME,
    ],
)
@pytest.mark.parametrize(
    "tType2",
    [
        types.AP_TABLE,
        types.NUMPY_DICT,
        types.NUMPY_RECARRAY,
        types.PA_TABLE,
        types.PD_DATAFRAME,
    ],
)
def test_convert_table_dicts(data_tables, tType1, tType2):
    """Perform type conversion on the cross-product of all types."""
    odict_1 = convert(data_tables, tType1)
    odict_2 = convert(odict_1, tType2)
    tables_r = convert(odict_2, types.AP_TABLE)
    assert compare_table_dicts(data_tables, tables_r)


@pytest.mark.skipif(
    not check_deps([apTable, pd]), reason="Missing panda or astropy.table"
)
def test_conversion_strings(data_tables, data_table):
    """Tests that the conversion functions work when given table types as strings."""

    # test convert works with a string
    tType = types.TABULAR_FORMATS[0]
    odict_1 = convert(data_tables, tType)
    assert compare_table_dicts(odict_1, data_tables)

    # test it still works without a string
    odict_2 = convert(data_tables, types.PD_DATAFRAME)

    # test it works on just one table
    tab = convert(data_table, tType)


def test_bad_table_type(data_table):
    """Testing for correct exception for bad table type"""

    with pytest.raises(KeyError):
        _ = convert([data_table], tType=500)
