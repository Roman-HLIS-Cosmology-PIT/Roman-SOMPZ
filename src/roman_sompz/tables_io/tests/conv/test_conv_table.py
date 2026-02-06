"""Tests for Converting Tables"""

import pytest

from tables_io import types
import tables_io.conv.conv_table
from tables_io.conv.conv_table import convert_table
from ..helpers.utilities import compare_table_dicts, check_deps
from tables_io.lazy_modules import tables, apTable, apDiffUtils, fits, h5py, pd, pq, jnp
import numpy as np


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
def test_convert_table(data_table, tType1, tType2):
    """Perform type conversion for tables."""
    t1 = convert_table(data_table, tType1)
    t2 = convert_table(t1, tType2)
    _ = convert_table(t2, types.AP_TABLE)

    # Testing against raising errors for failed conversion:

    bad_object = ["a", "b", ["c", "d"]]

    with pytest.raises(RuntimeError) as e:
        _ = convert_table(bad_object, tType1)

    # Testing against unsupported table type

    with pytest.raises(TypeError):
        _ = convert_table(t1, 5)


def test_bad_conversion(data_table):
    """Tests that the conversion fails as expected when given an incorrect table type."""

    # Testing Case Sensitivity
    with pytest.raises(TypeError) as e:
        bad = convert_table(data_table, "astropytable")

    # Testing against out of range key
    with pytest.raises(TypeError) as e:
        bad = convert_table(data_table, 500)

    # Testing Wrong Key
    with pytest.raises(TypeError) as e:
        bad = convert_table(data_table, "CSV")


@pytest.mark.skipif(
    not check_deps([apTable, pd]), reason="Missing panda or astropy.table"
)
def test_conversion_strings(data_table):
    """Tests that the conversion functions work when given table types as strings."""

    # test convert works with a string
    tType = types.TABULAR_FORMATS[0]
    tab = convert_table(data_table, tType)


def test_bad_types(monkeypatch, data_table):
    """Tests that type errors are raised"""

    def mock_bad_type(obj):
        return 500

    # Forcing the table type to only provide invalid table types
    monkeypatch.setattr(tables_io.conv.conv_table, "table_type", mock_bad_type)

    with pytest.raises(TypeError):
        _ = tables_io.conv.conv_table.convert_to_ap_table(data_table)

    with pytest.raises(TypeError):
        _ = tables_io.conv.conv_table.convert_to_dict(data_table)

    with pytest.raises(TypeError):
        _ = tables_io.conv.conv_table.convert_to_recarray(data_table)

    with pytest.raises(TypeError):
        _ = tables_io.conv.conv_table.convert_to_dataframe(data_table)

    with pytest.raises(TypeError):
        _ = tables_io.conv.conv_table.convert_to_pa_table(data_table)


def test_not_implemented(data_table):
    """Testing that Not-Implemented Methods raise NotImplementedError"""

    with pytest.raises(NotImplementedError):
        _ = tables_io.conv.conv_table.pa_table_to_recarray(data_table)


def test_extracting_metadata_from_dicts():
    """Testing that metadata gets extracted appropriately from dictionaries"""

    initial_dict = {"val1": np.linspace(1, 10, 50), "val2": np.linspace(2, 10, 50)}

    meta_dict = {"meta1": 5, "meta2": "test"}

    out_pa_table = tables_io.conv.conv_table.dict_to_pa_table(
        initial_dict, meta=meta_dict
    )

    out_df = tables_io.conv.conv_table.dict_to_dataframe(initial_dict, meta=meta_dict)
