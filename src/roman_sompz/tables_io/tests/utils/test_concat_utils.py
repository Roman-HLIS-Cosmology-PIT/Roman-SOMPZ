"""
Unit tests for concatenation tools
"""

import pytest
from tables_io import types, convert, concat, slice_table, slice_tabledict
from tests.helpers.utilities import compare_tables, check_deps
from tables_io.lazy_modules import apTable, pd
from tables_io.utils import concat_utils


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
def test_concat(data_tables, tType):
    """Test Concatenation across multiple types"""
    odict_1 = convert(data_tables, tType)
    odict_2 = concat([odict_1, odict_1], tType)
    concat_data = odict_2["data"]
    check_data_1 = slice_table(concat_data, slice(0, 50))
    check_data_2 = slice_table(concat_data, slice(1000, 1050))
    t_dict_1 = convert(check_data_1, types.AP_TABLE)
    t_dict_2 = convert(check_data_2, types.AP_TABLE)
    test_dict = slice_table(data_tables["data"], slice(0, 50))
    test_dict2 = slice_tabledict(data_tables, slice(0, 10))
    assert compare_tables(test_dict, t_dict_1)
    assert compare_tables(test_dict, t_dict_2)
    assert compare_tables(test_dict2["data"], data_tables["data"][0:10])


def test_concat_table(data_tables):
    """Testing Error on Table Type in Concatenate"""
    with pytest.raises(NotImplementedError):
        concat_utils.concat_table([data_tables, data_tables], 10)
