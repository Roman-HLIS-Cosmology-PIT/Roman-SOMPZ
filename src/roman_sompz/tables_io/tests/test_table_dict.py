"""
Tests for the TableDict Class

To Be Deprecated in future version.
"""

import pytest

from tables_io import types, TableDict
from tables_io.conv.conv_tabledict import convert
from tests.helpers.utilities import compare_table_dicts, check_deps
from tables_io.lazy_modules import apTable, jnp, h5py, pq


@pytest.mark.skipif(
    not check_deps([apTable, h5py, pq, jnp]), reason="Missing an IO package"
)
def test_bad_type(data_tables):
    """Test adding bad types to TableDict"""
    td = TableDict(data_tables)
    with pytest.raises(TypeError, match="item 4 was not recognized as a table."):
        td["aa"] = 4


@pytest.mark.skipif(
    not check_deps([apTable, h5py, pq, jnp]), reason="Missing an IO package"
)
@pytest.mark.parametrize(
    "tType, fmt, use_keys",
    [
        (types.AP_TABLE, "fits", False),
        (types.AP_TABLE, "hf5", False),
        (types.NUMPY_DICT, "hdf5", False),
        (types.PD_DATAFRAME, "h5", False),
        (types.PD_DATAFRAME, "pq", True),
    ],
)
def test_parquet_dict(
    data_tables, data_table, data_keys, tmp_path, tType, fmt, use_keys
):
    """Test writing / reading {tType} dataframes to {fmt}"""
    basepath = str(tmp_path)

    odict_c = convert(data_tables, tType)
    td_c = TableDict(odict_c)
    filepath = td_c.write(basepath, fmt)
    td_r = TableDict.read(
        filepath, tType=tType, fmt=fmt, keys=data_keys if use_keys else None
    )
    tables_r = td_r.convert(types.AP_TABLE)
    assert compare_table_dicts(data_tables, tables_r)
    if fmt in ["pq", "h5"]:
        return
    basepath2 = "%s_v2" % basepath
    filepath2 = td_c.write(basepath2)
    td_r2 = TableDict.read(filepath2, tType=tType)
    tables_r2 = td_r2.convert(types.AP_TABLE)
    assert compare_table_dicts(data_tables, tables_r2)
