import pytest
from tables_io import convert, convert_table
from tables_io.lazy_modules import apTable, lazyImport, pd
from tests.helpers.utilities import check_deps
from tables_io import types
import numpy as np


def test_types():
    """Test the typing functions"""
    assert not types.is_table_like(4)
    assert types.is_table_like(dict(a=np.ones(4)))
    assert not types.is_table_like(dict(a=np.ones(4), b=np.ones(5)))
    assert not types.is_tabledict_like(4)
    assert not types.is_tabledict_like(dict(a=np.ones(4)))
    assert types.is_tabledict_like(dict(data=dict(a=np.ones(4))))
    try:
        types.file_type("xx.out")
    except KeyError:
        pass
    else:
        raise KeyError("Failed to catch unknown fileType")


@pytest.mark.skipif(not check_deps([apTable, pd]), reason="Missing an IO package")
def test_type_finders():
    """Test the utils that identify apTables and data frames"""
    import pandas as pd
    from astropy.table import Table

    class DataFrameSub(pd.DataFrame):
        pass

    class TableSub(Table):
        pass

    d1 = pd.DataFrame()
    d2 = DataFrameSub()
    t1 = Table()
    t2 = TableSub()

    assert types.is_dataframe(d1)
    assert types.is_dataframe(d2)
    assert not types.is_dataframe(t1)
    assert not types.is_dataframe(t2)
    assert not types.is_dataframe({})
    assert not types.is_dataframe(77)

    assert not types.is_ap_table(d1)
    assert not types.is_ap_table(d2)
    assert types.is_ap_table(t1)
    assert types.is_ap_table(t2)
    assert not types.is_ap_table({})
    assert not types.is_ap_table(77)


def test_get_table_type(data_tables):
    """Testing the Getting the Table Type Functionality"""

    conv_tabledict = convert(data_tables, "astropyTable")
    testing_type = types.get_table_type(conv_tabledict)
    assert testing_type == "astropyTable"

    # Failing Multiple Types in Same TableDict
    with pytest.raises(TypeError):
        conv_tabledict["md"] = convert_table(conv_tabledict["md"], "pyarrowTable")
        _ = types.get_table_type(conv_tabledict)

    # Failing Type that Shouldn't Work
    with pytest.raises(TypeError):
        types.get_table_type({1, 2, 3})


def test_table_types_error():
    """Testing for Tables Types Errors"""
    with pytest.raises(TypeError):
        types.table_type({"tab1": 5})
