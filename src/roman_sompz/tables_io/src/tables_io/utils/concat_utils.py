"""Concatenation functions for tables_io"""

from collections import OrderedDict
from typing import Union, Optional, List, Mapping, Iterable

import numpy as np
from numpy.lib import recfunctions as rfn

from .array_utils import concatenate_dicts
from ..lazy_modules import apTable, pd, pa
from ..types import (
    AP_TABLE,
    NUMPY_DICT,
    NUMPY_RECARRAY,
    PD_DATAFRAME,
    PA_TABLE,
    tType_to_int,
)


### I. concatenating list of table-like objects


# I A. Generic `concat`
def concat_table(tableList: List, tType: Union[str, int]):
    """
    Vertically concatenates a sequence of `Table-like` objects. The concatenation
    is performed as an `outer` join, where no data is lost.

    Note: When concatenating `NUMPY_RECARRAY` objects, the output arrays will be masked
    arrays if any fill values are required by the concatenation.

    Accepted table formats:

    ==================  ===============
    Format string       Format integer
    ==================  ===============
    "astropyTable"      0
    "numpyDict"         1
    "numpyRecarray"     2
    "pandasDataFrame"   3
    "pyarrowTable"      4
    ==================  ===============

    Parameters
    ----------
    tablelist :  `list`
        The list of tables

    tType: `str` or `int`
        The tabular format of the tables given.

    Returns
    -------
    tab : `Table-like`
        The concatenated table

    Example
    -------

    >>> import tables_io
    >>> import pandas as pd
    >>> df = pd.DataFrame({'col_1': [1,2,3], 'col_2':[3,4,5]})
    >>> df_2 = pd.DataFrame({'col_2': [8,9], 'col_3': [10,11]})
    >>> tables_io.concat_table([df,df_2],'pandasDataFrame')
       col_1  col_2  col_3
    0    1.0      3    NaN
    1    2.0      4    NaN
    2    3.0      5    NaN
    3    NaN      8   10.0
    4    NaN      9   11.0

    """
    funcDict = {
        AP_TABLE: concat_ap_tables,
        NUMPY_DICT: concat_numpy_dicts,
        NUMPY_RECARRAY: concat_numpy_recarrays,
        PD_DATAFRAME: concat_dataframes,
        PA_TABLE: concat_pa_tables,
    }

    # convert tType to int if necessary
    int_tType = tType_to_int(tType)

    try:
        theFunc = funcDict[int_tType]
        return theFunc(tableList)
    except KeyError as msg:
        raise NotImplementedError(
            f"Unsupported FileType for concatObjs {tType}"
        ) from msg


### I B.  Multi-table concatenating


def concat_tabledict(odictlist: List[Mapping], tType: Union[str, int]) -> Mapping:
    """
    Vertically concatenates a list of `TableDict-like` objects. Each `Table-like` object
    in a `TableDict-like` object will be concatenated with any matching `Table-like` objects
    in the other `TableDict-like` objects (where matching means they have the same key). The
    final `TableDict-like` object will contain all unique `Table-like` objects (those with unique
    keys).

    The concatenation will be of join type `outer`, which means that no data will be lost.

    Note: If concatenating `NUMPY_RECARRAY` objects, the output arrays will be masked arrays if
    any fill values are required by the concatenation.

    Parameters
    ----------
    odictlist :  `list` of 'TableDict-like'
        The input objects
    tType: `str` or `int`
        The tabular format of the tables given.

    Returns
    -------
    tabs : `OrderedDict` of `Table-like`
        A `TableDict-like` object of the concatenated `Table-like` objects


    Example
    -------

    >>> import tables_io
    >>> from astropy.table import Table
    >>> odict_1 = OrderedDict([('tab_1', Table([[1.5,2.2],[5,3]],names=("x","y"))),
    ... ('tab_2', Table([[1,2.4,4],[5,3,7]],names=("x","y")))])
    >>> odict_2 = OrderedDict([('tab_1', Table([[5.2,7.6],[14,20],[8,16]],names=("x","y","z"))),
    ... ('tab_2', Table([[8,9.1,3],[1,4,8]],names=("x","y")))])
    >>> tables_io.concat([odict1, odict_2], ')
    OrderedDict([('tab_1',
              <Table length=4>
                 x      y     z
              float64 int64 int64
              ------- ----- -----
                  1.5     5    --
                  2.2     3    --
                  5.2    14     8
                  7.6    20    16),
             ('tab_2',
              <Table length=6>
                 x      y
              float64 int64
              ------- -----
                  1.0     5
                  2.4     3
                  4.0     7
                  8.0     1
                  9.1     4
                  3.0     8)])

    """
    odict_in = OrderedDict()

    for odict_ in odictlist:
        for key, val in odict_.items():
            if key not in odict_in.keys():
                odict_in[key] = [val]
            else:
                odict_in[key].append(val)

    return OrderedDict([(k, concat_table(v, tType)) for k, v in odict_in.items()])


### II. Concatenating specific data tables
### II A. Concatenating `astropy.table.Table`


def concat_ap_tables(tablelist: List):
    """
    Concatenate a list of `astropy.table.Table`

    Parameters
    ----------
    tablelist :  `list`
        The list of tables

    Returns
    -------
    tab : `astropy.table.Table`
        The concatenated table
    """
    return apTable.vstack(tablelist, join_type="outer")


### II B. Concatenating dicts of numpy arrays
def concat_numpy_dicts(tablelist: List):
    """
    Concatenate a list of `dicts` of `np.array` objects

    Parameters
    ----------
    tablelist :  `list`
        The list of tables

    Returns
    -------
    tab : `dict`
        The concatenated table
    """
    return concatenate_dicts(tablelist)


### II C. Concatenating numpy recarrays
def concat_numpy_recarrays(tablelist: List):
    """
    Concatenate a list of `dicts` of `np.recarray` objects

    Parameters
    ----------
    tablelist :  `list`
        The list of tables

    Returns
    -------
    tab : `dict`
        The table
    """
    return rfn.stack_arrays(tablelist)


### II D. Concatenating pandas dataframes
def concat_dataframes(tablelist: List):
    """
    Concatenate a list of `pandas.DataFrame`

    Parameters
    ----------
    tablelist :  `list`
        The tables

    Returns
    -------
    tab : `pandas.DataFrame`
        The concatenated table
    """
    return pd.concat(tablelist, join="outer", axis=0, ignore_index=True)


### II E. Concatenating pyarrow tables
def concat_pa_tables(tablelist: List):
    """
    Concatenate a list of `pyarrow.Table` objects

    Parameters
    ----------
    tablelist :  `list`
        The list of tables

    Returns
    -------
    tab : `pyarrow.Table`
        The concatenated table
    """
    return pa.concat_tables(tablelist)
