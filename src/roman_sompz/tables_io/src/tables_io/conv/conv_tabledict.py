"""Multi-Table Convert Functions for tables_io"""

from collections import OrderedDict

import numpy as np
from typing import Mapping, Union

from .conv_table import (
    convert_table,
    convert_to_ap_table,
    convert_to_dataframe,
    convert_to_dict,
    convert_to_pa_table,
    convert_to_recarray,
)
from ..utils.array_utils import force_to_pandables
from ..lazy_modules import apTable, fits, pd, pa
from ..types import (
    AP_TABLE,
    NUMPY_DICT,
    NUMPY_RECARRAY,
    PD_DATAFRAME,
    PA_TABLE,
    TABULAR_FORMAT_NAMES,
    TABULAR_FORMATS,
    is_table_like,
    table_type,
    tType_to_int,
)


### II.  Multi-table conversion utilities
def convert(obj, tType: Union[str, int]):
    """
    Converts `Table-like` or `TableDict-like` objects to a specific tabular format.
    The given table format type must be one of the supported types listed below.

    If given a `TableDict-like` object, each of the `Table-like` objects in it are
    converted to the desired format. `convert_obj` is used to convert the `Table-like`
    objects.

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
    obj :  'Table-like` or `TableDict-like`
        The input object.

    tType : `int` or `str`
        The table format to convert to.

    Returns
    -------
    out :  `Table-like` or `TableDict-like`
        The converted data.

    Example
    -------

    Converting a `Table-like` object, in this case an `astropy.table.Table` object,
    to a `pyarrow.Table` object:

    >>> import tables_io
    >>> from astropy.table import Table
    >>> tab = Table([[1,2], [3,4]],names=('x','y'))
    >>> tables_io.convert(tab,'pyarrowTable')
    pyarrow.Table
    x: int64
    y: int64
    ----
    x: [[1,2]]
    y: [[3,4]]

    """
    if is_table_like(obj):
        return convert_table(obj, tType)

    funcMap = {
        AP_TABLE: convert_to_ap_tables,
        NUMPY_DICT: convert_to_dicts,
        NUMPY_RECARRAY: convert_to_recarrays,
        PA_TABLE: convert_to_pa_tables,
        PD_DATAFRAME: convert_to_dataframes,
    }

    # Convert tType to int if necessary
    int_tType = tType_to_int(tType)

    try:
        theFunc = funcMap[int_tType]
    except KeyError as msg:
        raise KeyError(
            f"Unsupported tabular type {int_tType} ({TABULAR_FORMATS[int_tType]}, must be one of {TABULAR_FORMAT_NAMES})"
        ) from msg
    return theFunc(obj)


def convert_to_ap_tables(odict: Mapping) -> Mapping:
    """
    Convert several `objects` to `astropy.table.Table`

    Parameters
    ----------
    odict :  `Mapping`, (`str`, `Table-like`)
        The input objects

    Returns
    -------
    tabs : `OrderedDict` of `astropy.table.Table`
        The tables
    """
    return OrderedDict([(k, convert_to_ap_table(v)) for k, v in odict.items()])


def convert_to_dicts(odict: Mapping) -> Mapping:
    """
    Convert several `objects` to `OrderedDict`, (`str`, `numpy.array`)

    Parameters
    ----------
    odict :  `Mapping`, (`str`, `Table-like`)
        The input objects

    Returns
    -------
    tabs : `OrderedDict` of `OrderedDict`, (`str`, `numpy.array`)
        The tables
    """
    return OrderedDict([(k, convert_to_dict(v)) for k, v in odict.items()])


def convert_to_recarrays(odict: Mapping) -> Mapping:
    """
    Convert several `objects` to `np.recarray`

    Parameters
    ----------
    odict :  `Mapping`, (`str`, `Table-like`)
        The input objects

    Returns
    -------
    tabs : `OrderedDict` of `np.recarray`
        The tables
    """
    return OrderedDict([(k, convert_to_recarray(v)) for k, v in odict.items()])


def convert_to_pa_tables(odict: Mapping) -> Mapping:
    """
    Convert several `objects` to `pa.Table`

    Parameters
    ----------
    odict :  `Mapping`, (`str`, `Table-like`)
        The input objects

    Returns
    -------
    tabs : `OrderedDict` of `np.recarray`
        The tables
    """
    return OrderedDict([(k, convert_to_pa_table(v)) for k, v in odict.items()])


def convert_to_dataframes(odict: Mapping) -> Mapping:
    """
    Convert several `objects` to `pandas.DataFrame`

    Parameters
    ----------
    odict :  `Mapping`, (`str`, `Table-like`)
        The input objects

    Returns
    -------
    df :  `OrderedDict` of `pandas.DataFrame`
        The dataframes
    """
    return OrderedDict([(k, convert_to_dataframe(v)) for k, v in odict.items()])
