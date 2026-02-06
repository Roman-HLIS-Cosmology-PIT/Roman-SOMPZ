"""Slicing functions for tables_io"""

from collections import OrderedDict
from typing import Mapping


from .array_utils import slice_dict
from ..types import NUMPY_DICT, PD_DATAFRAME, table_type


# I F. Generic `slice`
def slice_table(obj, the_slice: slice):
    """
    Slice a `Table-like` object. The slice may be supplied as a single integer,
    or as a python `slice(start,stop,step)` object.

    Parameters
    ----------
    obj :  `table-like`
        Table like object to slice

    the_slice: `slice` or `int`
        The slice of the object to take. Either a single integer, or a
        python `slice(start, stop, step)` object.

    Returns
    -------
    tab : `table-like`
        The slice of the table

    Example
    -------

        >>> import tables_io
        >>> import pandas as pd
        >>> df = pd.DataFrame({'col1': [1,2,3], 'col2':[3,4,5]})
        >>> tables_io.slice_table(df, slice(1,2))
           col1  col2
        1     2     4

    """
    tType = table_type(obj)
    if tType is NUMPY_DICT:
        return slice_dict(obj, the_slice)
    if tType is PD_DATAFRAME:
        return obj.iloc[the_slice]
    return obj[the_slice]


def slice_tabledict(odict: Mapping, the_slice: slice) -> Mapping:
    """Slice many `Table-like` objects inside a `TableDict-like` object.
    This will take the same slice from each of the `Table-like` objects,
    and return a `TableDict-like` object with those slices.

    Parameters
    ----------
    odict :  `TableDict-like`
       Dictionary of objects to slice

    the_slice: `slice`
        A python `slice(start, stop, step)` object of the slice to take.


    Returns
    -------
    odict : `TableDict-like`
        The sliced tables

    Example
    -------

        >>> import tables_io
        >>> from astropy.table import Table
        >>> odict = OrderedDict([('tab_1', Table([[1,2],[5,3]],names=("x","y"))),
        ... ('tab_2', Table([[1,2,4],[5,3,7]],names=("x","y")))])
        >>> tables_io.slice(odict, slice(2,3))
        OrderedDict([('tab_1',
              <Table length=0>
                x     y
              int64 int64
              ----- -----),
             ('tab_2',
              <Table length=1>
                x     y
              int64 int64
              ----- -----
                  4     7)])

    """
    return OrderedDict([(k, slice_table(v, the_slice)) for k, v in odict.items()])
