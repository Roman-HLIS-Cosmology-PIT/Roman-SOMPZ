"""Single-Table Conversion Functions for tables_io"""

from collections import OrderedDict
from typing import Union, Mapping, Optional

import numpy as np

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

### I. Single `Table-like` conversions

# I A. Generic `convert`


def convert_table(obj, tType: Union[str, int]):
    """
    Convert a `Table-like` object to a specific tabular format.

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
    obj : `Table-like`
       The object being converted
    tType : `int` or `str`
       The type of object to convert to

    Returns
    -------
    out :  `Table-like`
        The converted object
    """

    # Convert tType to an int if necessary
    int_tType = tType_to_int(tType)

    if int_tType == AP_TABLE:
        try:
            return convert_to_ap_table(obj)
        except Exception as e:
            raise RuntimeError(
                f"Could not convert object to {TABULAR_FORMATS[int_tType]} because of error: \n {e}. \n \n Object to convert: {obj}"
            ) from e
    if int_tType == NUMPY_DICT:
        try:
            return convert_to_dict(obj)
        except Exception as e:
            raise RuntimeError(
                f"Could not convert object to {TABULAR_FORMATS[int_tType]} because of error: \n {e}. \n \n Object to convert: {obj}"
            ) from e
    if int_tType == NUMPY_RECARRAY:
        try:
            return convert_to_recarray(obj)
        except Exception as e:
            raise RuntimeError(
                f"Could not convert object to {TABULAR_FORMATS[int_tType]} because of error: \n {e}. \n \n Object to convert: {obj}"
            ) from e
    if int_tType == PA_TABLE:
        try:
            return convert_to_pa_table(obj)
        except Exception as e:
            raise RuntimeError(
                f"Could not convert object to {TABULAR_FORMATS[int_tType]} because of error: \n {e}. \n \n Object to convert: {obj}"
            ) from e
    if int_tType == PD_DATAFRAME:
        try:
            return convert_to_dataframe(obj)
        except Exception as e:
            raise RuntimeError(
                f"Could not convert object to {TABULAR_FORMATS[int_tType]} because of error: \n {e}. \n \n Object to convert: {obj}"
            ) from e
    raise TypeError(
        f"Cannot convert to unsupported tableType {int_tType} ({int_tType})"
    )


### I B. Converting to `astropy.table.Table`


def pa_table_to_ap_table(table):
    """
    Convert a `pyarrow.Table` to an `astropy.table.Table`

    Parameters
    ----------
    table :  `pyarrow.Table`
        The table

    Returns
    -------
    tab : `astropy.table.Table`
        The table
    """
    df = table.to_pandas()
    return data_frame_to_ap_table(df)


def data_frame_to_ap_table(df):
    """
    Convert a `pandas.DataFrame` to an `astropy.table.Table`

    Parameters
    ----------
    df :  `pandas.DataFrame`
        The dataframe

    Returns
    -------
    tab : `astropy.table.Table`
        The table
    """
    o_dict = OrderedDict()
    for colname in df.columns:
        col = df[colname]
        if col.dtype.name == "object":
            o_dict[colname] = np.vstack(col.to_numpy())
        else:
            o_dict[colname] = col.to_numpy()
    tab = apTable.Table(o_dict)
    for k, v in df.attrs.items():
        tab.meta[k] = v
    return tab


def convert_to_ap_table(obj):
    """
    Convert an object to an `astropy.table.Table`

    Parameters
    ----------
    obj : `object`
       The object being converted

    Returns
    -------
    tab : `astropy.table.Table`
        The table
    """
    tType = table_type(obj)
    if tType == AP_TABLE:
        return obj
    if tType == NUMPY_DICT:
        return apTable.Table(obj)
    if tType == PA_TABLE:
        return pa_table_to_ap_table(obj)
    if tType == NUMPY_RECARRAY:
        return apTable.Table(obj)
    if tType == PD_DATAFRAME:
        # try this: apTable.from_pandas(obj)
        return data_frame_to_ap_table(obj)
    raise TypeError(
        f"Table is an unsupported Table Type {tType}. Must be one of {TABULAR_FORMAT_NAMES.keys()}"
    )


### I C. Converting to `OrderedDict`, (`str`, `numpy.array`)


def ap_table_to_dict(tab) -> Mapping:
    """
    Convert an `astropy.table.Table` to an `OrderedDict` of `str` : `numpy.array`

    Parameters
    ----------
    tab :  `astropy.table.Table`
        The table

    Returns
    --------
    data : `OrderedDict`,  (`str` : `numpy.array`)
        The tabledata
    """
    data = OrderedDict()
    for key, val in zip(tab.colnames, tab.itercols()):
        data[key] = np.array(val)
    return data


def recarray_to_dict(rec: np.recarray) -> Mapping:
    """
    Convert an `np.recarray` to an `OrderedDict` of `str` : `numpy.array`

    Parameters
    ----------
    rec :  `np.recarray`
        The input recarray

    Returns
    --------
    data : `OrderedDict`,  (`str` : `numpy.array`)
        The tabledata
    """
    return OrderedDict([(colName, rec[colName]) for colName in rec.dtype.names])


def dataframe_to_dict(df) -> Mapping:
    """
    Convert a `pandas.DataFrame` to an `OrderedDict` of `str` : `numpy.array`

    Parameters
    ----------
    df :  `pandas.DataFrame`
        The dataframe

    Returns
    --------
    data : `OrderedDict`,  (`str` : `numpy.array`)
        The tabledata
    """
    data = OrderedDict()
    for key in df.keys():
        col = df[key]
        if col.dtype.name == "object":
            data[key] = np.vstack(col.to_numpy())
        else:
            data[key] = np.array(col)
    return data


def pa_table_to_dict(rec) -> Mapping:
    """
    Convert an `pa.Table` to an `OrderedDict` of `str` : `numpy.array`

    Parameters
    ----------
    rec :  `pa.Table`
        The input table

    Returns
    --------
    data : `OrderedDict`,  (`str` : `numpy.array`)
        The tabledata
    """
    return OrderedDict(
        [(colName, rec[colName].to_numpy()) for colName in rec.schema.names]
    )


def hdf5_group_to_dict(hg):
    """
    Convert a `hdf5` object to an `OrderedDict`, (`str`, `numpy.array`)

    Parameters
    ----------
    hg :  `h5py.File` or `h5py.Group`
        The hdf5 object

    Returns
    --------
    data : `OrderedDict`,  (`str` : `numpy.array`)
        The tabledata
    """
    data = OrderedDict()
    for key in hg.keys():
        data[key] = np.array(hg[key])
    return data


def convert_to_dict(obj):
    """
    Convert an object to an `OrderedDict`, (`str`, `numpy.array`)

    Parameters
    ----------
    obj : `object`
       The object being converted

    Returns
    -------
    tab : `astropy.table.Table`
        The table
    """
    tType = table_type(obj)
    if tType == AP_TABLE:
        return ap_table_to_dict(obj)
    if tType == PA_TABLE:
        return pa_table_to_dict(obj)
    if tType == NUMPY_DICT:
        return obj
    if tType == NUMPY_RECARRAY:
        return recarray_to_dict(obj)
    if tType == PD_DATAFRAME:
        return dataframe_to_dict(obj)
    raise TypeError(
        f"Could not convert table because it is an unsupported TableType {tType}. Must be one of {TABULAR_FORMAT_NAMES.keys()}"
    )


### I D. Converting to `np.recarray`


def pa_table_to_recarray(tab):
    """
    Convert an `pyarrow.Table` to an `numpy.recarray`

    Parameters
    ----------
    tab :  `pyarrow.Table`
        The table

    Returns
    --------
    rec : `numpy.recarray`
        The output rec array
    """
    raise NotImplementedError()


def ap_table_to_recarray(tab):
    """
    Convert an `astropy.table.Table` to an `numpy.recarray`

    Parameters
    ----------
    tab :  `astropy.table.Table`
        The table

    Returns
    --------
    rec : `numpy.recarray`
        The output rec array
    """
    return fits.table_to_hdu(tab).data


def convert_to_recarray(obj):
    """
    Convert an object to an `numpy.recarray`

    Parameters
    ----------
    obj : `object`
       The object being converted

    Returns
    -------
    rec : `numpy.recarray`
        The output recarray
    """
    tType = table_type(obj)
    if tType == AP_TABLE:
        return ap_table_to_recarray(obj)
    if tType == NUMPY_DICT:
        return ap_table_to_recarray(apTable.Table(obj))
    if tType == NUMPY_RECARRAY:
        return obj
    if tType == PD_DATAFRAME:
        return ap_table_to_recarray(data_frame_to_ap_table(obj))
    if tType == PA_TABLE:
        return ap_table_to_recarray(pa_table_to_ap_table(obj))
    raise TypeError(
        f"Could not convert table because it is an unsupported TableType {tType}. Must be one of {TABULAR_FORMAT_NAMES.keys()}"
    )


### I E. Converting to `pandas.DataFrame`


def ap_table_to_dataframe(tab):
    """
    Convert an `astropy.table.Table` to a `pandas.DataFrame`

    Parameters
    ----------
    tab : `astropy.table.Table`
        The table

    Returns
    -------
    df :  `pandas.DataFrame`
        The dataframe
    """
    o_dict = OrderedDict()
    for colname in tab.columns:
        col = tab[colname]
        o_dict[colname] = force_to_pandables(col.data)
    df = pd.DataFrame(o_dict)
    for k, v in tab.meta.items():
        df.attrs[k] = v
    return df


def pa_table_to_dataframe(table):
    """Converts `pyarrow.Table` object to a `pandas.DataFrame` object.

    Parameters
    ----------
    table : `pyarrow.Table`
        Input table

    Returns
    -------
    df : `pandas.DataFrame`
        The converted dataframe.
    """
    df = table.to_pandas()
    return df


def dict_to_dataframe(odict: Mapping, meta: Optional[Mapping] = None):
    """
    Convert an `OrderedDict`, (`str`, `numpy.array`) to a `pandas.DataFrame`

    Parameters
    ----------
    odict : `OrderedDict`, (`str`, `numpy.array`)
        The dict

    meta : `dict` or `None`
        Optional dictionary of metadata

    Returns
    -------
    df :  `pandas.DataFrame`
        The dataframe
    """
    outdict = OrderedDict()
    for k, v in odict.items():
        outdict[k] = force_to_pandables(v)
    df = pd.DataFrame(outdict)
    if meta is not None:
        for k, v in meta.items():
            df.attrs[k] = v
    return df


def convert_to_dataframe(obj):
    """
    Convert an object to a `pandas.DataFrame`

    Parameters
    ----------
    obj : `object`
       The object being converted

    Returns
    -------
    df :  `pandas.DataFrame`
        The dataframe
    """
    tType = table_type(obj)
    if tType == AP_TABLE:
        return ap_table_to_dataframe(obj)
    if tType == NUMPY_DICT:
        return dict_to_dataframe(obj)
    if tType == NUMPY_RECARRAY:
        odict = recarray_to_dict(obj)
        return dict_to_dataframe(odict)
    if tType == PA_TABLE:
        return pa_table_to_dataframe(obj)
    if tType == PD_DATAFRAME:
        return obj
    raise TypeError(
        f"Could not convert table because it is an unsupported tableType {tType}. Must be one of {TABULAR_FORMAT_NAMES.keys()}"
    )


### I F. Converting to `pa.Table`


def ap_table_to_pa_table(tab):
    """
    Convert an `astropy.table.Table` to a `pa.Table`

    Parameters
    ----------
    tab : `astropy.table.Table`
        The table

    Returns
    -------
    table :  `pa.Table`
        The output table
    """
    o_dict = OrderedDict()
    for colname in tab.columns:
        col = tab[colname]
        ndim = len(col.data.shape)
        if ndim == 1:
            o_dict[colname] = col.data
        elif ndim > 1:
            o_dict[colname] = force_to_pandables(col.data)

    metadata = {k: str(v) for k, v in tab.meta.items()}
    table = pa.Table.from_pydict(o_dict, metadata=metadata)
    return table


def dataframe_to_pa_table(df):
    """
    Convert a `pandas.DataFrame` to an `pa.Table`

    Parameters
    ----------
    df :  `pandas.DataFrame`
        The dataframe

    Returns
    -------
    table : `pa.Table`
        The table
    """
    table = pa.Table.from_pandas(df)
    return table


def dict_to_pa_table(odict: Mapping, meta: Optional[Mapping] = None):
    """
    Convert an `OrderedDict`, (`str`, `numpy.array`) to a `pa.Table`

    Parameters
    ----------
    odict : `OrderedDict`, (`str`, `numpy.array`)
        The dict

    meta : `dict` or `None`
        Optional dictionary of metadata

    Returns
    -------
    table :  `pa.Table`
        The table
    """
    out_dict = {key: force_to_pandables(val) for key, val in odict.items()}
    if meta is not None:
        metadata = {k: str(v) for k, v in meta.items()}
    else:
        metadata = None

    table = pa.Table.from_pydict(out_dict, metadata=metadata)
    return table


def convert_to_pa_table(obj):
    """
    Convert an object to a `pa.Table`

    Parameters
    ----------
    obj : `object`
       The object being converted

    Returns
    -------
    table :  `pa.Table`
        The table
    """
    tType = table_type(obj)
    if tType == AP_TABLE:
        return ap_table_to_pa_table(obj)
    if tType == NUMPY_DICT:
        return dict_to_pa_table(obj)
    if tType == NUMPY_RECARRAY:
        odict = recarray_to_dict(obj)
        return dict_to_dataframe(odict)
    if tType == PD_DATAFRAME:
        return dataframe_to_pa_table(obj)
    if tType == PA_TABLE:
        return obj
    raise TypeError(
        f"Could not convert table because it is an unsupported tableType {tType}. Must be one of {TABULAR_FORMAT_NAMES.keys()}"
    )
