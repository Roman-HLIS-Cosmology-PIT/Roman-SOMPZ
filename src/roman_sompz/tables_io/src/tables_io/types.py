"""Type definitions and related functions for tables_io"""

import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Union, Optional

import numpy as np

from .utils.array_utils import array_length
from .lazy_modules import pa

# Tabular data formats
AP_TABLE = 0
NUMPY_DICT = 1
NUMPY_RECARRAY = 2
PD_DATAFRAME = 3
PA_TABLE = 4

TABULAR_FORMAT_NAMES = OrderedDict(
    [
        ("astropyTable", AP_TABLE),
        ("numpyDict", NUMPY_DICT),
        ("numpyRecarray", NUMPY_RECARRAY),
        ("pandasDataFrame", PD_DATAFRAME),
        ("pyarrowTable", PA_TABLE),
    ]
)

TABULAR_FORMATS = OrderedDict([(val, key) for key, val in TABULAR_FORMAT_NAMES.items()])


# File Formats
ASTROPY_FITS = 0
ASTROPY_HDF5 = 1
NUMPY_HDF5 = 2
NUMPY_FITS = 3
PANDAS_HDF5 = 4
PANDAS_PARQUET = 5
PYARROW_HDF5 = 6
PYARROW_PARQUET = 7
PANDAS_CSV = 8


FILE_FORMAT_NAMES = OrderedDict(
    [
        ("astropyFits", ASTROPY_FITS),
        ("astropyHdf5", ASTROPY_HDF5),
        ("numpyHdf5", NUMPY_HDF5),
        ("numpyFits", NUMPY_FITS),
        ("pyarrowHdf5", PYARROW_HDF5),
        ("pandasHdf5", PANDAS_HDF5),
        ("pandaParquet", PANDAS_PARQUET),
        ("pyarrowParquet", PYARROW_PARQUET),
        ("pandasCsv", PANDAS_CSV),
    ]
)

# Default suffixes for various file formats
FILE_FORMAT_SUFFIXS = OrderedDict(
    [
        ("fits", ASTROPY_FITS),
        ("hf5", ASTROPY_HDF5),
        ("hdf5", NUMPY_HDF5),
        ("fit", NUMPY_FITS),
        ("h5", PANDAS_HDF5),
        ("hd5", PYARROW_HDF5),
        ("parquet", PYARROW_PARQUET),
        ("parq", PANDAS_PARQUET),
        ("pq", PANDAS_PARQUET),
        ("csv", PANDAS_CSV),
    ]
)

DEFAULT_TABLE_KEY = OrderedDict(
    [
        ("fits", ""),
        ("hf5", None),
        ("hdf5", None),
        ("hd5", "data"),
        ("fit", ""),
        ("h5", "data"),
        ("parquet", ""),
        ("parq", ""),
        ("pq", ""),
        ("csv", ""),
    ]
)

FILE_FORMATS = OrderedDict([(val, key) for key, val in FILE_FORMAT_NAMES.items()])

FILE_FORMAT_SUFFIX_MAP = OrderedDict(
    [(val, key) for key, val in FILE_FORMAT_SUFFIXS.items()]
)

# Default format to write various table types
NATIVE_FORMAT = OrderedDict(
    [
        (AP_TABLE, ASTROPY_HDF5),
        (NUMPY_DICT, NUMPY_HDF5),
        (NUMPY_RECARRAY, NUMPY_FITS),
        (PD_DATAFRAME, PANDAS_PARQUET),
        (PA_TABLE, PYARROW_PARQUET),
    ]
)

NATIVE_TABLE_TYPE = OrderedDict([(val, key) for key, val in NATIVE_FORMAT.items()])

# Allowed formats to write various table types
ALLOWED_FORMATS = OrderedDict(
    [
        (AP_TABLE, [ASTROPY_FITS, ASTROPY_HDF5]),
        (NUMPY_DICT, [NUMPY_HDF5]),
        (NUMPY_RECARRAY, [ASTROPY_FITS]),
        (PD_DATAFRAME, [PANDAS_PARQUET, PANDAS_HDF5, PANDAS_CSV]),
        (PA_TABLE, [PYARROW_PARQUET, PANDAS_PARQUET, PANDAS_HDF5]),
    ]
)

TABLE_FORMAT = {}
for key, val_list in ALLOWED_FORMATS.items():
    for val in val_list:
        TABLE_FORMAT[val] = key


def is_dataframe(obj):
    for c in obj.__class__.__mro__:
        if c.__name__ == "DataFrame" and c.__module__ in ["pandas.core.frame", "pandas"]:
            return True
    return False


def is_ap_table(obj):
    for c in obj.__class__.__mro__:
        if c.__name__ == "Table" and c.__module__ == "astropy.table.table":
            return True
    return False


def is_pa_table(obj):
    for c in obj.__class__.__mro__:
        if c.__name__ == "Table" and c.__module__ == "pyarrow.lib":
            return True
    return False


def table_type(obj) -> int:
    """Identify the type of table we have

    Parameters
    ----------
    obj : `object`
        The input object

    Returns
    -------
    otype : `int`
        The object type, one of `TABULAR_FORMATS.keys()`

    Raises
    ------
    TypeError
        The object is not a supported type
    IndexError
        One of the columns in a Mapping is the wrong length
    """
    if is_dataframe(obj):
        return PD_DATAFRAME
    if is_ap_table(obj):
        return AP_TABLE
    if is_pa_table(obj):
        return PA_TABLE
    if isinstance(obj, (np.recarray, np.ma.core.MaskedArray)):
        return NUMPY_RECARRAY
    if not isinstance(obj, Mapping):
        raise TypeError(
            f"Object of type {type(obj)} is not one of the supported types. \n Must be one of {list(TABULAR_FORMAT_NAMES.keys())}"
        )

    nRow = None
    for key, val in obj.items():
        if is_table_like(val):
            raise TypeError(f"Column {key} is a table of type {type(val)}")
        if not isinstance(val, Iterable):
            raise TypeError(f"Column {key} of type {type(val)} is not iterable")
        if nRow is None:
            nRow = array_length(val)
        else:
            if array_length(val) != nRow:
                raise IndexError(
                    f"Column {key} length {array_length(val)} != {nRow}. \n Column lengths are not equal, this is not a valid {TABULAR_FORMATS[NUMPY_DICT]} object"
                )  # pylint: disable=bad-string-format-type
    return NUMPY_DICT


def is_table_like(obj) -> bool:
    """Test to see if an object is one of the supported table types

    Parameters
    ----------
    obj : `object`
        The input object

    Returns
    -------
    table-like : `bool`
        True is the object is `Table-like`, False otherwise
    """
    try:
        _ = table_type(obj)
    except (TypeError, IndexError):
        return False
    return True


def is_tabledict_like(obj) -> bool:
    """Test to see if an object is a `Mapping`, (`str`, `Table-like`),
    or `TableDict-like`.

    Parameters
    ----------
    obj : `object`
        The input object

    Returns
    -------
    tabledict : `bool`
        True is the object is a `Mapping`, (`str`, `Table-like`), False otherwise
    """
    if not isinstance(obj, Mapping):
        return False
    for val in obj.values():
        if not is_table_like(val):
            return False
    return True


def file_type(filepath: str, fmt: Optional[str] = None) -> int:
    """Identify the type of file we have

    Parameters
    ----------
    filepath : `str`
        The path to the file

    fmt : `str` or `None`
        Overrides the file extension

    Returns
    -------
    otype : `int`
        The object type, one of `FILE_FORMATS.keys()`

    Raises
    ------
    KeyError
        The file format is not a support value
    """
    if fmt is None:
        fmt = os.path.splitext(filepath)[1][1:]
    try:
        return FILE_FORMAT_SUFFIXS[fmt]
    except KeyError as msg:
        raise KeyError(
            f"Unknown file format {fmt}, supported types are"
            f"{list(FILE_FORMAT_SUFFIXS.keys())}"
        ) from msg


def tType_to_int(tType: Union[str, int]) -> int:
    """Takes table type as an `int` or `str`, and converts it to the corresponding
      `int` if it's a `str`.

    Parameters
    ----------
    tType : Union[str, int]
        The tabular format

    Returns
    -------
    int
        The number corresponding to the tabular format

    Raises
    ------
    TypeError
        Raised if the given `str` is not one of the available tabular format options.
    """

    if isinstance(tType, str):
        try:
            int_tType = TABULAR_FORMAT_NAMES[tType]
        except:
            raise TypeError(
                f"Unsupported tableType '{tType}', must be one of {TABULAR_FORMAT_NAMES}"
            )
    if isinstance(tType, int):
        int_tType = tType

    return int_tType


def get_table_type(obj) -> str:
    """Gets the table type of a Table-like or TableDict-like object, and returns the name of that type.
    If the object is a TableDict-like object, it will check that all Table-like objects have the same type.
    Will raise an error if the object is not of a supported type.

    Parameters
    ----------
    obj : `Table-like` or `TableDict-like` object
        The object to determine the type of.

    Returns
    -------
    str
        Name of the tabular type

    Raises
    ------
    TypeError
        Raises a TypeError if the table is not of a supported type
    """

    # check if object is TableDict-like
    is_td = is_tabledict_like(obj)

    if is_td:
        # get the table type of the tables in the TableDict
        tab_types = []
        for keys in obj.keys():
            tab_types.append(table_type(obj[keys]))

        # if there is more than one table type raise an error
        if len(np.unique(tab_types)) > 1:
            raise TypeError(
                f"Object contains Table-like objects of multiple types (obj: {obj})"
            )
        int_tType = tab_types[0]

    else:

        try:
            # get the table type of the Table
            int_tType = table_type(obj)
        except Exception as e:
            raise TypeError(
                f"Object of type {type(obj)} is not one of the supported types. \n Must be one of {list(TABULAR_FORMAT_NAMES.keys())}"
            ) from e

    tType = TABULAR_FORMATS[int_tType]

    return tType
