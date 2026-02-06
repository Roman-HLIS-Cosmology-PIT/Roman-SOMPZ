"""IO Read Functions for tables_io"""

import os
import json
from collections import OrderedDict

import numpy as np
from typing import List, Mapping, Optional, Union

from ..utils.array_utils import force_to_pandables
from ..conv.conv_tabledict import convert
from ..conv.conv_table import dataframe_to_dict, hdf5_group_to_dict
from ..lazy_modules import apTable, fits, h5py, pa, pd, pq, ds
from ..types import (
    AP_TABLE,
    ASTROPY_FITS,
    ASTROPY_HDF5,
    DEFAULT_TABLE_KEY,
    FILE_FORMAT_SUFFIX_MAP,
    FILE_FORMAT_SUFFIXS,
    FILE_FORMATS,
    NATIVE_FORMAT,
    NATIVE_TABLE_TYPE,
    NUMPY_FITS,
    NUMPY_HDF5,
    PA_TABLE,
    PANDAS_HDF5,
    PANDAS_PARQUET,
    PANDAS_CSV,
    PYARROW_HDF5,
    PYARROW_PARQUET,
    PD_DATAFRAME,
    TABULAR_FORMAT_NAMES,
    file_type,
    is_tabledict_like,
    is_table_like,
    table_type,
)


# I. Top-level interface functions


def read(
    filepath: str,
    tType: Union[int, str, None] = None,
    fmt: Optional[str] = None,
    keys: Optional[List[str]] = None,
    allow_missing_keys: bool = False,
    **kwargs,
):
    """Reads in a given file to either a `Table-like` format if there is one table within the file,
    or a `TableDict-like` format if there are multiple tables or files. Uses :py:func:`read_native` to read the file.

    The `TableDict-like` format is an `OrderedDict` of `Table-like` objects. The `Table-like` objects
    currently supported are: `astropyTable`, `numpyRecarray`, `numpyDict` (dict of `numpy` arrays), `pandasDataFrame`,
    and `pyarrowTable`.

    If given just the filepath, the function will read any tables in the file to its default `Table-like`
    format in memory. If given a specific tabular type, the function will read in the file to the default
    type and then convert to the requested type.

    The `keys` argument is required when reading in multi-dataset parquet files, to specify which
    dataset files to read in. Otherwise, the only required argument is the filepath.

    Accepted tabular types:

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
    filepath : `str`
        Full path to the file to load
    tType : `int`, `str` or `None`
        Table type, if `None` the default table type will be used.
    fmt : `str` or `None`
        File format, if `None` it will be taken from the file extension.
    keys : `list` or `None`
        This argument is required for reading multiple associated parquet files.
        The keys should be the unique identifiers for each dataset or file.
    allow_missing_keys : `bool`, by default False
        If False will raise FileNotFoundError if a key is missing from the given file.
    kwargs :
        Additional arguments to pass to the native file reader

    Returns
    -------
    data : `OrderedDict` ( `str` -> `Table-like` )
        The data

    Example
    -------

        For a single `Table-like` object, we can read it in as follows:

        >>> import tables_io
        >>> df = tables_io.read('filename.h5')
        >>> print(df)
           col1  col2
        0     1     3
        1     2     4

        Notice that it has been automatically read in as the default tabular type for `h5` files,
        a `pandasDataFrame`.

        For a `TableDict-like` object, we read it in as follows:

        >>> table_dict = tables_io.read('filename.hdf5', tType='astropyTable')
        >>> table_dict
        OrderedDict({'tab_1': <Table length=2>
          x     y
        int64 int64
        ----- -----
            2     1
            4     3, 'tab_2': <Table length=2>
          a     b
        int64 int64
        ----- -----
            5     3
            7     4})

        Notice that the resulting `OrderedDict` has `astropyTable` objects as the values.


    """

    odict = read_native(filepath, fmt, keys, allow_missing_keys, **kwargs)

    if len(odict) == 1:
        # For special keys, use the table alone without an enclosing dictionary.
        single_dict_key = list(odict.keys())[0]
        if single_dict_key in ["", None, "__astropy_table__", "data"]:
            odict = odict[single_dict_key]
    if tType is None:  # pragma: no cover
        return odict

    return convert(odict, tType)


def read_native(
    filepath: str,
    fmt: Optional[str] = None,
    keys: Optional[List[str]] = None,
    allow_missing_keys: bool = False,
    **kwargs,
):
    """Reads in a file to its corresponding default tabular format.

    The format of the file is either given by `fmt`, or determined based on the `suffix` of
    the file path. This determines what tabular format the file is read in as. In all cases,
    the data from the file is returned as an `OrderedDict` or `TableDict-like` object, with
    `str` keys and `Table-like` values. The `Table-like` values can be `astropyTable`,
    `numpyRecarray`, `numpyDict` (dict of `numpy` arrays), `pandasDataFrame`, and
    `pyarrowTable`.

    Parameters
    ----------
    filepath : `str`
        Full path of the file to load
    fmt : `str` or `None`
        File format, if `None` it will be taken from the file extension.
    keys : `list` or `None`
        This argument is required for reading multiple associated parquet files.
        The keys should be the unique identifiers for each dataset or file.
    allow_missing_keys : `bool`, by default False.
        If False will raise FileNotFoundError if a key is missing from the given file.
    kwargs :
        Additional arguments to pass to the native file reader

    Returns
    -------
    data : `OrderedDict` ( `str` -> `Table-like` )
        The data

    Example
    -------

    Reading in a file that is in `NUMPY_HDF5` format:

    >>> import tables_io
    >>> tab = tables_io.read_native('filename.hdf5')
    >>> print(tab)
    OrderedDict({'tab_1': OrderedDict({'col_1': array([0., 2.]), 'col_2': array([2., 3.])}),
    'tab_2': OrderedDict({'col_a': array([1., 1.]), 'col_b': array([3., 3.])})})


    """
    fType = file_type(filepath, fmt)
    if fType == ASTROPY_FITS:
        try:
            return read_fits_to_ap_tables(filepath, keys=keys)
        except Exception as e:
            raise RuntimeError(
                read_native_error_message(
                    filepath, fType, fmt, keys, allow_missing_keys, **kwargs
                )
                + f" \n because of error: \n {e}"
            ) from e

    if fType == ASTROPY_HDF5:
        try:
            return read_HDF5_to_ap_tables(filepath, keys=keys)
        except Exception as e:
            raise RuntimeError(
                read_native_error_message(
                    filepath, fType, fmt, keys, allow_missing_keys, **kwargs
                )
                + f" \n because of error: \n {e}"
            ) from e
    if fType == NUMPY_HDF5:
        try:
            return read_HDF5_to_dicts(filepath, keys=keys)
        except Exception as e:
            raise RuntimeError(
                read_native_error_message(
                    filepath, fType, fmt, keys, allow_missing_keys, **kwargs
                )
                + f" \n because of error: \n {e}"
            ) from e
    if fType == NUMPY_FITS:
        try:
            return read_fits_to_recarrays(filepath, keys=keys)
        except Exception as e:
            raise RuntimeError(
                read_native_error_message(
                    filepath, fType, fmt, keys, allow_missing_keys, **kwargs
                )
                + f" \n because of error: \n {e}"
            ) from e
    if fType == PANDAS_HDF5:
        try:
            return read_H5_to_dataframes(filepath, keys=keys)
        except Exception as e:
            raise RuntimeError(
                read_native_error_message(
                    filepath, fType, fmt, keys, allow_missing_keys, **kwargs
                )
                + f" \n because of error: \n {e}"
            ) from e
    if fType == PANDAS_PARQUET:
        try:
            return read_pq_to_dataframes(filepath, keys, allow_missing_keys, **kwargs)
        except Exception as e:
            raise RuntimeError(
                read_native_error_message(
                    filepath, fType, fmt, keys, allow_missing_keys, **kwargs
                )
                + f" \n because of error: \n {e}"
            ) from e
    if fType == PYARROW_HDF5:
        try:
            return read_HDF5_to_tables(filepath, keys)
        except Exception as e:
            raise RuntimeError(
                read_native_error_message(
                    filepath, fType, fmt, keys, allow_missing_keys, **kwargs
                )
                + f" \n because of error: \n {e}"
            ) from e
    if fType == PYARROW_PARQUET:
        try:
            return read_pq_to_tables(filepath, keys, allow_missing_keys, **kwargs)
        except Exception as e:
            raise RuntimeError(
                read_native_error_message(
                    filepath, fType, fmt, keys, allow_missing_keys, **kwargs
                )
                + f" \n because of error: \n {e}"
            ) from e
    if fType == PANDAS_CSV:
        try:
            return read_csv_to_dataframes(filepath, keys, allow_missing_keys, **kwargs)
        except Exception as e:
            raise RuntimeError(
                read_native_error_message(
                    filepath, fType, fmt, keys, allow_missing_keys, **kwargs
                )
                + f" \n because of error: \n {e}"
            ) from e
    raise TypeError(
        f"Unsupported FileType {fType}. Supported types are: {list(FILE_FORMATS.values())}"
    )  # pragma: no cover


def io_open(filepath: str, fmt: Optional[str] = None, **kwargs):
    """Returns the file object. This allows you to
    open large files without reading the whole file into memory.

    It opens the file object with different packages depending on the file type. It uses
    astropy to open FITS files (`astropy.io.fits.open()`), h5py for any HDF5 files (`h5py.File()`), or
    pyarrow parquet for any parquet files (`pyarrow.parquet.ParquetFile()`). You can specify which
    file type you are supplying via the `fmt` argument, or it will automatically determine the file type
    from its suffix.

    If the given file is not one of the supported types, it will raise a TypeError.


    Parameters
    ----------
    filepath : `str`
        The path to the file to load.
    fmt : `str` or `None`
        The file format, if `None` it will be taken from the file extension.

    Returns
    -------
    File object. One of `pyarrow.parquet.ParquetFile`, `h5py.File` or `astropy.io.fits.HDUList`.

    Example
    -------

    For example, to read in a sample `fits` file:

    >>> import tables_io
    >>> hdul = tables_io.io_open("./data/test.fits", "fits")
    >>> hdul.info()
    No.    Name      Ver    Type      Cards   Dimensions   Format
      0  PRIMARY       1 PrimaryHDU       4   ()
      1  DF            1 BinTableHDU     37   10R x 14C   [K, E, E, E, E, E, E, E, E, E, E, E, E, D]

    """
    fType = file_type(filepath, fmt)
    if fType in [ASTROPY_FITS, NUMPY_FITS]:
        return fits.open(filepath, **kwargs)
    if fType in [ASTROPY_HDF5, NUMPY_HDF5, PANDAS_HDF5, PYARROW_HDF5]:
        return h5py.File(filepath, **kwargs)
    if fType in [PYARROW_PARQUET, PANDAS_PARQUET]:
        # basepath = os.path.splitext(filepath)[0]
        return pq.ParquetFile(filepath, **kwargs)
    if fType in [PANDAS_CSV]:
        if "iterator" not in kwargs:
            kwargs["iterator"] = True
        return pd.read_csv(filepath, **kwargs)
    raise TypeError(
        f"Unsupported FileType {fType}. Supported types are: {list(FILE_FORMATS.values())}"
    )  # pragma: no cover


def check_columns(
    filepath: str,
    columns_to_check: List[str],
    fmt: Optional[str] = None,
    parent_groupname: Optional[str] = None,
    **kwargs,
):
    """Read the file column names from file and ensure that it contains at least
    the columns specified in a provided list. If not, an error will be raised.

    * For FITS files, columns across all extensions will be checked at one time.
    * For HDF5 files, only columns within a single level of the specified parent_groupname will be checked.

    Note: If more columns are available in the file than specified in the list,
    the file will still pass the check.

    Parameters
    ----------
    filepath : `str`
        File name for the file to read. If there's no suffix, it will be applied based on the object type.
    columns_to_check: `list`
        A list of columns to be compared with the data
    fmt : `str` or `None`
        The input file format, If `None` this will use `io_open`
    parent_groupname: `str` or `None`
        For hdf5 files, the groupname for the data
    """

    fType = file_type(filepath, fmt)

    # Read the file below:
    file = io_open(filepath, fmt=None, **kwargs)

    if fType in [ASTROPY_FITS, NUMPY_FITS]:
        col_list = []
        for hdu in file[1:]:
            columns = hdu.columns
            for col in columns:
                if col.name not in col_list:
                    col_list.append(col.name)

    elif fType in [ASTROPY_HDF5, NUMPY_HDF5, PANDAS_HDF5, PYARROW_HDF5]:
        col_list = read_HDF5_group_names(filepath, parent_groupname=parent_groupname)

    elif fType in [PYARROW_PARQUET, PANDAS_PARQUET]:
        col_list = file.schema.names

    elif fType in [PANDAS_CSV]:
        data = file.read(nrows=1)
        col_list = data.columns.tolist()

    else:
        raise TypeError(
            f"Unsupported FileType {fType}. Supported types are: {list(FILE_FORMATS.values())}"
        )  # pragma: no cover

    # check columns
    intersection = set(columns_to_check).intersection(col_list)
    if len(intersection) < len(columns_to_check):
        diff = set(columns_to_check) - intersection
        raise KeyError("The following columns are not found: ", diff)


# II. Reading Files


# II A. Reading `astropy.table.Table` from FITS files


def read_fits_to_ap_tables(filepath: str, keys: Optional[List[str]] = None) -> Mapping:
    """
    Reads `astropy.table.Table` objects into an `OrderedDict` TableDict-like object from a FITS file.
    If a list of keys is given, will read only those tables.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list` or `None`
        A list of which tables to read, in lower case.

    Returns
    -------
    tables : `OrderedDict` of `astropy.table.Table`
        Keys will be HDU names, values will be tables
    """
    fin = fits.open(filepath)
    tables = OrderedDict()
    for i, hdu in enumerate(fin[1:]):
        if keys is not None:
            if hdu.name.lower() not in keys:
                continue

        # In base case, handle cases where no names are provided or
        # names are repeated. If no names are provided and more than one table
        # is in the FITS file, use string of extension number as its name

        ext_num = i + 1
        tab_name = hdu.name.lower()
        if (tab_name == "") & (len(fin) > 2):
            tab_name = str(ext_num)

        # Checking for repeated names:
        if tab_name in tables.keys():
            tab_name = f"{tab_name}_{str(ext_num)}"

        tables[tab_name] = apTable.Table.read(filepath, hdu=ext_num)
    return tables


# II B Reading `np.recarray` from FITS files


def read_fits_to_recarrays(filepath: str, keys: Optional[List[str]] = None) -> Mapping:
    """
    Reads `np.recarray` objects into an `OrderedDict` TableDict-like object from a FITS file.
    If a list of keys is given, will read only those tables.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list` or `None`
        A list of which HDU names to read, in lower case.

    Returns
    -------
    tables : `OrderedDict` of `np.recarray`
        Keys will be HDU names, values will be tables
    """
    fin = fits.open(filepath)
    tables = OrderedDict()
    for i, hdu in enumerate(fin[1:]):
        if keys is not None and hdu.name.lower() not in keys:
            continue

        # In base case, handle cases where no names are provided or
        # names are repeated. If no names are provided and more than one table
        # is in the FITS file, use string of extension number as its name

        ext_num = i + 1
        tab_name = hdu.name.lower()
        if (tab_name == "") & (len(fin) > 2):
            tab_name = str(ext_num)

        # Checking for repeated names:
        if tab_name in tables.keys():
            tab_name = f"{tab_name}_{str(ext_num)}"

        tables[tab_name] = hdu.data

    return tables


# II C Reading `astropy.table.Table` from HDF5 file


def read_HDF5_to_ap_tables(filepath: str, keys: Optional[List[str]] = None) -> Mapping:
    """
    Reads `astropy.table.Table` objects into an `OrderedDict` TableDict-like object from an hdf5 file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list` or `None`
        A list of which datasets to read in.

    Returns
    -------
    tables : `OrderedDict` of `astropy.table.Table`
        Keys will be 'paths', values will be tables
    """
    fin = h5py.File(filepath)
    tables = OrderedDict()
    for k in fin.keys():
        if keys is not None and k not in keys:
            continue
        tables[k] = apTable.Table.read(filepath, path=k, format="hdf5")
    return tables


## II D. Reading `OrderedDict` (`str`, `numpy.array`) and `np.array` from HDF5 file


def read_HDF5_group(filepath: str, groupname: Optional[str] = None):
    """Read and return the requested group and file object from an hdf5 file. If no group is provided, returns the `h5py.File` object twice.

    Parameters
    ----------
    filepath : `str`
        File in question
    groupname : `str` or `None`
        The name or path to the desired group.

    Returns
    -------
    grp : `h5py.Group` or `h5py.File`
        The requested group
    infp : `h5py.File`
        The input file (returned so that the user can explicitly close the file)
    """
    infp = h5py.File(filepath, "r")
    if groupname is None or not groupname:  # pragma: no cover
        return infp, infp
    return infp[groupname], infp


def read_HDF5_group_to_dict(hg, start: Optional[int] = None, end: Optional[int] = None):
    """
    Reads `numpy.array` objects from an open hdf5 file object. If given a dataset, returns a `numpy.array` of that dataset.
    If given a group, it will read `numpy.array` objects into an `OrderedDict` for all of the keys in that group.
    If start and end are provided, it will only read in the given slice [start:end] of all the datasets.

    Parameters
    ----------
    hg: `hdf5` object
        The hdf5 object to read in, either a dataset or a group.

    start : `int` or `None`
        Starting row of dataset(s) to read.

    end : `int` or `None`
        Ending row of dataset(s) to read.

    Returns
    -------
    tables : `OrderedDict` of `numpy.array` or a `numpy.array`
        Keys will be 'paths', values will be arrays in the case of an `OrderedDict`.
    """
    # pylint: disable=unused-argument
    if isinstance(hg, h5py.Dataset):
        return read_HDF5_dataset_to_array(hg, start, end)
    return OrderedDict(
        [(key, read_HDF5_dataset_to_array(val, start, end)) for key, val in hg.items()]
    )


def read_HDF5_group_names(
    filepath: str, parent_groupname: Optional[str] = None
) -> List[str]:
    """Read and return the list of group names from one level of an hdf5 file.

    Parameters
    ----------
    filepath : `str`
        File in question
    parent_groupname : `str` or `None`
        For hdf5 files, the parent groupname. All group names under this will be
        returned. If `None`, return the top level group names.

    Returns
    -------
    names : `list` of `str`
        The names of the groups in the file
    """
    infp = h5py.File(filepath, "r")
    if parent_groupname is None:  # pragma: no cover
        return list(infp.keys())

    try:
        subgroups = infp[parent_groupname].keys()
    except KeyError as msg:
        raise KeyError(
            f"Group {parent_groupname} not found in file {filepath}"
        ) from msg
    return list(subgroups)


def read_HDF5_to_dicts(filepath: str, keys: Optional[List[str]] = None) -> Mapping:
    """
    Reads `numpy.array` objects into an `OrderedDict` from an hdf5 file. If a list of keys is given,
    will only read those specific datasets.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list` or `None`
        A list of which tables to read from the file.

    Returns
    -------
    dicts : `OrderedDict`, (`str`, `OrderedDict`, (`str`, `numpy.array`) )
        The data
    """
    fin = h5py.File(filepath)
    l_out = []
    for key, val in fin.items():
        if keys is not None and key not in keys:
            continue
        l_out.append((key, read_HDF5_group_to_dict(val)))
    return OrderedDict(l_out)


def read_HDF5_dataset_to_array(
    dataset, start: Optional[int] = None, end: Optional[int] = None
) -> np.array:
    """Reads all or part of a hdf5 dataset into a `numpy.array`

    Parameters
    ----------
    dataset : `h5py.Dataset`
        The input dataset

    start : `int` or `None`
        Starting row

    end : `int` or `None`
        Ending row

    Returns
    -------
    out : `numpy.array`
        Something that pandas can handle
    """
    if start is None or end is None:
        return np.array(dataset)
    return np.array(dataset[start:end])


# II D. Reading `pandas.DataFrame` from HDF5


def read_H5_to_dataframe(filepath: str, key: Optional[str] = None):
    """
    Reads `pandas.DataFrame` objects from an 'h5' file (a pandas `hdf5` file).

    Parameters
    ----------
    filepath: `str`
        Path to input file
    key : `str` or `None`
        The key in the hdf5 file

    Returns
    -------
    df : `pandas.DataFrame`
        The dataframe
    """
    return pd.read_hdf(filepath, key)


def read_H5_to_dataframes(filepath: str, keys: Optional[List[str]] = None) -> Mapping:
    """Open an `h5` (pandas `hdf5`) file and and return an `OrderedDict` of `pandas.DataFrame` objects

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list` or `None`
        A list of which tables to read.

    Returns
    -------
    tab : `OrderedDict` (`str` : `pandas.DataFrame`)
       The data

    Notes
    -----
    We are using the file suffix 'h5' to specify 'hdf5' files written from DataFrames using `pandas`
    They have a different structure than 'hdf5' files written with `h5py` or `astropy.table`
    """
    fin = h5py.File(filepath)
    l_out = []
    for key in fin.keys():
        if keys is not None and key not in keys:
            continue
        l_out.append((key, read_H5_to_dataframe(filepath, key=key)))
    return OrderedDict(l_out)


# II E Reading `pandas.DataFrame` from parquet file


def read_pq_to_dataframe(filepath: str, columns: Optional[List[str]] = None, **kwargs):
    """
    Reads a `pandas.DataFrame` object from a parquet file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    columns : `list` (`str`) or `None`
        Names of the columns to read, `None` will read all the columns

    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    df : `pandas.DataFrame`
        The data frame
    """
    return pd.read_parquet(filepath, engine="pyarrow", columns=columns, **kwargs)


def read_pq_to_dataframes(
    filepath: str,
    keys: Optional[List[str]] = None,
    allow_missing_keys: bool = False,
    columns: Union[List[str], Mapping, None] = None,
    **kwargs,
) -> Mapping:
    """
    Reads `pandas.DataFrame` objects from an parquet file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list`
        Keys for the input objects.  Used to complete filepaths

    allow_missing_keys: `bool`
        If False will raise FileNotFoundError if a key is missing

    columns : `dict` of `list (str)`, `list` (`str`), or `None`
        Names of the columns to read.
            - if a dictionary, keys are the `keys`, and values are a list of string column names.
                for each keyed table, only the columns in the value list will be loaded.
                if the key is not found, all columns will be loaded.
            - if a list, only the columns in the list will be loaded.
            - `None` will read all the columns

    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    tables : `OrderedDict` of `pandas.DataFrame`
        Keys will be taken from keys
    """
    if keys is None:  # pragma: no cover
        keys = [""]
    dataframes = OrderedDict()
    basepath, ext = os.path.splitext(filepath)
    if not ext:  # pragma: no cover
        ext = "." + FILE_FORMAT_SUFFIX_MAP[PANDAS_PARQUET]
    for key in keys:
        try:
            column_list = None
            if pd.api.types.is_dict_like(columns):
                column_list = columns[key]
            elif pd.api.types.is_list_like(columns):
                column_list = columns
            print("column_list", column_list)

            dataframes[key] = read_pq_to_dataframe(
                f"{basepath}{key}{ext}", columns=column_list, **kwargs
            )
        except FileNotFoundError as msg:  # pragma: no cover
            if allow_missing_keys:
                continue
            raise msg
    return dataframes


# II F. Reading `OrderedDict` (`str`, `numpy.array`) from parquet file


def read_pq_to_dict(
    filepath: str, columns: Optional[List[str]] = None, **kwargs
) -> Mapping:
    """Open a parquet file and return an `OrderedDict` of `numpy.array` objects

    Parameters
    ----------
    filepath: `str`
        Path to input file

    columns : `list` (`str`) or `None`
        Names of the columns to read, `None` will read all the columns

    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    tab : `OrderedDict` (`str` : `numpy.array`)
       The data
    """
    tab = pq.read_table(filepath, columns=columns, **kwargs)
    return OrderedDict(
        [
            (c_name, col.to_numpy())
            for c_name, col in zip(tab.column_names, tab.itercolumns())
        ]
    )


def read_H5_to_dict(filepath: str, groupname: Optional[str] = None) -> Mapping:
    """Open an `h5` file and and return an `OrderedDict` of `numpy.array` objects.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    groupname : `str` or `None`
        The name of the group with the data

    Returns
    -------
    tab : `OrderedDict` (`str` : `numpy.array`)
       The data

    Notes
    -----
    We are using the file suffix 'h5' to specify 'hdf5' files written from DataFrames using `pandas`
    They have a different structure than 'hdf5' files written with `h5py` or `astropy.table`
    """
    df = read_H5_to_dataframe(filepath, groupname)
    return dataframe_to_dict(df)


def read_HDF5_to_dict(filepath: str, groupname: Optional[str] = None) -> Mapping:
    """Read in h5py hdf5 data, return a dictionary of all of the keys

    Parameters
    ----------
    filepath: `str`
        Path to input file

    groupname : `str` or `None`
        The groupname for the data

    Returns
    -------
    tab : `OrderedDict` (`str` : `numpy.array`)
       The data

    Notes
    -----
    We are using the file suffix 'hdf5' to specify 'hdf5' files written with `h5py` or `astropy.table`
    They have a different structure than 'h5' files written `panda`
    """
    hg, infp = read_HDF5_group(filepath, groupname)
    data = hdf5_group_to_dict(hg)
    infp.close()
    return data


# II G. Reading `pyarrow.Table` from HDF5 file


def read_HDF5_to_table(filepath: str, key: Optional[str] = None):
    """
    Reads `pyarrow.Table` objects from an hdf5 file.

    Parameters
    ----------
    filepath: `str`
        Path to input file
    key : `str` or `None`
        The key in the hdf5 file

    Returns
    -------
    table : `pyarrow.Table`
        The table
    """
    pydict = read_HDF5_to_dicts(filepath, [key])[key]
    t_dict = {}
    for key, val in pydict.items():
        t_dict[key] = force_to_pandables(val)
    return pa.Table.from_pydict(t_dict)


def read_HDF5_to_tables(filepath: str, keys: Optional[List[str]] = None) -> Mapping:
    """Open an `HDF5` file and and return an `OrderedDict` of `pyarrow.Table`

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list` or `None`
        Which tables to read

    Returns
    -------
    tab : `OrderedDict` (`str` : `pyarrow.Table`)
       The data

    """
    fin = h5py.File(filepath)
    l_out = []
    for key in fin.keys():
        if keys is not None and key not in keys:  # pragma: no cover
            continue
        l_out.append((key, read_HDF5_to_table(filepath, key=key)))
    return OrderedDict(l_out)


# II H. Reading `pyarrow.Table` from parquet file


def read_pq_to_table(filepath: str, columns: Optional[List[str]] = None, **kwargs):
    """
    Reads a `pyarrow.Table` object from an parquet file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    columns : `list` (`str`) or `None`
        Names of the columns to read, `None` will read all the columns

    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    table : `pyarrow.Table`
        The table
    """
    return pq.read_table(filepath, columns=columns, **kwargs)


def read_pq_to_tables(
    filepath: str,
    keys: Optional[List[str]] = None,
    allow_missing_keys: bool = False,
    columns: Union[List[str], Mapping, None] = None,
    **kwargs,
) -> Mapping:
    """
    Reads `pyarrow.Table` objects from a parquet file into an `OrderedDict`.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list`
        Keys for the input objects.  Used to complete filepaths

    allow_missing_keys: `bool`
        If False will raise FileNotFoundError if a key is missing. By default False.

    columns : `dict` of `list (str)`, `list` (`str`), or `None`
        Names of the columns to read.
            - if a dictionary, keys are the `keys`, and values are a list of string column names.
                for each keyed table, only the columns in the value list will be loaded.
                if the key is not found, all columns will be loaded.
            - if a list, only the columns in the list will be loaded.
            - `None` will read all the columns

    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    tables : `OrderedDict` of `pyarrow.Table`
        Keys will be taken from keys
    """
    if keys is None:  # pragma: no cover
        keys = [""]
    tables = OrderedDict()
    basepath, ext = os.path.splitext(filepath)
    if not ext:  # pragma: no cover
        ext = "." + FILE_FORMAT_SUFFIX_MAP[PANDAS_PARQUET]
    for key in keys:
        try:
            column_list = None
            if pd.api.types.is_dict_like(columns):  # pragma: no cover
                column_list = columns[key]
            elif pd.api.types.is_list_like(columns):  # pragma: no cover
                column_list = columns
            print("column_list", column_list)

            tables[key] = read_pq_to_table(
                f"{basepath}{key}{ext}", columns=column_list, **kwargs
            )
        except FileNotFoundError as msg:  # pragma: no cover
            if allow_missing_keys:
                continue
            raise msg
    return tables


# II E Reading `pandas.DataFrame` from csv file


def try_parse(val) -> Union[np.array, list, dict, str]:
    """Tries to parse a string into a numpy array or a JSON object.
    This function attempts to convert a string representation of a numpy array or a JSON object

    Parameters
    ----------
    val : `str`
        The string to parse

    Returns
    -------
    val : `numpy.array` or `list` or `dict` or `str`
        If the string is a valid numpy array or JSON object, it returns the parsed object.
        If parsing fails, it returns the original string.
    """

    try:
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            return np.array([np.float64(x) for x in val.strip("[]").split()])
    except Exception:
        pass

    try:
        if isinstance(val, str) and (val.startswith("[") or val.startswith("{")):
            return json.loads(val)
    except Exception:
        pass

    return val


def read_csv_to_dataframes(
    filepath: str,
    keys: Optional[List[str]] = None,
    allow_missing_keys: bool = False,
    columns: Union[List[str], Mapping, None] = None,
    **kwargs,
) -> Mapping:
    """
    Reads `pandas.DataFrame` objects from a csv file into an `OrderedDict`.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list`
        Keys for the input objects.  Used to complete filepaths

    allow_missing_keys: `bool`
        If False will raise FileNotFoundError if a key is missing. By default False.

    columns : `dict` of `list (str)`, `list` (`str`), or `None`
        Names of the columns to read.
            - if a dictionary, keys are the `keys`, and values are a list of string column names.
                for each keyed table, only the columns in the value list will be loaded.
                if the key is not found, all columns will be loaded.
            - if a list, only the columns in the list will be loaded.
            - `None` will read all the columns

    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    tables : `OrderedDict` of `pandas.DataFrame`
        Keys will be taken from keys
    """

    if keys is None:  # pragma: no cover
        keys = [""]
    dataframes = OrderedDict()
    basepath, ext = os.path.splitext(filepath)
    if not ext:  # pragma: no cover
        ext = "." + FILE_FORMAT_SUFFIX_MAP[PANDAS_CSV]

    for key in keys:
        try:
            column_list = None
            if pd.api.types.is_dict_like(columns):  # pragma: no cover
                column_list = columns[key]
            elif pd.api.types.is_list_like(columns):  # pragma: no cover
                column_list = columns
            print("column_list", column_list)

            if column_list is not None and "usecols" not in kwargs:
                kwargs["usecols"] = column_list

            df = pd.read_csv(f"{basepath}{key}{ext}", **kwargs)

            if isinstance(df, pd.DataFrame):
                for col in df.columns:
                    if isinstance(df[col].iloc[0], str) or not df[col].iloc:
                        df[col] = df[col].apply(try_parse)

            dataframes[key] = df
            return dataframes
        except FileNotFoundError as msg:  # pragma: no cover
            if allow_missing_keys:
                continue
            raise msg


# III. Miscellaneous


def read_native_error_message(
    filepath: str,
    fType: int,
    fmt: Optional[str],
    keys: Optional[List[str]],
    allow_missing_keys: bool,
    **kwargs,
) -> str:
    """Generates an error message to be printed out if a file cannot be read in by read_native.

    Parameters
    ----------
    filepath : `str`
        Full path of the file to load
    fmt : `str` or `None`
        File format, if `None` it will be taken from the file extension.
    keys : `list` or `None`
        This argument is required for reading multiple associated parquet files.
        The keys should be the unique identifiers for each dataset or file.
    allow_missing_keys : `bool`, by default False.
        If False will raise FileNotFoundError if a key is missing from the given file.
    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    str
        The error message string.
    """
    return f"{FILE_FORMATS[fType]} file could not be read in with the following arguments: \n filepath: '{filepath}', fmt: '{fmt}', keys: {keys}, allow_missing_keys: {allow_missing_keys}, and **kwargs: {kwargs}"
