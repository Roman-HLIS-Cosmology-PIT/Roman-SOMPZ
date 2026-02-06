"""Io write functions for tables_io"""

import os
import json
from collections import OrderedDict
from typing import Mapping, Optional, Union, List

import numpy as np

from ..conv.conv_tabledict import convert
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
    file_type,
    is_tabledict_like,
    is_table_like,
    table_type,
)

# I. Top-level interface functions


def write(obj, filepath: str, fmt: Optional[str] = None) -> Optional[str]:
    """Writes `Table-like` or `TableDict-like` objects to a file or files. If the format (`fmt`) is given,
    or the file has a suffix, the function will convert the given data to the associated tabular type,
    and then write out the file as the requested type. If no file type is requested, the function will
    use :py:func:`write_native` to write the file to the default file type for the tabular type.

    Note: This function will automatically overwrite any previously existing files at the given filepath.

    Parameters
    ----------
    obj : `Table-like` or `TableDict-like`
        The data to write
    filepath : `str`
        File name for the file to write. If there's no suffix, it will be applied based on the object type.
    fmt : `str` or `None`
        The output file format, If `None` and the file path provided does not have a suffix,
        this will use `write_native` to write out the default file type for the given tabular type.

    Returns
    -------
    filepath: `str` or None
        Returns the path to the new file, or None if there was no data given.

    Example
    -------

        >>> import tables_io
        >>> import pandas as pd
        >>> tab = pd.DataFrame({'col1': [2,4,6], 'col2': [5,7,9]})
        >>> tables_io.write(tab, 'data','h5') # tells the function to write it to PANDAS_HDF5
        'data.h5'
        >>> tables_io.write(tab, 'data.h5') # does the same thing as the line above
        'data.h5'
    """
    if fmt is None:
        splitpath = os.path.splitext(filepath)
        if not splitpath[1]:
            return write_native(obj, filepath)
        fmt = splitpath[1][1:]

    try:
        fType = FILE_FORMAT_SUFFIXS[fmt]
    except KeyError as msg:  # pragma: no cover
        raise KeyError(
            f"Unknown file format {fmt} from {filepath}, options are {list(FILE_FORMAT_SUFFIXS.keys())}"
        ) from msg

    if is_table_like(obj):
        odict = OrderedDict([(DEFAULT_TABLE_KEY[fmt], obj)])
    elif is_tabledict_like(obj):
        odict = obj
    elif not obj:
        return None
    else:
        raise TypeError(f"Can not write object of type {type(obj)}")

    if fType in [ASTROPY_HDF5, NUMPY_HDF5, NUMPY_FITS, PANDAS_PARQUET, PYARROW_PARQUET]:
        try:
            nativeTType = NATIVE_TABLE_TYPE[fType]
        except KeyError as msg:  # pragma: no cover
            raise KeyError(f"Native file type not known for {fmt}") from msg

        forcedOdict = convert(odict, nativeTType)
        if os.path.splitext(filepath)[1]:
            fullpath = filepath
        else:
            fullpath = f"{filepath}.{fmt}"

        try:
            return write_native(forcedOdict, fullpath)
        except Exception as e:
            raise RuntimeError(
                f"Failed to write table to '{fullpath}' as  {FILE_FORMATS[fType]} because of error: \n {e}"
            ) from e

    if not os.path.splitext(filepath)[1]:
        filepath = filepath + "." + fmt

    try:
        if fType == ASTROPY_FITS:
            forcedOdict = convert(odict, AP_TABLE)
            write_ap_tables_to_fits(forcedOdict, filepath)
            return filepath
        if fType == PANDAS_HDF5:
            forcedOdict = convert(odict, PD_DATAFRAME)
            write_dataframes_to_HDF5(forcedOdict, filepath)
            return filepath
        if fType == PYARROW_HDF5:
            forcedPaTables = convert(odict, PA_TABLE)
            write_tables_to_HDF5(forcedPaTables, filepath)
            return filepath
        if fType == PANDAS_CSV:
            forcedOdict = convert(odict, PD_DATAFRAME)
            write_dataframes_to_csv(forcedOdict, filepath)
            return filepath
    except Exception as e:
        raise RuntimeError(
            f"Failed to write table to '{filepath}' as {FILE_FORMATS[fType]} because of error: \n {e}"
        ) from e

    raise TypeError(
        f"Unsupported File type {fType}. Supported types are: {list(FILE_FORMATS.values())}"
    )  # pragma: no cover


def write_native(odict, filepath: str) -> Optional[str]:
    """Writes `Table-like` or `TableDict-like` objects to a file or files. The file type will be determined
    by the default file type given the tabular format. The supported file types are:
    astropyHDF5 (".hf5"), numpyHDF5 (".hdf5"), numpyFits (".fit"), pandasParquet (".parq"), pyarrowParquet (".parquet").

    To write to a specific file format, use :py:func:`write` instead.

    Note: This function will automatically overwrite any previously existing files at the given filepath.

    Parameters
    ----------
    odict : `Table-like` or `TableDict-like`
        The data to write
    filepath : `str`
        File name for the file to write. If there's no suffix, it will be applied based on the object type.

    Returns
    -------
    filepath: `str` or None
        Returns the path to the new file, or None if there was no data given.

    Example
    -------


        >>> import tables_io
        >>> from astropy.table import Table
        >>> tab = Table([[1,3,5],[10,8,6]], names=('c1','c2'))
        >>> tables_io.write(tab, 'data') # writes the file to ASTROPY_HDF5 by default
        'data.hf5'

    """
    istable = False
    if is_table_like(odict):
        istable = True
        tType = table_type(odict)
    elif is_tabledict_like(odict):
        tType = table_type(list(odict.values())[0])
    elif not odict:  # pragma: no cover
        return None
    else:  # pragma: no cover
        raise TypeError(f"Cannot write object of type {type(odict)}")

    try:
        fType = NATIVE_FORMAT[tType]
    except KeyError as msg:  # pragma: no cover
        raise KeyError(f"No native format for table type {tType}") from msg
    fmt = FILE_FORMAT_SUFFIX_MAP[fType]
    if not os.path.splitext(filepath)[1]:
        filepath = filepath + "." + fmt

    if istable:
        odict = OrderedDict([(DEFAULT_TABLE_KEY[fmt], odict)])

    try:
        os.unlink(filepath)
    except FileNotFoundError:
        pass

    if fType == ASTROPY_HDF5:
        write_ap_tables_to_HDF5(odict, filepath)
        return filepath
    if fType == NUMPY_HDF5:
        write_dicts_to_HDF5(odict, filepath)
        return filepath
    if fType == NUMPY_FITS:
        write_recarrays_to_fits(odict, filepath)
        return filepath
    if fType == PANDAS_PARQUET:
        write_dataframes_to_pq(odict, filepath)
        return filepath
    if fType == PYARROW_PARQUET:
        write_tables_to_pq(odict, filepath)
        return filepath
    if fType == PANDAS_CSV:
        write_dataframes_to_csv(odict, filepath)
        return filepath
    raise TypeError(
        f"Unsupported Native file type {fType}. Must be one of ['astropyHdf5','numpyHdf5','numpyFits','pandaParquet','pyarrowParquet']"
    )  # pragma: no cover


# II. Writing Files

# II A. Writing HDF5 files


def initialize_HDF5_write_single(
    filepath: str, groupname: Optional[str] = None, comm=None, **kwds
):
    """Prepares an HDF5 file for output, where the file will be have datasets in only one group.
    The keywords (`**kwds`) argument(s) are required. They provide the data structure of the file.
    The name of each keyword argument provides the name of the dataset, and the value of the argument should
    be a dictionary with the dataset information (see below for details).

    The function will run in series if no MPI communicator (`comm`) is provided. To write the file
    in parallel, the MPI communicator argument is required.

    Parameters
    ----------
    filepath : `str`
        The output file name
    groupname : `str` or `None`
        The output group name
    comm: `communicator`
        MPI communicator to do parallel writing
    **kwds: one or more dictionaries
        Each keyword should provide a tuple of ( (shape), (dtype) )

        shape : `tuple` ( `int` )
            The shape of the data for this dataset
        dtype : `str`
            The data type for this dataset

    Returns
    -------
    group : `h5py.File` or `h5py.Group`
        The group to write to.
    fout : `h5py.File`
        The output file

    Example
    -------

    To initialize an HDF5 file with two datasets with different shapes:

        >>> from tables_io import hdf5
        >>> data = dict(scalar=((100000,), 'f4'), vect=((100000, 3), 'f4')
        >>> group, fout = hdf5.initialize_HDF5_write_single('test.hdf5',data=data)
        >>> print(group.name))
        '/data'


    To do the same in parallel with MPI using `mpi4py`:

        >>> from tables_io import hdf5
        >>> from mpi4py import MPI
        >>> data = dict(scalar=((100000,), 'f4'), vect=((100000, 3), 'f4')
        >>> fout = hdf5.initialize_HDF5_write_single('test.hdf5',comm=MPI.COMM_WORLD, data=data)


    """
    outdir = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(outdir):  # pragma: no cover
        os.makedirs(outdir, exist_ok=True)
    if comm is None:
        outf = h5py.File(filepath, "w")
    else:
        if not h5py.get_config().mpi:
            raise TypeError(
                "hdf5py module not prepared for parallel writing."
            )  # pragma: no cover
        outf = h5py.File(filepath, "w", driver="mpio", comm=comm)
    if groupname is None:
        group = outf
    else:
        group = outf.create_group(groupname)
    for key, shape in kwds.items():
        group.create_dataset(key, shape[0], shape[1])
    return group, outf


def initialize_HDF5_write(filepath: str, comm=None, **kwds):
    """Prepares an HDF5 file for output, where the file will be split up into one or more groups.
    The keywords (`**kwds`) argument(s) are required. They provide the data structure of the file.
    The name of each keyword argument provides the group name, and the value of the argument should
    be a dictionary with dataset name and information (see below for details).

    The function will run in series if no MPI communicator (`comm`) is provided. To write the file
    in parallel, the MPI communicator argument is required.

    Parameters
    ----------
    filepath : `str`
        The output file name
    comm: `communicator`
        MPI communicator to do parallel writing
    kwds: one or more `dict` arguments
        Each keyword should provide a dictionary with the group name and data set information
        of the form:
        ``group = {'data1' : ( (shape1), (dtype1) ), 'data2' : ( (shape2), (dtype2) )}``

        group : `str`
            Name of the Hdf5 group
        data  : `str`
            Name of the column to be written
        shape : `tuple` ( `int` )
            The shape of the data for this dataset
        dtype : `str`
            The data type for this dataset

    Returns
    -------
    group : `dict` of `h5py.File` or `h5py.Group`
        A dictionary of the groups to write to.
    fout : `h5py.File`
        The output file


    Example
    -------

    To initialize an HDF5 file with two groups named `group1` and `group2`:


        >>> from tables_io import hdf5
        >>> group1 = {'data1' : ((10,), 'f8'), 'data2': ((50,2), 'f8')}
        >>> group2 = {'data3': ((20,20), 'f8)}
        >>> groups, fout = hdf5.initializeHdf5Write('test.hdf5', group1=group1, group2=group2)


    To do the same in parallel with MPI using `mpi4py`:

        >>> from tables_io import hdf5
        >>> from mpi4py import MPI
        >>> group1 = {'data1' : ((10,), 'f8'), 'data2': ((50,2), 'f8')}
        >>> group2 = {'data3': ((20,20), 'f8)}
        >>> fout = hdf5.initialize_HDF5_write('test.hdf5',comm=MPI.COMM_WORLD, group1=group1, group2=group2)
    """
    outdir = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(outdir):  # pragma: no cover
        os.makedirs(outdir, exist_ok=True)
    if comm is None:
        outf = h5py.File(filepath, "w")
    else:
        if not h5py.get_config().mpi:
            raise TypeError(
                "hdf5py module not prepared for parallel writing."
            )  # pragma: no cover
        outf = h5py.File(filepath, "w", driver="mpio", comm=comm)
    groups = {}
    for k, v in kwds.items():
        group = outf.create_group(k)
        groups[k] = group
        for key, shape in v.items():
            group.create_dataset(key, shape[0], shape[1])
    return groups, outf


def write_dict_to_HDF5_chunk_single(fout, odict: Mapping, start: int, end: int, **kwds):
    """Writes a data chunk from a `Table-like` object to an hdf5 file

    Parameters
    ----------
    fout : `h5py.File`
        The file

    odict : `OrderedDict`, (`str`, `numpy.array`)
        The data being written

    start : `int`
        Starting row number to place the data

    end : `int`
        Ending row number to place the data

    Notes
    -----
    The kwds can be used to control the output locations, i.e., to
    rename the columns in data_dict when they go into the output file.

    For each item in data_dict, the output location is set as

    ``k_out = kwds.get(key, key)``

    This will check the kwds to see if they contain `key` and if so, will
    return the corresponding value.  Otherwise it will just return `key`.

    I.e., if `key` is present in kwds in will override the name.
    """
    for key, val in odict.items():
        k_out = kwds.get(key, key)
        fout[k_out][start:end] = val


def write_dict_to_HDF5_chunk(groups, odict: Mapping, start: int, end: int, **kwds):
    """Writes a data chunk from an `OrderedDict` or `TableDict-like` object to an hdf5 file in groups.

    Parameters
    ----------
    groups : `h5py.Group`
        The h5py groups or file object (which is also a group object)

    odict : `OrderedDict`, (`str`, `OrderedDict`(`str`, `numpy.array`))
        The data being written

    start : `int`
        Starting row number to place the data

    end : `int`
        Ending row number to place the data

    Notes
    -----
    The kwds can be used to control the output locations, i.e., to
    rename the columns in the input data when they go into the output file.
    The format of `kwds` should be `old_key = new_key`, where `old_key` is the
    key to be replaced by `new_key`.

    For each item in the input data, the output location is set as

    `k_out = kwds.get(key, key)`

    This will check the kwds to see if they contain `key` and if so, will
    return the corresponding value.  Otherwise it will just return `key`.

    I.e., if `key` is present in kwds it will override the name.
    """
    for group_name, group in groups.items():
        for key, val in odict[group_name].items():
            k_out = kwds.get(key, key)
            group[k_out][start:end] = val


def finalize_HDF5_write(fout, groupname: Optional[str] = None, **kwds):
    """Writes any last data given as keyword arguments, and closes an hdf5 file.
    If `groupname` is given, will create a group with that name before writing the data.
    If not, no new group will be created.

    Parameters
    ----------
    fout : `h5py.File`
        The file

    groupname: None or `str`
        The name to give the group. If None, no group will be created.

    Notes
    -----
    The keywords can be used to write additional data, where `key` is the name of the dataset and `value` should be the dataset to write.
    """
    if groupname is None:  # pragma: no cover
        group = fout
    else:
        group = fout.create_group(groupname)
    for k, v in kwds.items():
        group[k] = v
    fout.close()


# II B. Writing `astropy.table.Table`


def write_ap_tables_to_fits(tables: Mapping, filepath: str, **kwargs):
    """
    Writes a dictionary of `astropy.table.Table` to a single FITS file

    Parameters
    ----------
    tables : `dict` of `astropy.table.Table`
        Keys will be HDU names, values will be tables
    filepath: `str`
        Path to output file
    kwargs:
        kwargs are passed to `astropy.io.fits.writeto` call.
    """
    out_list = [fits.PrimaryHDU()]
    for k, v in tables.items():
        hdu = fits.table_to_hdu(v)
        hdu.name = k
        out_list.append(hdu)
    hdu_list = fits.HDUList(out_list)
    hdu_list.writeto(filepath, **kwargs)


def write_ap_tables_to_HDF5(tables: Mapping, filepath: str, **kwargs):
    """
    Writes a dictionary of `astropy.table.Table` to a single hdf5 file

    Parameters
    ----------
    tables : `dict` of `astropy.table.Table`
        Keys will be passed to 'path' parameter

    filepath: `str`
        Path to output file
    kwargs:
        kwargs are passed to `astropy.table.Table` call.
    """
    for k, v in tables.items():
        v.write(filepath, path=k, append=True, format="hdf5", **kwargs)


# II C. Writing `numpy.recarray`


def write_recarrays_to_fits(recarrays: Mapping, filepath: str, **kwargs):
    """
    Writes a dictionary of `np.recarray` to a single FITS file

    Parameters
    ----------
    recarrays  : `dict` of `np.recarray`
        Keys will be HDU names, values will be tables

    filepath: `str`
        Path to output file

    kwargs:
        kwargs are passed to `astropy.io.fits.writeto` call.
    """
    out_list = [fits.PrimaryHDU()]
    for k, v in recarrays.items():
        hdu = fits.BinTableHDU.from_columns(v.columns)
        hdu.name = k
        out_list.append(hdu)
    hdu_list = fits.HDUList(out_list)
    hdu_list.writeto(filepath, **kwargs)


# II D. Writing `OrderedDict`, (`str`, `numpy.array`) to `hdf5`


def write_dict_to_HDF5(
    odict: Mapping, filepath: str, groupname: Optional[str], **kwargs
):
    """
    Writes a dictionary of `numpy.array` or `jaxlib.xla_extension.DeviceArray`
    to a single hdf5 file

    Parameters
    ----------
    odict : `Mapping`, (`str`, `numpy.array` or `jaxlib.xla_extension.DeviceArray`)
        The data being written

    filepath: `str`
        Path to output file

    groupname : `str` or `None`
        The groupname for the data
    """
    # pylint: disable=unused-argument
    fout = h5py.File(filepath, "a")
    if groupname is None:  # pragma: no cover
        group = fout
    else:
        group = fout.require_group(groupname)
    for key, val in odict.items():
        try:
            if isinstance(val, np.ndarray):
                group.create_dataset(key, dtype=val.dtype, data=val.data)
            elif isinstance(val, list):
                arr = np.array(val)
                group.create_dataset(key, dtype=arr.dtype, data=arr)
            elif isinstance(val, dict):
                write_dict_to_HDF5(val, filepath, f"{group.name}/{key}")
            else:
                # In the future, it may be better to specifically case for
                # jaxlib.xla_extension.DeviceArray here. For now, we're
                # resorting to duck typing so we don't have to import jax just
                # to check the type.
                group.create_dataset(key, dtype=val.dtype, data=val.addressable_data(0))
        except Exception as msg:  # pragma: no cover
            print(f"Warning.  Failed to convert column {str(msg)}")
    fout.close()


def write_dicts_to_HDF5(odicts: Mapping, filepath: str):
    """
    Writes a `TableDict-like` object, a `OrderedDict` of dictionaries of `numpy.array`, to a single hdf5 file.

    Note: This will remove any previously existing files at the filepath.

    Parameters
    ----------
    odicts : `OrderedDict`, (`str`, `Table-like`)
        The data being written

    filepath: `str`
        Path to output file
    """
    try:
        os.unlink(filepath)
    except FileNotFoundError:
        pass
    for key, val in odicts.items():
        write_dict_to_HDF5(val, filepath, key)


# II E. Writing `pandas.DataFrame`


def write_dataframes_to_HDF5(dataFrames: Mapping, filepath: str):
    """
    Writes a dictionary of `pandas.DataFrame` to a single hdf5 file

    Parameters
    ----------
    dataFrames : `dict` of `pandas.DataFrame`
        Keys will be passed to 'key' parameter

    filepath: `str`
        Path to output file
    """
    for key, val in dataFrames.items():
        val.to_hdf(filepath, key=key)


def write_dataframes_to_pq(dataFrames: Mapping, filepath: str, **kwargs):
    """
    Writes a dictionary of `pandas.DataFrame` to parquet files

    Parameters
    ----------
    tables : `dict` of `pandas.DataFrame`
        Keys will be passed to 'path' parameter

    filepath: `str`
        Path to output file

    """
    basepath, ext = os.path.splitext(filepath)
    if not ext:  # pragma: no cover
        ext = "." + FILE_FORMAT_SUFFIX_MAP[PANDAS_PARQUET]
    for k, v in dataFrames.items():
        _ = v.to_parquet(f"{basepath}{k}{ext}", **kwargs)


def write_dataframes_to_csv(dataFrames: Mapping, filepath: str, **kwargs):
    """
    Writes a dictionary of `pandas.DataFrame` to csv files

    Parameters
    ----------
    tables : `dict` of `pandas.DataFrame`
        Keys will be passed to 'path' parameter

    filepath: `str`
        Path to output file

    """

    if "index" not in kwargs:  # don't write out the index by default
        kwargs["index"] = False

    basepath, ext = os.path.splitext(filepath)
    if not ext:  # pragma: no cover
        ext = "." + FILE_FORMAT_SUFFIX_MAP[PANDAS_CSV]

    for k, v in dataFrames.items():
        _ = v.to_csv(f"{basepath}{k}{ext}", **kwargs)


# II F. Writing pyarrow table(s)


def write_table_to_HDF5(table, filepath: str, key: str):
    """
    Writes a `pyarrow.Table` to a single hdf5 file

    Parameters
    ----------
    table : `dict` of `pyarrow.Table`
        Keys will be passed to 'key' parameter

    filepath: `str`
        Path to output file

    key: `str`
        The hdf5 groupname
    """
    write_dict_to_HDF5(table.to_pydict(), filepath, key)


def write_tables_to_HDF5(tables: Mapping, filepath: str):
    """
    Writes a dictionary of `pyarrow.Table` to a single hdf5 file

    Parameters
    ----------
    tables : `dict` of `pyarrow.Table`
        Keys will be passed to 'key' parameter

    filepath: `str`
        Path to output file
    """
    for key, val in tables.items():
        write_table_to_HDF5(val, filepath, key)


def write_tables_to_pq(tables: Mapping, filepath: str, **kwargs):
    """
    Writes a dictionary of `pyarrow.Table` to parquet files. If no extension is
    given in the base path, it will be written as a `.parq` file.

    Parameters
    ----------
    tables : `dict` of `pyarrow.Table`
        Keys will be passed to 'path' parameter

    filepath: `str`
        Path to output file

    """
    basepath, ext = os.path.splitext(filepath)
    if not ext:  # pragma: no cover
        ext = "." + FILE_FORMAT_SUFFIX_MAP[PANDAS_PARQUET]
    for k, v in tables.items():
        pq.write_table(v, f"{basepath}{k}{ext}", **kwargs)
