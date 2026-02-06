"""Array-related utility functions for tables_io"""

from collections import OrderedDict

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Mapping, Union, List


def array_length(arr: ArrayLike) -> int:
    """Get the length of an array

    The works on scalars and arrays, so it is safe to use

    For scalars it returns 0
    For arrays it np.shape(arr)[0]

    Parameters
    ----------
    arr : `array-like`
        The input array

    Returns
    -------
    length : `int`
        The object length
    """
    shape = np.shape(arr)
    if not shape:
        return 0
    return shape[0]


def force_to_pandables(arr: ArrayLike, check_nrow: Optional[int] = None):
    """
    Forces a  `numpy.array` into a format that pandas can handle

    Parameters
    ----------
    arr : `numpy.array`
        The input array

    check_nrow : `int` or `None`
        If not None, require that `arr.shape[0]` match this value

    Returns
    -------
    out : `numpy.array` or `list` of `numpy.array`
        Something that pandas can handle
    """
    ndim = np.ndim(arr)
    if ndim == 0:
        return arr
    shape = np.shape(arr)
    nrow = shape[0]
    if check_nrow is not None and check_nrow != nrow:
        raise ValueError(f"Number of rows does not match: {nrow} != {check_nrow}")
    if ndim == 1:
        return arr
    if ndim == 2:
        return list(arr)
    shape = np.shape(arr)
    ncol = np.prod(shape[1:])
    return list(arr.reshape(nrow, ncol))


def get_group_input_data_length(hg) -> int:
    """Return the length of a HDF5 group

    Parameters
    ----------
    hg : `h5py.Group` or `h5py.File`
        The input data group

    Returns
    -------
    length : `int`
        The length of the data

    Notes
    -----
    For a multi-D array this return the length of the first axis
    and not the total size of the array.

    Normally that is what you want to be iterating over.

    The group is meant to represent a table, hence all child datasets
    should be the same length
    """
    firstkey = list(hg.keys())[0]
    nrows = len(hg[firstkey])
    firstname = hg[firstkey].name
    for value in hg.values():
        if len(value) != nrows:
            raise ValueError(
                f"Group does not represent a table. Length ({len(value)})"
                f"of column {value.name} not not match length ({nrows}) of"
                f"first column {firstname}"
            )
    return nrows


def get_initialization_for_ODict(
    in_dict: Mapping, nRow_out: Optional[int] = None
) -> Mapping:
    """shape of arrays and dtypes in a dictionary.
    This is useful for initialize hdf5 files

    Parameters
    ----------
    in_dict : `dict`
        The dictionary with the data
    nRow_out : `int`
        The number of rows in the output

    Returns
    -------
    out_dict : dict (str, tuple(tuple, dtype))
    """
    out_dict = {}
    for key, val in in_dict.items():
        shape = list(np.shape(val))
        if nRow_out is not None:
            shape[0] = nRow_out
        out_dict[key] = (tuple(shape), val.dtype)
    return out_dict


def print_dict_shape(in_dict: Mapping):
    """Print the shape of arrays in a dictionary.
    This is useful for debugging `astropy.Table` creation.

    Parameters
    ----------
    in_dict : `dict`
        The dictionary to print
    """
    for key, val in in_dict.items():
        print(key, np.shape(val))


def slice_dict(in_dict: Mapping, subslice: Union[int, slice]) -> Mapping:
    """Create a new `dict` by taking a slice of of every array in a `dict`

    Parameters
    ----------
    in_dict : `dict`
        The dictionary to extract from
    subslice : `int` or `slice`
        Used to slice the arrays

    Returns
    -------
    out_dict : `dict`
        The converted dictionary
    """

    out_dict = OrderedDict()
    for key, val in in_dict.items():
        try:
            out_dict[key] = val[subslice]
        except (KeyError, TypeError):
            out_dict[key] = val
    return out_dict


def check_keys(in_dicts: List[Mapping]):
    """Check that the keys in all the in_dicts match

    Parameters
    ----------
    in_dicts : `list`, (`OrderedDict`, (`str`, `numpy.array`))
        The dictionaries for which compare keys

    Raises
    ------
    KeyError
        if keys do not match
    """
    if not in_dicts:
        return
    master_keys = in_dicts[0].keys()
    for in_dict in in_dicts[1:]:
        if in_dict.keys() != master_keys:
            raise ValueError(
                f"Keys do not match: {list(in_dict.keys())} != {list(master_keys)}"
            )


def concatenate_dicts(in_dicts: List[Mapping]) -> Mapping:
    """Create a new `dict` by concatenating each array in `in_dicts`

    Parameters
    ----------
    in_dicts : `list`, (`OrderedDict`, (`str`, `numpy.array`))
        The dictionaries to stack

    Returns
    -------
    out_dict : `dict`
        The stacked dictionary
    """
    if not in_dicts:
        return OrderedDict()
    check_keys(in_dicts)
    out_dict = OrderedDict([(key, None) for key in in_dicts[0].keys()])
    for key in out_dict.keys():
        out_dict[key] = np.concatenate([in_dict[key] for in_dict in in_dicts])
    return out_dict
