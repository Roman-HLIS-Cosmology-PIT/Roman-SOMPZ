""" Unit tests for the fileIO module """

import sys
import unittest

import numpy as np
import pytest

from tables_io.utils import array_utils
from tables_io.lazy_modules import apTable, lazyImport, pd
from tests.helpers.utilities import check_deps


def test_array_length():
    """Test the pandas reading"""
    assert array_utils.array_length(4) == 0
    assert array_utils.array_length(np.ones(5)) == 5
    assert array_utils.array_length(np.ones((5, 5, 5))) == 5
    assert array_utils.array_length([3, 4, 4]) == 3


def test_force_to_pandasable():
    """Test the force_to_pandasable function"""
    rv = np.random.uniform(size=100)
    assert array_utils.force_to_pandables(4) == 4
    assert np.allclose(array_utils.force_to_pandables(rv), rv)
    try:
        array_utils.force_to_pandables(rv, 95)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Failed to catch array length mismatch in arrayUtils.forceToPandables"
        )
    rv2d = rv.reshape(10, 10)
    rv2d_check = np.vstack(array_utils.force_to_pandables(rv2d))
    assert np.allclose(rv2d, rv2d_check)

    rv3d = rv.reshape(10, 2, 5)
    rv3d_check = np.vstack(array_utils.force_to_pandables(rv3d)).reshape(rv3d.shape)
    assert np.allclose(rv3d, rv3d_check)


def test_slice_dict():
    """Test the slice_dict method"""
    test_data = dict(
        scalar=np.random.uniform(size=10),
        vector=np.random.uniform(size=100).reshape(10, 10),
        mat=np.random.uniform(size=1000).reshape(10, 10, 10),
    )
    sliced = array_utils.slice_dict(test_data, 1)
    assert np.allclose(sliced["scalar"], test_data["scalar"][1])
    assert np.allclose(sliced["vector"], test_data["vector"][1])
    assert np.allclose(sliced["mat"], test_data["mat"][1])

    mask = np.zeros((10), bool)
    mask[1] = True
    mask[4] = True
    sliced = array_utils.slice_dict(test_data, mask)
    assert np.allclose(sliced["scalar"], test_data["scalar"][mask])
    assert np.allclose(sliced["vector"], test_data["vector"][mask])
    assert np.allclose(sliced["mat"], test_data["mat"][mask])

    # Testing for subslice errors:

    error_slice = array_utils.slice_dict({"temp": 3}, 5)
    assert error_slice["temp"] == 3


def test_print_dict_shape():
    """Test the print_dict_shape method"""
    test_data = dict(
        scalar=np.random.uniform(size=10),
        vector=np.random.uniform(size=100).reshape(10, 10),
        mat=np.random.uniform(size=1000).reshape(10, 10, 10),
    )
    array_utils.print_dict_shape(test_data)


def test_concatenate_dicts():
    """Test the print_dict_shape method"""
    test_data = dict(
        scalar=np.random.uniform(size=10),
        vector=np.random.uniform(size=100).reshape(10, 10),
        mat=np.random.uniform(size=1000).reshape(10, 10, 10),
    )
    od = array_utils.concatenate_dicts([test_data, test_data])
    assert np.allclose(od["scalar"][0:10], test_data["scalar"])
    assert np.allclose(od["vector"][0:10], test_data["vector"])
    assert np.allclose(od["mat"][0:10], test_data["mat"])
    assert np.allclose(od["scalar"][10:], test_data["scalar"])
    assert np.allclose(od["vector"][10:], test_data["vector"])
    assert np.allclose(od["mat"][10:], test_data["mat"])


def test_get_initialization_for_ODict():
    """Testing the initialization of the Ordered Dictionary"""
    test_data = dict(
        scalar=np.random.uniform(size=10),
        vector=np.random.uniform(size=100).reshape(10, 10),
        mat=np.random.uniform(size=1000).reshape(10, 10, 10),
    )

    dd = array_utils.get_initialization_for_ODict(test_data)
    assert dd["scalar"][0] == (10,)
    assert dd["vector"][0] == (10, 10)
    assert dd["mat"][0] == (10, 10, 10)
    dd = array_utils.get_initialization_for_ODict(test_data, 12)
    assert dd["scalar"][0] == (12,)
    assert dd["vector"][0] == (12, 10)
    assert dd["mat"][0] == (12, 10, 10)


def test_check_keys_errors():
    """Testing Check Keys Errors"""

    assert isinstance(array_utils.check_keys([]), type(None))

    # Ensure that it fails on keys that don't exist in non-0 dictionary
    with pytest.raises(ValueError):
        array_utils.check_keys([{"test_key": 4}, dict()])


def test_concatenate_dicts_errors():
    """Testing Against Errors in Concatenate Dicts"""

    new_dict = array_utils.concatenate_dicts([])

    assert len(new_dict) == 0
