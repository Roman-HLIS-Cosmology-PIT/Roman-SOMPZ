"""
Unit tests for io_layer module

Future refactor should migrate this to pytest.
"""

import os
import pytest

import unittest
from tables_io import types, convert, io_open, read, write, iterator, check_columns
from tests.helpers.utilities import (
    compare_table_dicts,
    compare_tables,
    make_test_data,
    check_deps,
)
from tables_io.lazy_modules import apTable, jnp, h5py, pd, pq


@pytest.mark.skipif(
    not check_deps([apTable, h5py, pq, jnp]), reason="Missing an IO package"
)
class IoTestCase(unittest.TestCase):  # pylint: disable=too-many-instance-attributes
    """Test the utility functions"""

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self._tables = make_test_data()
        self._table = self._tables["data"]
        self._files = []

    def tearDown(self):
        """Clean up any mock data files created by the tests."""
        for ff in self._files:
            if os.path.exists(ff):
                os.unlink(ff)

    def _do_loopback(self, tType, basepath, fmt, **kwargs):
        """Utility function to do loopback tests"""

        odict_c = convert(self._tables, tType)
        filepath = write(odict_c, basepath, fmt)
        self._files.append(filepath)
        odict_r = read(filepath, tType=tType, fmt=fmt, **kwargs)
        tables_r = convert(odict_r, types.AP_TABLE)

        # Testing to raise errors on read (i.e., trying with invalid key)

        with pytest.raises(RuntimeError) as e:
            obj_r = read(filepath, tType=tType, fmt=fmt, keys=100)

        return compare_table_dicts(self._tables, tables_r, **kwargs)

    def _do_loopback_with_keys(
        self, tType, basepath, fmt, keys, columns=None, **kwargs
    ):
        """Utility function to do loopback tests in formats that require keyed files"""
        odict_c = convert(self._tables, tType)
        filepath = write(odict_c, basepath, fmt)
        expected_tables = {}
        self._files.append("%s.%s" % (basepath, fmt))
        for key in keys:
            self._files.append("%s%s.%s" % (basepath, key, fmt))
            expected_tables[key] = self._tables[key]
        odict_r = read(
            filepath, tType=tType, fmt=fmt, keys=keys, columns=columns, **kwargs
        )
        tables_r = convert(odict_r, types.AP_TABLE)
        column_list = None
        if pd.api.types.is_dict_like(columns):
            for key in columns.keys():
                if pd.api.types.is_dict_like(tables_r):
                    assert compare_tables(
                        expected_tables[key],
                        tables_r[key],
                        columns=columns[key],
                        **kwargs,
                    )
                else:
                    assert compare_tables(
                        list(expected_tables.values())[0],
                        tables_r,
                        columns=columns[key],
                        **kwargs,
                    )
        else:
            if pd.api.types.is_dict_like(tables_r):
                return compare_table_dicts(
                    expected_tables, tables_r, columns=columns, **kwargs
                )
            else:
                return compare_tables(
                    list(expected_tables.values())[0],
                    tables_r,
                    columns=columns,
                    **kwargs,
                )

        # Testing to raise errors on read (i.e., trying with invalid key)

        with pytest.raises(RuntimeError) as e:
            obj_r = read(filepath, tType=tType, fmt=fmt, keys=100)

    def _do_loopback_jax(self, basepath, fmt, keys=None, **kwargs):
        """Utility function to do loopback tests writing data as a jax array"""
        odict_c = convert(self._tables, types.NUMPY_DICT)
        for key, val in odict_c["data"].items():
            odict_c["data"][key] = jnp.array(val)
        filepath = write(odict_c, basepath, fmt)
        if keys is None:
            self._files.append(filepath)
        else:
            for key in keys:
                self._files.append("%s%s.%s" % (basepath, key, fmt))
        odict_r = read(filepath, tType=types.NUMPY_DICT, fmt=fmt, keys=keys, **kwargs)
        tables_r = convert(odict_r, types.AP_TABLE)
        assert compare_table_dicts(self._tables, tables_r, **kwargs)

    def _do_loopback_single(self, tType, basepath, fmt, keys=None, **kwargs):
        """Utility function to do loopback tests"""
        obj_c = convert(self._table, tType)
        filepath = write(obj_c, basepath, fmt)
        try:
            os.unlink(filepath)
        except FileNotFoundError:
            pass
        filepath_check = write(obj_c, filepath, None)
        assert filepath == filepath_check
        if keys is None:
            self._files.append(filepath)
        else:
            for key in keys:
                self._files.append("%s%s.%s" % (basepath, key, fmt))
        obj_r = read(filepath, tType=tType, fmt=fmt, keys=keys, **kwargs)
        table_r = convert(obj_r, types.AP_TABLE)
        assert compare_tables(self._table, table_r, **kwargs)

        if types.FILE_FORMAT_SUFFIXS[fmt] not in types.NATIVE_TABLE_TYPE:
            return

        basepath_native = "%s_native" % basepath
        filepath_native = write(obj_c, basepath_native)
        self._files.append(filepath_native)
        obj_r_native = read(filepath_native, tType=tType, keys=keys)
        table_r_native = convert(obj_r_native, types.AP_TABLE)
        assert compare_tables(self._table, table_r_native, **kwargs)

        # Testing to raise errors on read (i.e., trying with invalid key)

        with pytest.raises(RuntimeError) as e:
            obj_r = read(filepath, tType=tType, fmt=fmt, keys=100)

    def _do_iterator(self, filepath, tType, expectFail=False, chunk_size=50, **kwargs):
        """Test the chunk iterator"""
        dl = []
        try:
            for _, _, data in iterator(
                filepath, tType=tType, chunk_size=chunk_size, **kwargs
            ):
                dl.append(convert(data, types.AP_TABLE))
            table_iterated = apTable.vstack(dl)
            assert compare_tables(self._table, table_iterated, **kwargs)
        except NotImplementedError as msg:
            if expectFail:
                pass
            else:
                raise msg

    def _do_open(self, filepath, in_context=True):
        """ """
        if not in_context:
            f = io_open(filepath)
            assert f
            return
        with io_open(filepath) as f:
            assert f

    def _do_check_columns(self, filepath):
        columns_to_check = ["imaginary_column"]
        try:
            check_columns(filepath, columns_to_check)
        except KeyError:
            pass

    def testFitsLoopback(self):
        """Test writing / reading to FITS"""
        self._do_loopback(types.AP_TABLE, "test_out", "fits")
        self._do_loopback_single(types.AP_TABLE, "test_out_single", "fits")
        self._do_loopback_with_keys(types.AP_TABLE, "test_out_lookback", "fits", ["md"])
        self._do_iterator("test_out_single.fits", types.AP_TABLE, True)
        self._do_open("test_out_single.fits")
        self._do_open("test_out.fits")
        self._do_check_columns("test_out.fits")

    def testRecarrayLoopback(self):
        """Test writing / reading to FITS"""
        self._do_loopback(types.NUMPY_RECARRAY, "test_out", "fit")
        self._do_loopback_single(types.NUMPY_RECARRAY, "test_out_single", "fit")
        self._do_loopback_with_keys(
            types.NUMPY_RECARRAY, "test_out_lookback", "fit", ["md"]
        )
        self._do_iterator("test_out_single.fit", types.NUMPY_RECARRAY, True)
        self._do_open("test_out_single.fit")
        self._do_open("test_out.fit")
        self._do_check_columns("test_out.fit")

    def testHf5Loopback(self):
        """Test writing / reading astropy tables to HDF5"""
        self._do_loopback(types.AP_TABLE, "test_out", "hf5")
        self._do_loopback_single(types.AP_TABLE, "test_out_single", "hf5")
        self._do_loopback_with_keys(types.AP_TABLE, "test_out_lookback", "hf5", ["md"])
        self._do_iterator("test_out_single.hf5", types.AP_TABLE, True, chunk_size=50)
        self._do_open("test_out_single.hf5")
        self._do_open("test_out.hf5")
        self._do_check_columns("test_out.hf5")

    def testHdf5Loopback(self):
        """Test writing / reading numpy arrays to HDF5"""
        self._do_loopback(types.NUMPY_DICT, "test_out", "hdf5")
        self._do_loopback_single(types.NUMPY_DICT, "test_out_single", "hdf5")
        self._do_loopback_with_keys(
            types.NUMPY_DICT, "test_out_lookback", "hdf5", ["md"]
        )
        self._do_iterator(
            "test_out_single.hdf5", types.NUMPY_DICT, False, chunk_size=50
        )
        self._do_open("test_out_single.hdf5")
        self._do_open("test_out.hdf5")
        self._do_check_columns("test_out.hdf5")

    def testHd5Loopback(self):
        """Test pyarrow tables to HDF5"""
        self._do_loopback(types.PA_TABLE, "test_out", "hd5")
        self._do_loopback_single(types.PA_TABLE, "test_out_single", "hd5")
        self._do_loopback_with_keys(types.PA_TABLE, "test_out_lookback", "hdf5", ["md"])
        # self._do_iterator("test_out_single.hd5", types.PA_TABLE, False, chunk_size=50)
        self._do_open("test_out_single.hd5")
        self._do_open("test_out.hd5")
        self._do_check_columns("test_out.hd5")

    def testHdf5LoopbackWithJaxArray(self):
        """Test writing / reading astropy tables to HDF5 with a jax array"""
        self._do_loopback_jax("test_out", "hdf5")

    def testH5Loopback(self):
        """Test writing / reading pandas dataframes to HDF5"""
        self._do_loopback(types.PD_DATAFRAME, "test_out", "h5")
        self._do_loopback(types.PA_TABLE, "test_out_pa", "h5")
        self._do_loopback_single(types.PD_DATAFRAME, "test_out_single", "h5")
        self._do_loopback_with_keys(
            types.PD_DATAFRAME, "test_out_lookback", "h5", ["md"]
        )
        self._do_iterator("test_out_single.h5", types.PD_DATAFRAME, True, chunk_size=50)
        self._do_open("test_out_single.h5")
        self._do_open("test_out.h5")
        self._do_check_columns("test_out.h5")

    def testPQLoopback(self):
        """Test writing / reading pandas dataframes to parquet"""

        self._do_loopback_with_keys(
            types.PD_DATAFRAME, "test_out", "pq", list(self._tables.keys())
        )
        self._do_loopback_with_keys(types.PD_DATAFRAME, "test_out", "pq", ["data"])
        self._do_loopback_with_keys(types.PD_DATAFRAME, "test_out", "pq", ["md"])
        self._do_loopback_with_keys(
            types.PD_DATAFRAME, "test_out", "pq", ["data"], columns=["scalar"]
        )
        self._do_loopback_with_keys(
            types.PD_DATAFRAME, "test_out", "pq", ["md"], columns=["a"]
        )
        self._do_loopback_with_keys(
            types.PD_DATAFRAME,
            "test_out",
            "pq",
            ["data", "md"],
            columns={"md": ["a"], "data": ["scalar"]},
        )
        self._do_loopback_single(types.PD_DATAFRAME, "test_out_single", "pq", [""])
        self._do_loopback_single(
            types.PD_DATAFRAME, "test_out_single_v2", "parquet", [""]
        )

        self._do_iterator("test_out_single.pq", types.PD_DATAFRAME, chunk_size=50)
        self._do_iterator(
            "test_out_single.pq", types.PD_DATAFRAME, chunk_size=50, columns=["scalar"]
        )
        self._do_open("test_out_single.pq")
        self._do_check_columns("test_out_single.pq")

    def testParquetLoopback(self):
        """Test writing / reading pyarros tables to parquet files"""
        self._do_loopback_with_keys(
            types.PA_TABLE, "test_out", "parquet", ["data", "md"]
        )
        self._do_loopback_single(types.PA_TABLE, "test_out_single", "parquet", [""])
        self._do_iterator(
            "test_out_single.parquet", types.PA_TABLE, True, chunk_size=50
        )
        self._do_open("test_out_single.parquet")
        self._do_check_columns("test_out_single.parquet")

    def testCsvLoopback(self):
        """Test writing / reading to csv files"""
        self._do_loopback_with_keys(types.AP_TABLE, "test_out", "csv", ["data", "md"])
        self._do_loopback_single(types.AP_TABLE, "test_out_single", "csv", [""])
        self._do_iterator("test_out_single.csv", types.AP_TABLE, True, chunk_size=50)
        self._do_open("test_out_single.csv", in_context=False)
        self._do_check_columns("test_out_single.csv")

    def testBad(self):  # pylint: disable=no-self-use
        """Test that bad calls to write are dealt with"""
        try:
            write("aa", "null", "fits")
        except TypeError:
            pass
        else:
            raise TypeError("Failed to catch unwritable type")
        assert write(False, "null", "fits") is None


if __name__ == "__main__":
    unittest.main()
