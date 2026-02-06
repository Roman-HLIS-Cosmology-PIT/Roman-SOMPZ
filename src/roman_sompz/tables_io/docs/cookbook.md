# Cookbook

(basic-table-operations)=

## Basic table operations

### Read in a file to a specific format

To read in a `Table-like` or `TableDict-like` object from a file to a specific tabular type, use the {py:func}`read <tables_io.io_utils.read.read>` function. For example, to read in a Parquet file:

```{doctest}

>>> import tables_io
>>> tab = tables_io.read('filename.parquet')
>>> tab
pyarrow.Table
x: double
z: double
----
x: [[1,1.5,2]]
z: [[0.2,0.7,1.2]]

```

You can also supply the type via the `tType` argument, either as a string or as an integer. The allowed tabular types are listed in <project:./quickstart.md#supported-tabular-formats>.

For example, to read in tables from an HDF5 file as `astropyTable` objects:

```{doctest}

>>> import tables_io
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

```

### Write out a file to a specific file format

To write out a `Table-like` or `TableDict-like` object to a specific file format, use the {py:func}`write <tables_io.io_utils.write.write>` function.

In this example, we write out a `pandasDataFrame` object to an HDF5 file in two different ways. The first is by using the `fmt` argument to specify the file type. The second is using the suffix of the file name.

```{doctest}

>>> import tables_io
>>> import pandas as pd
>>> tab = pd.DataFrame({'col1': [2,4,6], 'col2': [5,7,9]})
>>> tables_io.write(tab, 'data','h5')
'data.h5'
>>> tables_io.write(tab, 'data.h5')
'data.h5'

```

### Get a file object

To access a file object directly instead of just reading in the data tables, use {py:func}`io_open <tables_io.io_utils.read.io_open>`. In the case of HDF5 files, this allows you to get metadata from the file, such as the keys etc, without reading in all of the data:

```{doctest}

>>> import tables_io
>>> file = tables_io.io_open("./data/test.hdf5","hdf5")
>>> file.keys()
<KeysViewHDF5 ['id', 'mag_err_g_lsst', 'mag_err_i_lsst', 'mag_err_r_lsst', 'mag_err_u_lsst',
'mag_err_y_lsst', 'mag_err_z_lsst', 'mag_g_lsst', 'mag_i_lsst', 'mag_r_lsst', 'mag_u_lsst',
'mag_y_lsst', 'mag_z_lsst', 'redshift']>

```

This can be done with other file types as well, though the file object will behave differently depending
on the type of file you're opening. For example, for a FITS file:

```{doctest}

>>> import tables_io
>>> hdul = tables_io.io_open("./data/test.fits", "fits")
>>> hdul.info()
No.    Name      Ver    Type      Cards   Dimensions   Format
    0  PRIMARY       1 PrimaryHDU       4   ()
    1  DF            1 BinTableHDU     37   10R x 14C   [K, E, E, E, E, E, E, E, E, E, E, E, E, D]

```

### Converting tables to different formats

To convert a `Table-like` object, use the {py:func}`convert <tables_io.conv.conv_tabledict.convert>` function. Here we convert an `astropyTable` to a `pyarrowTable`.

```{doctest}

>>> import tables_io
>>> from astropy.table import Table
>>> tab = Table([[2,4,6],[3,6,9]],names=("x","y"))
>>> new_tab = tables_io.convert(tab, 'pyarrowTable')
pyarrow.Table
x: int64
y: int64
----
x: [[2,4,6]]
y: [[3,6,9]]

```

To convert `TableDict-like` objects, you use the same function. Here we convert an `OrderedDict` of `pandasDataFrame` objects to one of `numpyDict` objects.

```{doctest}

>>> import tables_io
>>> import pandas as pd
>>> # setting up the TableDict-like object
>>> pdict = OrderedDict()
>>> pdict['tab_1'] = pd.DataFrame({'col_1': [5,4,3], 'col_2': [3,6,9]})
>>> pdict['tab_2'] = pd.DataFrame({'col_a': [9,10], 'col_b': [11,12]})
>>> tables_io.convert(pdict,'numpyDict')
OrderedDict([('tab_1',
              OrderedDict([('col_1', array([5, 4, 3])),
                           ('col_2', array([3, 6, 9]))])),
             ('tab_2',
              OrderedDict([('col_a', array([ 9, 10])),
                           ('col_b', array([11, 12]))]))])

```

### Concatenating Table-like objects

To concatenate two or more `Table-like` objects, you use the {py:func}`concat_table <tables_io.utils.concat_utils.concat_table>` function. This function can only concatenate objects that have the same tabular type. So for example, they must both be `pandasDataFrame` objects, as in the example below. The tables must be passed as a collection of items (i.e. list, tuple), and it requires the tabular type of the tables to be concatenated as an argument.

```{doctest}

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

```

### Concatenating TableDict-like objects

To concatenate two or more `TableDict-like` objects, use the {py:func}`concat_tabledict <tables_io.utils.concat_utils.concat_tabledict>` function. Similar to the {py:func}`concat_table <tables_io.utils.concat_utils.concat_table>` function, the `TableDict-like` objects must have the same tabular type to be concatenated. They also must be passed in as a list, along with the tabular type of the `TableDict-like` objects.

```{doctest}

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

```

### Slicing a Table-like object

To get a slice from a `Table-like` object, use the {py:func}`slice_table <tables_io.utils.slice_utils.slice_table>` function as shown here:

```{doctest}

>>> import tables_io
>>> import pandas as pd
>>> df = pd.DataFrame({'col1': [1,2,3], 'col2':[3,4,5]})
>>> tables_io.slice_table(df, slice(1,2))
    col1  col2
1     2     4

```

Here we used a Python [slice(start, stop, step)](https://docs.python.org/3/library/functions.html#slice) object to identify the slice. This is the recommended format to use for slicing `Table-like` objects, though you can use an integer as well.

```{warning}
The slice you get with an integer is not consistent across different table types. Most table types will return the **row** with that index, but a `pyarrowTable` will return the **column** with that index.
```

### Slicing a TableDict-like object

To get a slice from all `Table-like` objects in a `TableDict-like` object, use the {py:func}`slice_tabledict <tables_io.utils.slice_utils.slice_tabledict>` function as shown below:

```{doctest}

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

```

As with {py:func}`slice_table <tables_io.utils.slice_utils.slice_table>`, the {py:func}`slice_tabledict <tables_io.utils.slice_utils.slice_tabledict>` function takes either an integer or the Python [slice(start, stop, step)](https://docs.python.org/3/library/functions.html#slice) object to identify the slice.

```{warning}
The slice you get with an integer is not consistent across different table types. Most table types will return the **row** with that index, but a `pyarrowTable` will return the **column** with that index.
```

## Handling HDF5 files

### Iterating through data in an HDF5 file

To iterate through an HDF5 file, yielding only a section of data at a time, we can use the {py:func}`iterator <tables_io.io_utils.iterator.iterator>` or {py:func}`iterator_native <tables_io.io_utils.iterator.iterator>` functions as shown below. You can provide the size of the data section you would like as an `int` to `chunk_size`, size here meaning the number of rows (or length of the `numpy.arrays` in the case of `numpyDict` tables). The default `chunk_size` is 100,000.

To determine the number of rows of data in the file, and therefore what an appropriate chunk size would be, you can use the {py:func}`get_input_data_length <tables_io.io_utils.iterator.get_input_data_length>` function from the `hdf5` module as follows:

```{doctest}

>>> from tables_io import hdf5
>>> hdf5.get_input_data_length('datafile.hdf5')
7

```

Here we did not supply the `groupname` of the data, since the default was appropriate.

Since the length of our file is 7, we will choose a chunk size smaller than that, say 3. To output the data to a `pandasDataFrame`, we use the {py:func}`iterator <tables_io.io_utils.iterator.iterator>` function as shown below:

```{doctest}

>>> import tables_io
>>> for start, stop, data in tables_io.iterator('datafile.hdf5','pandasDataFrame',chunk_size=3):
>>>    print(start,stop,data)
0 3    col1  col2
0     1     5
1     2    10
2     3    15
3 6    col1  col2
0     4    20
1     5    25
2     6    30
6 7    col1  col2
0     7    35

```

To iterate through the file and output the data in its native tabular type instead, we use {py:func}`iterator_native <tables_io.io_utils.iterator.iterator>` as below:

```{doctest}

>>> for start, stop, data in tables_io.iterator_native('datafile.hdf5',chunk_size=3):
>>>     print(start, stop, data)
0 3 OrderedDict({'col1': array([1, 2, 3]), 'col2': array([ 5, 10, 15])})
3 6 OrderedDict({'col1': array([4, 5, 6]), 'col2': array([20, 25, 30])})
6 7 OrderedDict({'col1': array([7]), 'col2': array([35])})

```

If you want to use MPI, it is currently only supported for HDF5 files. You can specify the rank and MPI size to only iterate through the data chunks that correspond to the current node (rank), as shown below:

```{doctest}

>>> for start, stop, data in tables_io.iterator_native('datafile.hdf5', chunk_size=3,
.. rank=0, parallel_size=3):
>>>     print(start, stop, data)
0 3 OrderedDict({'col1': array([1, 2, 3]), 'col2': array([ 5, 10, 15])})

```

### Writing an HDF5 file from multiple places with MPI

To write data to an HDF5 file using MPI, where multiple processes write to the same file, you need to make sure that your installation of h5py and HDF5 are built with MPI support. This should be the case if you installed the `tables_io` conda environment from `environment.yml` as described in the <project:./quickstart.md#parallel-installation> section.

Here's an example python file that writes out some `astropyTables` to the file `test_mpi_write.hdf5`:

```{literalinclude} assets/mpi.py

```

Generally, the file must be initialized using {py:func}`initialize_HDF5_write <tables_io.io_utils.write.initialize_HDF5_write>`. The data structure that will be written to the file is supplied via a dictionary of the tables. Here, we have a dictionary of dictionaries, where the outer dictionary becomes a group called `data`, and the inner dictionary supplies the names for each dataset (which correspond to columns in the `astropyTable` objects). The values for each dataset key should be `(size, dtype)`.

Once the file has been initialized, each process writes rows `start:end` to the file using {py:func}`write_dict_to_HDF5_chunk <tables_io.io_utils.write.write_dict_to_HDF5_chunk>`. The file is then be closed with {py:func}`finalize_HDF5_write <tables_io.io_utils.write.finalize_HDF5_write>`. This is also an opportunity to write any additional data to the file that doesn't match the structure of the rest of the data, for example some metadata.

Here's an example output when running this file using `mpiexec`:

```{doctest}

>>> mpiexec -n 4 python mpi.py
rank: 0, size: 4, start: 0, end: 25
rank: 2, size: 4, start: 50, end: 75
rank: 1, size: 4, start: 25, end: 50
rank: 3, size: 4, start: 75, end: 100

```

This creates an HDF5 file with two groups, `data` and `metadata`.

## Other examples

### Iterating through data in a parquet file

To iterate through a `parquet` file, yielding only a section of data at a time, we can use the {py:func}`iterator <tables_io.io_utils.iterator.iterator>` or {py:func}`iterator_native <tables_io.io_utils.iterator.iterator>` functions as shown below. You can provide the size of the data section you would like as an `int` to `chunk_size`, size here meaning the number of rows (or length of the `numpy.arrays` in the case of `numpyDict` tables). The default `chunk_size` is 100,000.

To determine the number of rows of data in the file, and therefore what an appropriate chunk size would be, you can use the {py:func}`get_input_data_length <tables_io.io_utils.iterator.get_input_data_length>` function from the `hdf5` module as follows:

```{doctest}

>>> from tables_io import hdf5
>>> hdf5.get_input_data_length('datafile.parquet')
7

```

Here we did not supply the `keys` of the data, since we want all the columns.

Since the length of our file is 7, we will choose a chunk size smaller than that, say 3. To output the data to a `pandasDataFrame`, we use the {py:func}`iterator <tables_io.io_utils.iterator.iterator>` function as shown below:

```{doctest}

>>> import tables_io
>>> for start, stop, data in tables_io.iterator('datafile.parquet','pandasDataFrame',chunk_size=3):
>>>    print(start,stop,data)
0 3    col1  col2
0     1     5
1     2    10
2     3    15
3 6    col1  col2
0     4    20
1     5    25
2     6    30
6 7    col1  col2
0     7    35

```

To iterate through the file and output the data in its native tabular type instead, we use {py:func}`iterator_native <tables_io.io_utils.iterator.iterator>` as below:

```{doctest}

>>> for start, stop, data in tables_io.iterator_native('datafile.parquet',chunk_size=3):
>>>     print(start, stop, data)
0 3 pyarrow.Table
col1: int64
col2: int64
----
col1: [[1,2,3]]
col2: [[5,10,15]]
3 6 pyarrow.Table
col1: int64
col2: int64
----
col1: [[4,5,6]]
col2: [[20,25,30]]
6 7 pyarrow.Table
col1: int64
col2: int64
----
col1: [[7]]
col2: [[35]]

```
