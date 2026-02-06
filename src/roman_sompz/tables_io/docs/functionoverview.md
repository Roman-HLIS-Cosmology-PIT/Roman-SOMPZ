# Function Overview

## Main functions

### Functions that read in data

| Function                                                          | Description                                                                |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------- |
| {py:func}`read <tables_io.io_utils.read.read>`                    | Reads in a file to a `Table-like` or `TableDict-like` object               |
| {py:func}`read_native <tables_io.io_utils.read.read_native>`      | Reads in a file to a `TableDict-like` object of the native tabular type    |
| {py:func}`io_open <tables_io.io_utils.read.io_open>`              | Reads a file as a file object                                              |
| {py:func}`check_columns <tables_io.io_utils.read.check_columns>`  | Checks that the a file has the same or fewer columns than a given list     |
| {py:func}`iterator <tables_io.io_utils.iterator.iterator>`        | Reads in data from a file one chunk at a time to a given format            |
| {py:func}`iterator_native <tables_io.io_utils.iterator.iterator>` | Reads in data from a file one chunk at a time to its native tabular format |

### Functions that write data

| Function                                                        | Description                                                              |
| --------------------------------------------------------------- | ------------------------------------------------------------------------ |
| {py:func}`write <tables_io.io_utils.write.write>`               | Writes a `Table-like` or `TableDict-like` object to a file               |
| {py:func}`write_native <tables_io.io_utils.write.write_native>` | Writes a `Table-like` or `TableDict-like` object to its native file type |

### Other functions

| Function                                                                    | Description                                                                     |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| {py:func}`convert <tables_io.conv.conv_tabledict.convert>`                  | Convert a `Table-like` or `TableDict-like` object to a specified tabular format |
| {py:func}`convert_table <tables_io.conv.conv_table.convert_table>`          | Convert a `Table-like` object to a specified tabular format                     |
| {py:func}`concat_tabledict <tables_io.utils.concat_utils.concat_tabledict>` | Concatenate multiple `TableDict-like` objects                                   |
| {py:func}`concat_table <tables_io.utils.concat_utils.concat_table>`         | Concatenate multiple `Table-like` objects                                       |
| {py:func}`slice_tabledict <tables_io.utils.slice_utils.slice_tabledict>`    | Slice a `TableDict-like` object                                                 |
| {py:func}`slice_table <tables_io.utils.slice_utils.slice_table>`            | Slice a `Table-like` object                                                     |
| {py:func}`get_table_type <tables_io.types.get_table_type>`                  | Returns the tabular type of a `Table-like` or `TableDict-like` object           |

## HDF5 module

The HDF5 module exists to allow users easier access to manually do chunked operations with HDF5 files. While {py:func}`iterator <tables_io.io_utils.iterator.iterator>` provides a way to do chunked reads of HDF5 files, this is the only way to do chunked writes of HDF5 files. To use the `hdf5` module run:

```bash
from tables_io import hdf5
```

### Functions that read in data

| Function &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| {py:func}`read_HDF5_group <tables_io.io_utils.read.read_HDF5_group>`                                                                          | Read and return the requested group and file object from an HDF5 file.       |
| {py:func}`read_HDF5_group_names <tables_io.io_utils.read.read_HDF5_group_names`                                                               | Read and return the list of group names from one level of an HDF5 file       |
| {py:func}`read_HDF5_to_dict <tables_io.io_utils.read.read_HDF5_to_dict>`                                                                      | Reads in data from an HDF5 file to a dictionary                              |
| {py:func}`read_HDF5_group_to_dict <tables_io.io_utils.read.read_HDF5_group_to_dict>`                                                          | Reads in data from an open HDF5 file object to a dictionary or `numpy.array` |
| {py:func}`read_HDF5_dataset_to_array <tables_io.io_utils.read.read_HDF5_dataset_to_array>`                                                    | Reads in all or part of an HDF5 dataset                                      |

### Functions that write data

| Function &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| {py:func}`initialize_HDF5_write <tables_io.io_utils.write.initialize_HDF5_write>`                                               | Prepares an HDF5 file for output.                                                      |
| {py:func}`write_dict_to_HDF5_chunk <tables_io.io_utils.write.write_dict_to_HDF5_chunk>`                                         | Writes a chunk of data from a `TableDict-like` object to an HDF5 group or file object. |
| {py:func}`write_dicts_to_HDF5 <tables_io.io_utils.write.write_dicts_to_HDF5>`                                                   | Writes a `Table-like` object to an HDF5 file given a filepath.                         |
| {py:func}`finalize_HDF5_write <tables_io.io_utils.write.finalize_HDF5_write>`                                                   | Writes any last data and closes an HDF5 file object.                                   |

### Utility functions

| Function &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| {py:func}`get_group_input_data_length <tables_io.utils.array_utils.get_group_input_data_length>`                                              | Returns the length of an HDF5 group object                                    |
| {py:func}`get_input_data_length <tables_io.io_utils.iterator.get_input_data_length>`                                                          | Opens a file and gets the length of the first axis of the data in that file.  |
| {py:func}`data_ranges_by_rank <tables_io.io_utils.iterator.data_ranges_by_rank>`                                                              | Given a number of rows and chunk size, yields the data range for each process |
