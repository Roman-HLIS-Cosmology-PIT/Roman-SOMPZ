# Technical Debt

On this page, current known issues and technical debt will be recorded. These issues would make
good first contributions for new contributors to the project. You can also take a look at the [issues](https://github.com/LSSTDESC/tables_io/issues) page for more.

## Code Debt

- [ ] {py:func}`iter_H5_to_dataframe <tables_io.io_utils.iterator.iter_H5_to_dataframe>` is not implemented, iterator cannot iterate over 'h5' files
- [ ] {py:func}`pa_table_to_recarray <tables_io.conv.conv_table.pa_table_to_recarray>` is not implemented, but the functionality is being accomplished via converting to astropy Table first
- [ ] Migrate `tests/io_utils/test_io_classic.py` to `pytest` fully, and remove `unittest` import.
- [ ] Functions that could use tests or more in-depth tests:
  - [ ] split_tasks_by_rank
  - [ ] data_ranges_by_rank
  - [ ] check_columns
  - [ ] io_open

### Deprecation

The following functions and classes are deprecated, and should be removed along _some timeline_.

- [ ] {py:class}`TableDict <tables_io.table_dict>` - not currently in use in the code
- [ ] {py:func}`writeNative <tables_io.writeNative>` - replaced by {py:func}`write_native <tables_io.io_utils.write.write_native>`
- [ ] {py:func}`readNative <tables_io.readNative>` - replaced by {py:func}`read_native <tables_io.io_utils.read.read_native>`
- [ ] {py:func}`iteratorNative <tables_io.iteratorNative>` - replaced by {py:func}`iterator_native <tables_io.io_utils.iterator.iterator_native>`
- [ ] {py:func}`convertObj <tables_io.convertObj>` - replaced by {py:func}`convert_table <tables_io.conv.conv_table.convert_table>`
- [ ] {py:func}`concatObjs <tables_io.concatObjs>` - replaced by {py:func}`concat_table <tables_io.utils.concat_utils.concat_table>`
- [ ] {py:func}`concat <tables_io.concat>` - replaced by {py:func}`concat_tabledict <tables_io.utils.concat_utils.concat_tabledict>`
- [ ] {py:func}`sliceObj <tables_io.sliceObj>` - replaced by {py:func}`slice_table <tables_io.utils.slice_utils.slice_table>`
- [ ] {py:func}`sliceObjs <tables_io.sliceObjs>` - replaced by {py:func}`slice_tabledict <tables_io.utils.slice_utils.slice_tabledict>`
