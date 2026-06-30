"""
Utilities for memory-safe, MPI-parallel reading of large parquet files.

The standard PyArrow dataset scanner keeps every row group it has touched in
its internal buffer pool and does not return those pages to the OS even after
Python deletes the references.  For large files (tens of GB) read by many MPI
ranks on the same node this causes unbounded RSS growth and OOM kills - even 
four nodes in NERSC is not enough.

The functions here bypass the scanner and give the caller explicit ownership of
one row group at a time.  After the caller is done with a row group it calls
release_os_memory() to push the freed pages all the way back to the OS through
three layers: Python GC → PyArrow pool → glibc heap.
"""

import gc
import ctypes
import pyarrow as pa
import pyarrow.parquet as pq


def release_os_memory():
    """Return freed memory to the OS through all three caching layers."""
    gc.collect()
    pa.default_memory_pool().release_unused()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def get_parquet_num_rows(filepath):
    """Return the total row count from parquet file metadata (no data loaded)."""
    return pq.ParquetFile(filepath).metadata.num_rows


def parquet_row_group_iterator(filepath, needed_cols, chunk_size, rank, size):
    """Yield (rg_last_row, batch_iter) for each row group in the file.

    rg_last_row : int
        The global row index ONE PAST the last row of this row group.
        After calling comm.Barrier() and verifying all ranks finished
        batch_iter, it is safe to store this as a resume checkpoint.

    batch_iter : generator
        Yields (global_start, global_end, chunk_dict) for each batch
        owned by this rank within the row group.
        The row group is only loaded from disk when batch_iter is first
        iterated — if the caller skips batch_iter entirely (e.g. because
        rg_last_row <= resume_from) the row group is never touched.
        The row group is freed automatically when batch_iter finishes
        or is garbage-collected.

    Typical usage::

        for rg_last_row, batch_iter in parquet_row_group_iterator(...):
            if rg_last_row <= resume_from:
                continue                        # skip, never loads disk
            for s, e, data in batch_iter:
                process(s, e, data)
            comm.Barrier()                      # wait for all ranks
            checkpoint(rg_last_row)             # now safe to bookmark
    """
    pf = pq.ParquetFile(filepath)
    global_chunk_idx = 0
    global_row = 0

    for rg_idx in range(pf.metadata.num_row_groups):
        rg_rows = pf.metadata.row_group(rg_idx).num_rows
        rg_last_row = global_row + rg_rows
        n_batches = (rg_rows + chunk_size - 1) // chunk_size

        # Snapshot loop variables so the inner generator captures the right values.
        # (Without this, all closures would share the same loop variable.)
        _start_chunk_idx = global_chunk_idx
        _start_global_row = global_row

        def _batch_iter(rg_idx=rg_idx, rg_rows=rg_rows,
                        start_chunk_idx=_start_chunk_idx,
                        start_global_row=_start_global_row):
            rg_table = pf.read_row_group(rg_idx, columns=needed_cols)
            try:
                local_row = 0
                chunk_idx = start_chunk_idx
                g_row = start_global_row
                while local_row < rg_rows:
                    batch_rows = min(chunk_size, rg_rows - local_row)
                    if chunk_idx % size == rank:
                        chunk_dict = rg_table.slice(local_row, batch_rows).to_pydict()
                        yield g_row, g_row + batch_rows, chunk_dict
                        del chunk_dict
                    chunk_idx += 1
                    local_row += batch_rows
                    g_row += batch_rows
            finally:
                # Runs whether the inner loop finished normally, was partially
                # consumed, or the generator was garbage-collected mid-way.
                del rg_table
                release_os_memory()

        yield rg_last_row, _batch_iter()

        # Advance outer counters to match what the inner generator would have done,
        # regardless of whether batch_iter was consumed by the caller.
        global_chunk_idx += n_batches
        global_row = rg_last_row
