from mpi4py import MPI
from tables_io import hdf5
from astropy.table import Table
import numpy as np


def make_table(rank: int, nrows: int = 25):
    # Makes a chunk of data to write
    data = dict(col_1=np.ones(nrows) * rank, col_2=np.random.uniform(size=nrows))
    table = Table(data)
    return dict(data=table)


# calculate start and end values
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size
nrows = 25
start = rank * nrows
end = (rank + 1) * nrows
tot_rows = nrows * size
print(f"rank: {rank}, start: {start}, end: {end}")

# set up the format of the data to initialize the file
dout = {"data": {"col_1": ((tot_rows,), "float64"), "col_2": ((tot_rows,), "float64")}}
groups, fout = hdf5.initialize_HDF5_write(
    "./test_mpi_write.hdf5", comm=MPI.COMM_WORLD, **dout
)


# write data to file
data = make_table(rank, nrows)
hdf5.write_dict_to_HDF5_chunk(groups, data, start, end)

# close the file and write some additional data
metadata = Table({"ids": np.arange(0, 25, 1, dtype=np.int32)})
hdf5.finalize_HDF5_write(fout, "metadata", **metadata)
