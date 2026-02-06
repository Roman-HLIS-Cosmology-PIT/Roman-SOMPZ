"""A module for manually manipulating HDF5 files in tables_io"""

from .. import io_utils
from ..utils import array_utils

# Exposing HDF5 Interfaces


initialize_HDF5_write = io_utils.write.initialize_HDF5_write

initialize_HDF5_write_single = io_utils.write.initialize_HDF5_write_single

write_dict_to_HDF5_chunk = io_utils.write.write_dict_to_HDF5_chunk

write_dict_to_HDF5_chunk_single = io_utils.write.write_dict_to_HDF5_chunk_single

write_dicts_to_HDF5 = io_utils.write.write_dicts_to_HDF5

finalize_HDF5_write = io_utils.write.finalize_HDF5_write

read_HDF5_group = io_utils.read.read_HDF5_group

read_HDF5_group_names = io_utils.read.read_HDF5_group_names

read_HDF5_to_dict = io_utils.read.read_HDF5_to_dict

read_HDF5_group_to_dict = io_utils.read.read_HDF5_group_to_dict

read_HDF5_dataset_to_array = io_utils.read.read_HDF5_dataset_to_array

get_group_input_data_length = array_utils.get_group_input_data_length

get_input_data_length = io_utils.iterator.get_input_data_length

# Convenience Functions for MPI

data_ranges_by_rank = io_utils.iterator.data_ranges_by_rank
