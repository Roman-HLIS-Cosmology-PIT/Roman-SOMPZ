"""Tests for Writing Functions
"""

import pytest
import numpy as np
import os

from tables_io import io_utils
from tests.helpers.utilities import check_deps
from tables_io.lazy_modules import h5py


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_write_output_file(h5_test_outfile, tmp_path):
    """Test writing an HDF5 output file"""
    n_pdf = 40
    n_bins = 21
    pz_pdf = np.random.uniform(size=(n_pdf, n_bins))
    z_grid = np.linspace(0, 4, n_bins)
    z_mode = z_grid[np.argmax(pz_pdf, axis=1)]

    data_dict = {"data": dict(zmode=z_mode, pz_pdf=pz_pdf)}

    groups, outf = io_utils.write.initialize_HDF5_write(
        h5_test_outfile,
        data=dict(photoz_mode=((n_pdf,), "f4"), photoz_pdf=((n_pdf, n_bins), "f4")),
    )
    io_utils.write.write_dict_to_HDF5_chunk(
        groups, data_dict, 0, n_pdf, zmode="photoz_mode", pz_pdf="photoz_pdf"
    )
    io_utils.write.finalize_HDF5_write(outf, "md", zgrid=z_grid)
    os.unlink(h5_test_outfile)

    # Testing failed writes

    with pytest.raises(RuntimeError):
        io_utils.write.write(data_dict, tmp_path / "nonexistent_dir/tmp.fits")

    with pytest.raises(RuntimeError):
        io_utils.write.write(data_dict, tmp_path / "nonexistent_dir/tmp.hdf5")


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_write_output_parallel_file(h5_test_outfile):
    """Testing Parallel Writing"""
    from mpi4py import MPI

    """ Testing parallel write """
    comm = MPI.COMM_WORLD
    n_pdf = 40
    n_bins = 21
    pz_pdf = np.random.uniform(size=(n_pdf, n_bins))
    z_grid = np.linspace(0, 4, n_bins)
    z_mode = z_grid[np.argmax(pz_pdf, axis=1)]

    data_dict = {"data": dict(zmode=z_mode, pz_pdf=pz_pdf)}

    groups, outf = io_utils.write.initialize_HDF5_write(
        h5_test_outfile,
        data=dict(photoz_mode=((n_pdf,), "f4"), photoz_pdf=((n_pdf, n_bins), "f4")),
        comm=comm,
    )
    io_utils.write.write_dict_to_HDF5_chunk(
        groups, data_dict, 0, n_pdf, zmode="photoz_mode", pz_pdf="photoz_pdf"
    )
    io_utils.write.finalize_HDF5_write(outf, "md", zgrid=z_grid)
    os.unlink(h5_test_outfile)


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_write_output_file_single(h5_test_outfile):
    """Test writing an output file"""
    npdf = 40
    nbins = 21
    pz_pdf = np.random.uniform(size=(npdf, nbins))
    zgrid = np.linspace(0, 4, nbins)
    zmode = zgrid[np.argmax(pz_pdf, axis=1)]

    data_dict = dict(zmode=zmode, pz_pdf=pz_pdf)

    group, outf = io_utils.write.initialize_HDF5_write_single(
        h5_test_outfile,
        "data",
        photoz_mode=((npdf,), "f4"),
        photoz_pdf=((npdf, nbins), "f4"),
    )
    io_utils.write.write_dict_to_HDF5_chunk_single(
        group, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf"
    )
    io_utils.write.finalize_HDF5_write(outf, "md", zgrid=zgrid)

    os.unlink(h5_test_outfile)

    group, outf = io_utils.write.initialize_HDF5_write_single(
        h5_test_outfile,
        None,
        photoz_mode=((npdf,), "f4"),
        photoz_pdf=((npdf, nbins), "f4"),
    )
    io_utils.write.write_dict_to_HDF5_chunk_single(
        group, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf"
    )
    io_utils.write.finalize_HDF5_write(outf, "md", zgrid=zgrid)

    os.unlink(h5_test_outfile)


@pytest.mark.skipif(not check_deps([h5py]), reason="Missing HDF5")
def test_write_output_parallel_file_single(h5_test_outfile):
    """Testing parallel writing"""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    """ Test writing an output file """
    npdf = 40
    nbins = 21
    pz_pdf = np.random.uniform(size=(npdf, nbins))
    zgrid = np.linspace(0, 4, nbins)
    zmode = zgrid[np.argmax(pz_pdf, axis=1)]

    data_dict = dict(zmode=zmode, pz_pdf=pz_pdf)

    group, outf = io_utils.write.initialize_HDF5_write_single(
        h5_test_outfile,
        "data",
        photoz_mode=((npdf,), "f4"),
        photoz_pdf=((npdf, nbins), "f4"),
        comm=comm,
    )
    io_utils.write.write_dict_to_HDF5_chunk_single(
        group, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf"
    )
    io_utils.write.finalize_HDF5_write(outf, "md", zgrid=zgrid)

    os.unlink(h5_test_outfile)

    group, outf = io_utils.write.initialize_HDF5_write_single(
        h5_test_outfile,
        None,
        photoz_mode=((npdf,), "f4"),
        photoz_pdf=((npdf, nbins), "f4"),
        comm=comm,
    )
    io_utils.write.write_dict_to_HDF5_chunk_single(
        group, data_dict, 0, npdf, zmode="photoz_mode", pz_pdf="photoz_pdf"
    )
    io_utils.write.finalize_HDF5_write(outf, "md", zgrid=zgrid)

    os.unlink(h5_test_outfile)
