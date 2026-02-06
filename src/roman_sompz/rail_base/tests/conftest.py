import os

import pytest

from rail.utils.path_utils import find_rail_file


@pytest.fixture(name="get_evaluation_files", scope="package")
def get_evaluation_files(request: pytest.FixtureRequest) -> tuple[str, str]:
    possible_local_file = "./examples_data/evaluation_data/data/output_fzboost.hdf5"
    if os.path.exists(possible_local_file):
        pdfs_file = os.path.abspath(possible_local_file)
    else:
        pdfs_file = "examples_data/evaluation_data/data/output_fzboost.hdf5"
        try:
            os.makedirs(os.path.dirname(pdfs_file))
        except FileExistsError:
            pass
        curl_com = f"curl -o {pdfs_file} https://portal.nersc.gov/cfs/lsst/PZ/output_fzboost.hdf5"
        os.system(curl_com)
    ztrue_file = find_rail_file("examples_data/testdata/test_dc2_validation_9816.hdf5")

    def cleanup_evaluation_files() -> None:
        os.unlink(possible_local_file)

    request.addfinalizer(cleanup_evaluation_files)

    return pdfs_file, ztrue_file
