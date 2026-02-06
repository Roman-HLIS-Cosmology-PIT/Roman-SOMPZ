import os

import numpy as np
import pytest

from rail.interfaces import PZFactory
from rail.utils.path_utils import find_rail_file


def test_pz_factory() -> None:
    stage = PZFactory.build_cat_estimator_stage(
        "train_z",
        "TrainZEstimator",
        "rail.estimation.algos.train_z",
        "tests/interfaces/model_inform_trainz.pkl",
        "dummy.in",
    )

    input_file = find_rail_file("examples_data/testdata/validation_10gal.hdf5")

    out_single = PZFactory.estimate_single_pz(stage, {"d": np.array([1, 1])})
    assert out_single.npdf == 1

    _out_handle = PZFactory.run_cat_estimator_stage(
        stage,
        input_file,
    )

    check_stage = PZFactory.get_cat_estimator_stage("train_z")
    assert check_stage == stage

    with pytest.raises(KeyError):
        PZFactory.get_cat_estimator_stage("nope")

    PZFactory.reset()
    assert not PZFactory._stage_dict

    try:
        os.unlink("inprogress_output_train_z.hdf5")
    except FileNotFoundError:
        pass
    try:
        os.unlink("output_train_z.hdf5")
    except FileNotFoundError:
        pass
