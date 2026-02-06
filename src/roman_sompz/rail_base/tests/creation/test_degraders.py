import os
from typing import Any

import numpy as np
import pandas as pd
import pytest

from rail.core.data import DATA_STORE, DataHandle, TableHandle
from rail.creation.degraders.addRandom import AddColumnOfRandom
from rail.creation.degraders.quantityCut import QuantityCut


@pytest.fixture
def data() -> DataHandle:
    """Some dummy data to use below."""

    DS = DATA_STORE()
    DS.__class__.allow_overwrite = True

    # generate random normal data
    rng = np.random.default_rng(0)
    x = rng.normal(loc=26, scale=1, size=(100, 7))

    # replace redshifts with reasonable values
    x[:, 0] = np.linspace(0, 2, x.shape[0])

    # return data in handle wrapping a pandas DataFrame
    df = pd.DataFrame(x, columns=["redshift", "u", "g", "r", "i", "z", "y"])
    return DS.add_data("data", df, TableHandle, path="dummy.pq")


@pytest.fixture
def data_forspec() -> DataHandle:
    """Some dummy data to use below."""

    DS = DATA_STORE()
    DS.__class__.allow_overwrite = True

    # generate random normal data
    rng = np.random.default_rng(0)
    x = rng.normal(loc=26, scale=1, size=(200000, 7))

    # replace redshifts with reasonable values
    x[:, 0] = np.linspace(0, 2, x.shape[0])

    # return data in handle wrapping a pandas DataFrame
    df = pd.DataFrame(x, columns=["redshift", "u", "g", "r", "i", "z", "y"])
    return DS.add_data("data_forspec", df, TableHandle, path="dummy_forspec.pq")


@pytest.mark.parametrize(
    "cuts,error",
    [
        (1, TypeError),
        ({"u": "cut"}, TypeError),
        ({"u": dict()}, TypeError),
        ({"u": [1, 2, 3]}, ValueError),
        ({"u": [1, "max"]}, TypeError),
        ({"u": [2, 1]}, ValueError),
        ({"u": TypeError}, TypeError),
    ],
)
def test_QuantityCut_bad_params(cuts: Any, error: type[Exception]) -> None:
    """Test bad parameters that should return Type and Value errors"""
    with pytest.raises(error):
        QuantityCut.make_stage(cuts=cuts)


def test_QuantityCut_returns_correct_shape(  # pylint: disable=redefined-outer-name
    data: Any,
) -> None:
    """Make sure QuantityCut is returning the correct shape"""

    cuts = {
        "u": 30,
        "redshift": (1, 2),
    }
    degrader = QuantityCut.make_stage(cuts=cuts)

    degraded_data = degrader(data).data

    assert (
        degraded_data.shape
        == data.data.query("u < 30 & redshift > 1 & redshift < 2").shape
    )
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))

    degrader_w_flag = QuantityCut.make_stage(
        name="degrader_w_flag", cuts=cuts, drop_rows=False
    )

    degraded_data_w_flag = degrader_w_flag(data).data

    test_mask = np.zeros(len(data.data), dtype=int)
    out_indices = data.data.query("u < 30 & redshift > 1 & redshift < 2").index.values
    test_mask[out_indices] = 1

    assert (degraded_data_w_flag["flag"] == test_mask).all()
    os.remove(
        degrader_w_flag.get_output(
            degrader_w_flag.get_aliased_tag("output"), final_name=True
        )
    )


def test_add_random(data: Any) -> None:  # pylint: disable=redefined-outer-name
    add_random = AddColumnOfRandom.make_stage()

    test_data = add_random(data, seed=1234).data
    assert len(test_data[add_random.config.col_name]) == len(data.data)

    os.remove(
        add_random.get_output(add_random.get_aliased_tag("output"), final_name=True)
    )
