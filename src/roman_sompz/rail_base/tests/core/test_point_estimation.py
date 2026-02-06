# pylint: disable=no-member
import numpy as np
import pytest
import qp

from rail.estimation.estimator import CatEstimator


def test_custom_point_estimate() -> None:
    """This test checks to make sure that the inheritance mechanism is working
    for child classes of `CatEstimator`.
    """

    MEANING_OF_LIFE = 42.0

    class TestEstimator(CatEstimator):
        name = "TestEstimator"

        def _calculate_mode_point_estimate(
            self, qp_dist: None = None, grid: None = None
        ) -> np.ndarray:
            return np.ones(100) * MEANING_OF_LIFE

    config_dict = {"calculated_point_estimates": ["zmode"]}

    test_estimator = TestEstimator.make_stage(name="test", **config_dict)

    locs = 2 * (np.random.uniform(size=(100, 1)) - 0.5)
    scales = 1 + 0.2 * (np.random.uniform(size=(100, 1)) - 0.5)
    test_ensemble = qp.Ensemble(
        qp.stats.norm, data=dict(loc=locs, scale=scales)
    )  # pylint: disable=no-member

    result = test_estimator.calculate_point_estimates(test_ensemble)

    assert np.all(result.ancil["zmode"] == MEANING_OF_LIFE)


def test_basic_point_estimate() -> None:
    """This test checks to make sure that all the basic point estimates are
    executed when requested in the configuration dictionary.
    """

    config_dict = {
        "calculated_point_estimates": ["zmean", "zmedian", "zmode", "zbest"],
        "zmin": 0.0,
        "zmax": 3.0,
        "nzbins": 301,
    }

    test_estimator = CatEstimator.make_stage(name="test", **config_dict)

    locs = 2 * (np.random.uniform(size=(100, 1)) - 0.5)
    scales = 1 + 0.2 * (np.random.uniform(size=(100, 1)) - 0.5)
    test_ensemble = qp.Ensemble(
        qp.stats.norm, data=dict(loc=locs, scale=scales)
    )  # pylint: disable=no-member
    result = test_estimator.calculate_point_estimates(test_ensemble, None)

    # note: we're not interested in testing the values of point estimates,
    # just that they were added to the ancillary data.
    assert "zmode" in result.ancil
    assert "zmedian" in result.ancil
    assert "zmean" in result.ancil
    assert "zbest" in result.ancil


def test_mode_no_grid() -> None:
    """This exercises the KeyError logic in `_calculate_mode_point_estimate`."""
    config_dict = {"zmin": 0.0, "nzbins": 100, "calculated_point_estimates": ["mode"]}

    test_estimator = CatEstimator.make_stage(name="test", **config_dict)

    with pytest.raises(AttributeError) as excinfo:
        _ = test_estimator.calculate_point_estimates(None, None)

    assert "object has no attribute" in str(excinfo.value)


def test_mode_no_point_estimates() -> None:
    """This exercises the KeyError logic in `_calculate_mode_point_estimate`."""
    config_dict = {"zmin": 0.0, "nzbins": 100}

    test_estimator = CatEstimator.make_stage(name="test", **config_dict)

    locs = 2 * (np.random.uniform(size=(100, 1)) - 0.5)
    scales = 1 + 0.2 * (np.random.uniform(size=(100, 1)) - 0.5)
    test_ensemble = qp.Ensemble(
        qp.stats.norm, data=dict(loc=locs, scale=scales)
    )  # pylint: disable=no-member

    output_ensemble = test_estimator.calculate_point_estimates(test_ensemble, None)

    assert output_ensemble.ancil is None


def test_keep_existing_ancil_data() -> None:
    """Make sure that we don't overwrite the ancil data if it already exists."""
    config_dict = {
        "zmin": 0.0,
        "zmax": 3.0,
        "nzbins": 100,
        "calculated_point_estimates": ["zmode"],
    }

    test_estimator = CatEstimator.make_stage(name="test", **config_dict)

    locs = 2 * (np.random.uniform(size=(100, 1)) - 0.5)
    scales = 1 + 0.2 * (np.random.uniform(size=(100, 1)) - 0.5)
    test_ensemble = qp.Ensemble(qp.stats.norm, data=dict(loc=locs, scale=scales))

    test_ensemble.set_ancil({"foo": np.zeros(100)})

    output_ensemble = test_estimator.calculate_point_estimates(test_ensemble, None)

    assert "foo" in output_ensemble.ancil
    assert np.all(output_ensemble.ancil["foo"] == 0.0)
    assert len(output_ensemble.ancil["foo"]) == 100


def test_write_new_ancil_data() -> None:
    """Make sure that we don't overwrite the ancil data if it already exists."""
    config_dict = {
        "zmin": 0.0,
        "zmax": 3.0,
        "nzbins": 100,
        "calculated_point_estimates": ["zmode"],
    }

    test_estimator = CatEstimator.make_stage(name="test", **config_dict)

    locs = 2 * (np.random.uniform(size=(100, 1)) - 0.5)
    scales = 1 + 0.2 * (np.random.uniform(size=(100, 1)) - 0.5)
    test_ensemble = qp.Ensemble(qp.stats.norm, data=dict(loc=locs, scale=scales))

    test_ensemble.set_ancil({"foo": np.zeros(100)})

    output_ensemble = test_estimator.calculate_point_estimates(test_ensemble, None)

    assert "zmode" in output_ensemble.ancil
    assert len(output_ensemble.ancil["zmode"]) == 100
