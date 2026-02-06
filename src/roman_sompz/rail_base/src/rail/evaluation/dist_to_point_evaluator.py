from typing import Any

import numpy as np
from ceci.config import StageParameter as Param
from qp.metrics.concrete_metric_classes import DistToPointMetric

from rail.core.data import QPHandle, TableHandle
from rail.evaluation.evaluator import Evaluator


class DistToPointEvaluator(Evaluator):
    """Evaluate the performance of a photo-z estimator against reference point estimate"""

    name = "DistToPointEvaluator"
    config_options = Evaluator.config_options.copy()
    config_options.update(
        limits=Param(
            list,
            [0.0, 3.0],
            required=False,
            msg="The default end points for calculating metrics on a grid.",
        ),
        dx=Param(
            float,
            0.01,
            required=False,
            msg="The default step size when calculating metrics on a grid.",
        ),
        quantile_grid=Param(
            list,
            np.linspace(0, 1, 100).tolist(),
            required=False,
            msg="The quantile value grid on which to evaluate the CDF values. (0, 1)",
        ),
        x_grid=Param(
            list,
            np.linspace(0, 2.5, 301).tolist(),
            required=False,
            msg="The x-value grid at which to evaluate the pdf values.",
        ),
        hdf5_groupname=Param(
            str, "photometry", required=False, msg="HDF5 Groupname for truth table."
        ),
        reference_dictionary_key=Param(
            str,
            "redshift",
            required=False,
            msg="The key in the `truth` dictionary where the redshift data is stored.",
        ),
    )
    inputs = [("input", QPHandle), ("truth", TableHandle)]

    metric_base_class = DistToPointMetric

    def _process_chunk(self, data_tuple: Any, first: bool) -> None:
        start = data_tuple[0]
        end = data_tuple[1]
        estimate_data = data_tuple[2]
        reference_data = data_tuple[3][self.config.reference_dictionary_key]

        self._process_all_chunk_metrics(
            estimate_data, reference_data, start, end, first
        )

    def _process_all(self, data_tuple: Any) -> None:
        estimate_data = data_tuple[0]
        if self.config.hdf5_groupname:
            reference_data = data_tuple[1][self.config.hdf5_groupname][
                self.config.reference_dictionary_key
            ]
        else:  # pragma: no cover
            reference_data = data_tuple[1][self.config.reference_dictionary_key]
        self._process_all_metrics(estimate_data, reference_data)
