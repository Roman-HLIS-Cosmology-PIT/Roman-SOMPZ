from typing import Any

import numpy as np
from ceci.config import StageParameter as Param
from qp.metrics.base_metric_classes import MetricOutputType
from qp.metrics.point_estimate_metric_classes import PointToPointMetric

from rail.core.data import TableLike, QPHandle, TableHandle
from rail.evaluation.evaluator import Evaluator


class PointToPointEvaluator(Evaluator):
    """Evaluate the performance of a photo-z estimator against reference point estimate"""

    name = "PointToPointEvaluator"
    config_options = Evaluator.config_options.copy()
    config_options.update(
        hdf5_groupname=Param(
            str, "photometry", required=False, msg="HDF5 Groupname for truth table."
        ),
        reference_dictionary_key=Param(
            str,
            "redshift",
            required=False,
            msg="The key in the `truth` dictionary where the redshift data is stored.",
        ),
        point_estimate_key=Param(
            str, "zmode", required=False, msg="The key in the point estimate table."
        ),
    )
    inputs = [("input", QPHandle), ("truth", TableHandle)]

    metric_base_class = PointToPointMetric

    def _process_chunk(self, data_tuple: Any, first: bool) -> None:
        start = data_tuple[0]
        end = data_tuple[1]
        estimate_data = np.squeeze(data_tuple[2].ancil[self.config.point_estimate_key])
        reference_data = data_tuple[3][self.config.reference_dictionary_key]

        self._process_all_chunk_metrics(
            estimate_data, reference_data, start, end, first
        )

    def _process_all(self, data_tuple: Any) -> None:
        estimate_data = np.squeeze(data_tuple[0].ancil[self.config.point_estimate_key])
        reference_data = data_tuple[1][self.config.hdf5_groupname][
            self.config.reference_dictionary_key
        ]

        self._process_all_metrics(estimate_data, reference_data)


class PointToPointBinnedEvaluator(Evaluator):
    """Evaluate the performance of a photo-z estimator against reference point estimate"""

    name = "PointToPointBinnedEvaluator"
    config_options = Evaluator.config_options.copy()
    config_options.update(
        hdf5_groupname=Param(
            str, "photometry", required=False, msg="HDF5 Groupname for truth table."
        ),
        reference_dictionary_key=Param(
            str,
            "redshift",
            required=False,
            msg="The key in the `truth` dictionary where the redshift data is stored.",
        ),
        point_estimate_key=Param(
            str, "zmode", required=False, msg="The key in the point estimate table."
        ),
        bin_col=Param(
            str, "redshift", required=False, msg="The column metrics are binned by"
        ),
        bin_min=Param(
            float, 0.0, required=False, msg="The mininum value of the binning edge"
        ),
        bin_max=Param(
            float, 3.0, required=False, msg="The maximum value of the binning edge"
        ),
        nbin=Param(
            int, 10, required=False, msg="The mininum value of the binning edge"
        ),
        force_exact=Param(
            bool,
            default=True,
            required=False,
            msg="Force the exact calculation.  This will not allow parallelization",
        ),
    )
    inputs = [("input", QPHandle), ("truth", TableHandle)]

    metric_base_class = PointToPointMetric

    def run(self) -> None:
        self._build_config_dict()

        print(f"Requested metrics: {list(self._metric_config_dict.keys())}")
        data = self._get_all_data()
        estimate_data = np.squeeze(data[0].ancil[self.config.point_estimate_key])
        reference_data = data[1][self.config.hdf5_groupname][
            self.config.reference_dictionary_key
        ]
        bin_edges = np.linspace(
            self.config.bin_min, self.config.bin_max, self.config.nbin
        )
        bin_col = data[1][self.config.hdf5_groupname][self.config.bin_col]
        bin_indices = np.digitize(bin_col, bins=bin_edges) - 1
        self.run_single_node()
        table_list = []

        for i in range(len(bin_edges) - 1):
            estimate_data_subset = estimate_data[bin_indices == i]
            reference_data_subset = reference_data[bin_indices == i]

            summary_table = self._process_all_metrics_binned(
                estimate_data_subset, reference_data_subset
            )
            table_list.append(summary_table)

        out_dict = {}
        out_dict["bin_center"] = (bin_edges[:-1] + bin_edges[1:]) / 2
        metric_keys = table_list[0].keys()
        for key in metric_keys:
            out_dict[key] = np.array(
                [table_list[j][key][0] for j in range(len(table_list))]
            )
        self._summary_handle = self.add_handle("summary", data=out_dict)

    def _process_all_metrics_binned(
        self, estimate_data: Any, reference_data: Any
    ) -> TableLike:
        """This function writes out metric values when operating in non-parallel mode.

        Parameters
        ----------
        estimate_data
            The estimated values (of the appropriate type, float or pdf) to be used
            by the requested metrics.

        reference_data
            The reference or known values (of the appropriate type, float or pdf)
            to be used by the requested metrics.

        Raises
        ------
        ValueError
            Raises an error if an unknown metric is requested.
        """

        summary_table = {}

        for metric, this_metric in self._cached_metrics.items():
            if metric not in self._metric_dict:  # pragma: no cover
                raise ValueError(
                    f"Unsupported metric requested: '{metric}'. "
                    f"Available metrics are: {sorted(self._metric_dict.keys())}"
                )

            metric_result = this_metric.evaluate(estimate_data, reference_data)

            if this_metric.metric_output_type == MetricOutputType.single_value:
                summary_table[metric] = np.array([metric_result])

        return summary_table

    def _process_all(self, data_tuple: Any) -> None:
        estimate_data = np.squeeze(data_tuple[0].ancil[self.config.point_estimate_key])
        reference_data = data_tuple[1][self.config.hdf5_groupname][
            self.config.reference_dictionary_key
        ]

        self._process_all_metrics(estimate_data, reference_data)
