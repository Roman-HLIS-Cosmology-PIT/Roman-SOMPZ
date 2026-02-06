"""
" Abstract base class defining an Evaluator

The key feature is that the evaluate method.
"""

from typing import Any, Generator, Iterable

import numpy as np
from ceci.config import StageParameter as Param
from qp.metrics import MetricInputType, MetricOutputType
from qp.metrics.base_metric_classes import BaseMetric

from rail.core.data import QPOrTableHandle
from rail.evaluation.evaluator import Evaluator


class SingleEvaluator(Evaluator):  # pylint: disable=too-many-instance-attributes
    """Evaluate the performance of a photo-Z estimator"""

    name = "SingleEvaluator"
    config_options = Evaluator.config_options.copy()
    config_options.update(
        point_estimates=Param(list, msg="List of point estimates to use", default=[]),
        truth_point_estimates=Param(
            list, msg="List of true point values to use", default=[]
        ),
        hdf5_groupname=Param(
            str, "photometry", required=False, msg="HDF5 Groupname for truth table."
        ),
    )
    inputs = [("input", QPOrTableHandle), ("truth", QPOrTableHandle)]

    metric_base_class = BaseMetric

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Initialize Evaluator"""
        super().__init__(args, **kwargs)
        self._input_data_type = QPOrTableHandle.PdfOrValue.unknown
        self._truth_data_type = QPOrTableHandle.PdfOrValue.unknown
        self._out_table: dict = {}
        self._summary_table: dict = {}

    def run(self) -> None:  # pylint: disable=too-many-branches
        """Run method

        Evaluate all the metrics and put them into a table

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Get the truth data from the data store under this stages 'truth' tag
        Puts the data into the data store under this stages 'output' tag
        """
        input_data_handle = self.get_handle("input", allow_missing=True)
        truth_data_handle = self.get_handle("truth", allow_missing=True)

        self._input_data_type = input_data_handle.check_pdf_or_point()
        self._truth_data_type = truth_data_handle.check_pdf_or_point()

        Evaluator.run(self)

    def _process_chunk(self, data_tuple: Any, first: bool) -> None:
        start = data_tuple[0]
        end = data_tuple[1]
        input_data = data_tuple[2]
        truth_data = data_tuple[3]

        for metric, this_metric in self._cached_metrics.items():
            if (
                this_metric.metric_input_type == MetricInputType.single_ensemble
            ):  # pragma: no cover
                if not self._input_data_type.has_dist():  # pragma: no cover
                    print(
                        f"skipping {metric} {self._input_data_type} {this_metric.metric_input_type}"
                    )
                    continue
                key_val = f"{metric}"
                self._process_chunk_single_ensemble(this_metric, key_val, input_data)
            elif (
                this_metric.metric_input_type == MetricInputType.dist_to_dist
            ):  # pragma: no cover
                if (
                    not self._input_data_type.has_dist()
                    or not self._truth_data_type.has_dist()
                ):  # pragma: no cover
                    print(
                        f"skipping {metric} {self._input_data_type} {this_metric.metric_input_type}"
                    )
                    continue
                key_val = f"{metric}"
                self._process_chunk_dist_to_dist(
                    this_metric, key_val, input_data, truth_data
                )
            elif this_metric.metric_input_type == MetricInputType.dist_to_point:
                if (
                    not self._input_data_type.has_dist()
                    or not self._truth_data_type.has_point()
                ):  # pragma: no cover
                    print(
                        f"skipping {metric} {self._input_data_type} {this_metric.metric_input_type}"
                    )
                    continue
                for truth_point_estimate_ in self.config.truth_point_estimates:
                    key_val = f"{metric}_{truth_point_estimate_}"
                    self._process_chunk_dist_to_point(
                        this_metric,
                        key_val,
                        input_data,
                        np.squeeze(truth_data[truth_point_estimate_]),
                    )
            elif this_metric.metric_input_type == MetricInputType.point_to_point:
                if (
                    not self._input_data_type.has_point()
                    or not self._truth_data_type.has_point()
                ):  # pragma: no cover
                    print(
                        f"skipping {metric} {self._input_data_type} {this_metric.metric_input_type}"
                    )
                    continue
                for point_estimate_ in self.config.point_estimates:
                    point_data = np.squeeze(input_data.ancil[point_estimate_])
                    for truth_point_estimate_ in self.config.truth_point_estimates:
                        key_val = f"{metric}_{point_estimate_}_{truth_point_estimate_}"
                        self._process_chunk_point_to_point(
                            this_metric,
                            key_val,
                            point_data,
                            np.squeeze(truth_data[truth_point_estimate_]),
                        )
            elif (
                this_metric.metric_input_type == MetricInputType.point_to_dist
            ):  # pragma: no cover
                if (
                    not self._input_data_type.has_point()
                    or not self._truth_data_type.has_dist()
                ):  # pragma: no cover
                    print(
                        f"skipping {metric} {self._input_data_type} {this_metric.metric_input_type}"
                    )
                    continue
                for point_estimate_ in self.config.point_estimates:
                    key_val = f"{metric}_{point_estimate_}"
                    self._process_chunk_point_to_dist(
                        this_metric,
                        key_val,
                        point_data,
                        truth_data[truth_point_estimate_],
                    )

        self._output_table_chunk_data(start, end, self._out_table, first)

    def _process_all(self, data_tuple: Any) -> None:
        input_data = data_tuple[0]
        truth_data = data_tuple[1]

        if self.config.hdf5_groupname:
            truth_data = truth_data[self.config.hdf5_groupname]

        for metric, this_metric in self._cached_metrics.items():
            if (
                this_metric.metric_input_type == MetricInputType.single_ensemble
            ):  # pragma: no cover
                if not self._input_data_type.has_dist():  # pragma: no cover
                    continue
                key_val = f"{metric}"
                self._process_all_single_ensemble(this_metric, key_val, input_data)
            elif (
                this_metric.metric_input_type == MetricInputType.dist_to_dist
            ):  # pragma: no cover
                if (
                    not self._input_data_type.has_dist()
                    or not self._truth_data_type.has_dist()
                ):  # pragma: no cover
                    continue
                key_val = f"{metric}"
                self._process_all_dist_to_dist(
                    this_metric, key_val, input_data, truth_data
                )
            elif this_metric.metric_input_type == MetricInputType.dist_to_point:
                if (
                    not self._input_data_type.has_dist()
                    or not self._truth_data_type.has_point()
                ):  # pragma: no cover
                    continue
                for truth_point_estimate_ in self.config.truth_point_estimates:
                    key_val = f"{metric}_{truth_point_estimate_}"
                    self._process_all_dist_to_point(
                        this_metric,
                        key_val,
                        input_data,
                        truth_data[truth_point_estimate_],
                    )
            elif this_metric.metric_input_type == MetricInputType.point_to_point:
                if (
                    not self._input_data_type.has_point()
                    or not self._truth_data_type.has_point()
                ):  # pragma: no cover
                    continue
                for point_estimate_ in self.config.point_estimates:
                    point_data = input_data.ancil[point_estimate_]
                    for truth_point_estimate_ in self.config.truth_point_estimates:
                        key_val = f"{metric}_{point_estimate_}_{truth_point_estimate_}"
                        self._process_all_point_to_point(
                            this_metric,
                            key_val,
                            np.squeeze(point_data),
                            truth_data[truth_point_estimate_],
                        )
            elif (
                this_metric.metric_input_type == MetricInputType.point_to_dist
            ):  # pragma: no cover
                if (
                    not self._input_data_type.has_point()
                    or not self._truth_data_type.has_dist()
                ):  # pragma: no cover
                    continue
                for point_estimate_ in self.config.point_estimates:
                    key_val = f"{metric}_{point_estimate_}"
                    self._process_all_point_to_dist(
                        this_metric,
                        key_val,
                        point_data,
                        truth_data[truth_point_estimate_],
                    )

        out_table_to_write = {
            key: np.array(val).astype(float) for key, val in self._out_table.items()
        }
        self._output_handle = self.add_handle("output", data=out_table_to_write)
        self._summary_handle = self.add_handle("summary", data=self._summary_table)
        self._single_distribution_summary_handle = self.add_handle(
            "single_distribution_summary", data=self._single_distribution_summary_data
        )

    def _process_chunk_single_ensemble(
        self, this_metric: BaseMetric, key: str, input_data: Any
    ) -> None:  # pragma: no cover
        if this_metric.metric_output_type == MetricOutputType.single_value:
            if not hasattr(this_metric, "accumulate"):
                print(
                    f"{this_metric.metric_name} with output type "
                    "single_value does not support parallel processing"
                )
                return

            centroids = this_metric.accumulate(input_data)

            if self.comm:
                self._cached_data[key] = centroids
            else:
                if key in self._cached_data:
                    self._cached_data[key].append(centroids)
                else:
                    self._cached_data[key] = [centroids]

        elif this_metric.metric_output_type == MetricOutputType.single_distribution:
            if not hasattr(this_metric, "accumulate"):  # pragma: no cover
                print(
                    f"{this_metric.metric_name} with output type "
                    "MetricOutputType.single_value does not support parallel processing yet"
                )
                return

            accumulated_data = this_metric.accumulate(input_data)
            if self.comm:
                self._cached_data[key] = accumulated_data
            else:
                if key in self._cached_data:
                    self._cached_data[key].append(accumulated_data)
                else:
                    self._cached_data[key] = [accumulated_data]
        else:
            self._out_table[key] = this_metric.evaluate(input_data)

    def _process_chunk_dist_to_dist(
        self, this_metric: BaseMetric, key: str, input_data: Any, truth_data: Any
    ) -> None:  # pragma: no cover
        if this_metric.metric_output_type == MetricOutputType.single_value:
            if not hasattr(this_metric, "accumulate"):
                print(
                    f"{this_metric.metric_name} with output type "
                    "single_value does not support parallel processing yet"
                )
                return

            centroids = this_metric.accumulate(input_data, truth_data)
            if self.comm:
                self._cached_data[key] = centroids
            else:
                if key in self._cached_data:
                    self._cached_data[key].append(centroids)
                else:
                    self._cached_data[key] = [centroids]

        elif this_metric.metric_output_type == MetricOutputType.single_distribution:
            if not hasattr(this_metric, "accumulate"):  # pragma: no cover
                print(
                    f"{this_metric.metric_name} with output type "
                    "MetricOutputType.single_value does not support parallel processing yet"
                )
                return

            accumulated_data = this_metric.accumulate(input_data, truth_data)
            if self.comm:
                self._cached_data[key] = accumulated_data
            else:
                if key in self._cached_data:
                    self._cached_data[key].append(accumulated_data)
                else:
                    self._cached_data[key] = [accumulated_data]

        else:
            self._out_table[key] = this_metric.evaluate(input_data, truth_data)

    def _process_chunk_dist_to_point(
        self, this_metric: BaseMetric, key: str, input_data: Any, truth_data: Any
    ) -> None:
        if this_metric.metric_output_type == MetricOutputType.single_value:
            if not hasattr(this_metric, "accumulate"):  # pragma: no cover
                print(
                    f"{this_metric.metric_name} with output type "
                    "single_value does not support parallel processing yet"
                )
                return

            centroids = this_metric.accumulate(input_data, truth_data)
            if self.comm:  # pragma: no cover
                self._cached_data[key] = centroids
            else:
                if key in self._cached_data:
                    self._cached_data[key].append(centroids)
                else:
                    self._cached_data[key] = [centroids]

        elif (
            this_metric.metric_output_type == MetricOutputType.single_distribution
        ):  # pragma: no cover
            if not hasattr(this_metric, "accumulate"):
                print(
                    f"{this_metric.metric_name} with output type "
                    "single_value does not support parallel processing yet"
                )
                return
            accumulated_data = this_metric.accumulate(input_data, truth_data)
            if self.comm:
                self._cached_data[key] = accumulated_data
            else:
                if key in self._cached_data:
                    self._cached_data[key].append(accumulated_data)
                else:
                    self._cached_data[key] = [accumulated_data]

        else:  # pragma: no cover
            self._out_table[key] = this_metric.evaluate(input_data, truth_data)

    def _process_chunk_point_to_dist(
        self, this_metric: BaseMetric, key: str, input_data: Any, truth_data: Any
    ) -> None:  # pragma: no cover
        if this_metric.metric_output_type == MetricOutputType.single_value:
            if not hasattr(this_metric, "accumulate"):  # pragma: no cover
                print(
                    f"{this_metric.metric_name} with output type "
                    "single_value does not support parallel processing yet"
                )
                return

            centroids = this_metric.accumulate(input_data, truth_data)
            if self.comm:  # pragma: no cover
                self._cached_data[key] = centroids
            else:
                if key in self._cached_data:
                    self._cached_data[key].append(centroids)
                else:
                    self._cached_data[key] = [centroids]

        elif this_metric.metric_output_type == MetricOutputType.single_distribution:
            if not hasattr(this_metric, "accumulate"):  # pragma: no cover
                print(
                    f"{this_metric.metric_name} with output type "
                    "single_value does not support parallel processing yet"
                )
                return
            accumulated_data = this_metric.accumulate(input_data, truth_data)
            if self.comm:
                self._cached_data[key] = accumulated_data
            else:
                if key in self._cached_data:
                    self._cached_data[key].append(accumulated_data)
                else:
                    self._cached_data[key] = [accumulated_data]
        else:
            self._out_table[key] = this_metric.evaluate(input_data, truth_data)

    def _process_chunk_point_to_point(
        self, this_metric: BaseMetric, key: str, input_data: Any, truth_data: Any
    ) -> None:
        if this_metric.metric_output_type == MetricOutputType.single_value:
            if not hasattr(this_metric, "accumulate"):  # pragma: no cover
                print(
                    f"{this_metric.metric_name} with output type "
                    "single_value does not support parallel processing yet"
                )
                return

            centroids = this_metric.accumulate(input_data, truth_data)
            if self.comm:  # pragma: no cover
                self._cached_data[key] = centroids
            else:
                if key in self._cached_data:
                    self._cached_data[key].append(centroids)
                else:
                    self._cached_data[key] = [centroids]

        elif (
            this_metric.metric_output_type == MetricOutputType.single_distribution
        ):  # pragma: no cover
            if not hasattr(this_metric, "accumulate"):
                print(
                    f"{this_metric.metric_name} with output type "
                    "single_value does not support parallel processing yet"
                )
                return
            accumulated_data = this_metric.accumulate(input_data, truth_data)
            if self.comm:
                self._cached_data[key] = accumulated_data
            else:
                if key in self._cached_data:
                    self._cached_data[key].append(accumulated_data)
                else:
                    self._cached_data[key] = [accumulated_data]
        else:
            self._out_table[key] = this_metric.evaluate(input_data, truth_data)

    def _process_all_single_ensemble(
        self, this_metric: BaseMetric, key: str, input_data: Any
    ) -> None:  # pragma: no cover
        if this_metric.metric_output_type == MetricOutputType.single_value:
            self._summary_table[key] = np.array([this_metric.evaluate(input_data)])

        elif this_metric.metric_output_type == MetricOutputType.single_distribution:
            self._single_distribution_summary_data[key] = this_metric.evaluate(
                input_data
            )
        else:
            self._out_table[key] = this_metric.evaluate(input_data)

    def _process_all_dist_to_dist(
        self, this_metric: BaseMetric, key: str, input_data: Any, truth_data: Any
    ) -> None:  # pragma: no cover
        if this_metric.metric_output_type == MetricOutputType.single_value:
            self._summary_table[key] = np.array(
                [this_metric.evaluate(input_data, truth_data)]
            )

        elif this_metric.metric_output_type == MetricOutputType.single_distribution:
            self._single_distribution_summary_data[key] = this_metric.evaluate(
                input_data, truth_data
            )
        else:
            self._out_table[key] = this_metric.evaluate(input_data, truth_data)

    def _process_all_dist_to_point(
        self, this_metric: BaseMetric, key: str, input_data: Any, truth_data: Any
    ) -> None:
        if this_metric.metric_output_type == MetricOutputType.single_value:
            self._summary_table[key] = np.array(
                [this_metric.evaluate(input_data, truth_data)]
            )

        elif (
            this_metric.metric_output_type == MetricOutputType.single_distribution
        ):  # pragma: no cover
            self._single_distribution_summary_data[key] = this_metric.evaluate(
                input_data, truth_data
            )
        else:  # pragma: no cover
            self._out_table[key] = this_metric.evaluate(input_data, truth_data)

    def _process_all_point_to_dist(
        self, this_metric: BaseMetric, key: str, input_data: Any, truth_data: Any
    ) -> None:  # pragma: no cover
        if this_metric.metric_output_type == MetricOutputType.single_value:
            self._summary_table[key] = np.array(
                [this_metric.evaluate(input_data, truth_data)]
            )

        elif this_metric.metric_output_type == MetricOutputType.single_distribution:
            self._single_distribution_summary_data[key] = this_metric.evaluate(
                input_data, truth_data
            )
        else:
            self._out_table[key] = this_metric.evaluate(input_data, truth_data)

    def _process_all_point_to_point(
        self, this_metric: BaseMetric, key: str, input_data: Any, truth_data: Any
    ) -> None:
        if this_metric.metric_output_type == MetricOutputType.single_value:
            self._summary_table[key] = np.array(
                [this_metric.evaluate(input_data, truth_data)]
            )

        elif (
            this_metric.metric_output_type == MetricOutputType.single_distribution
        ):  # pragma: no cover
            self._single_distribution_summary_data[key] = this_metric.evaluate(
                input_data, truth_data
            )
        else:
            self._out_table[key] = this_metric.evaluate(input_data, truth_data)

    def _setup_iterator(self, itrs: list[Iterable] | None = None) -> Generator:
        if itrs is None:
            tags = ["input", "truth"]
            itrs = [
                self.input_iterator(tag, groupname=self.config.hdf5_groupname)
                for tag in tags
            ]
        return Evaluator._setup_iterator(self, itrs)
