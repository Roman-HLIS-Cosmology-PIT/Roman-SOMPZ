"""
Abstract base classes defining classifiers.
"""

import gc
from typing import Any

import qp

from rail.core.common_params import SHARED_PARAMS
from rail.core.data import (DataHandle, Hdf5Handle, ModelHandle, ModelLike,
                            QPHandle, TableHandle, TableLike)
from rail.core.stage import RailStage


class CatClassifier(RailStage):  # pragma: no cover
    """The base class for assigning classes to catalogue-like table.

    Classifier uses a generic "model", the details of which depends on the sub-class.

    CatClassifier take as "input" a catalogue-like table, assign each object into
    a tomographic bin, and provide as "output" a tabular data which can be appended
    to the catalogue.
    """

    name = "CatClassifier"
    config_options = RailStage.config_options.copy()
    config_options.update(
        chunk_size=SHARED_PARAMS,
        hdf5_groupname=SHARED_PARAMS,
    )
    inputs = [("model", ModelHandle), ("input", TableHandle)]
    outputs = [("output", TableHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Initialize Classifier"""
        super().__init__(args, **kwargs)
        self._output_handle: TableHandle | None = None
        self.model: ModelLike | None = None
        if not isinstance(args, dict):  # pragma: no cover
            args = vars(args)
        self.open_model(**args)

    def classify(self, input_data: TableLike) -> DataHandle:
        """The main run method for the classifier, should be implemented
        in the specific subclass.

        This will attach the input_data to this `CatClassifier`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this Classifier
        by using `self.add_data('output', output_data)`.

        Finally, this will return a TableHandle providing access to that output data.

        Parameters
        ----------
        input_data
            A dictionary of all input data

        Returns
        -------
        DataHandle
            Class assignment for each galaxy.
        """
        self.set_data("input", input_data)
        self.run()
        self.finalize()
        return self.get_handle("output")


class PZClassifier(RailStage):
    """The base class for assigning classes (tomographic bins) to per-galaxy PZ
    estimates.

    PZClassifier takes as "input" a `qp.Ensemble` with per-galaxy PDFs, and
    provides as "output" tabular data which can be appended to the catalogue.
    """

    name = "PZClassifier"
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=SHARED_PARAMS)
    inputs = [("input", QPHandle)]
    outputs = [("output", Hdf5Handle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Initialize the PZClassifier."""
        super().__init__(args, **kwargs)
        self._output_handle: DataHandle | None = None

    def classify(self, input_data: qp.Ensemble) -> DataHandle:
        """The main run method for the classifier, should be implemented
        in the specific subclass.

        This will attach the input_data to this `PZClassifier`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this
        Classifier by using `self.add_data('output', output_data)`.

        The run() method relies on the _process_chunk() method, which should be
        implemented by subclasses to perform the actual classification on each
        chunk of data. The results from each chunk are then combined in the
        _finalize_run() method. (Alternatively, override run() in a subclass to
        perform the classification without parallelization.)

        Finally, this will return a TableHandle providing access to that output data.

        Parameters
        ----------
        input_data
            Per-galaxy p(z), and any ancilary data associated with it

        Returns
        -------
        DataHandle
            Class assignment for each galaxy, typically in the form of a
            dictionary with IDs and class labels.
        """
        self.set_data("input", input_data)
        self.run()
        self.finalize()
        return self.get_handle("output")

    def _finalize_run(self) -> None:
        """Finalize the classification process after processing all chunks."""
        assert self._output_handle is not None
        self._output_handle.finalize_write()

    def _process_chunk(
        self, start: int, end: int, data: qp.Ensemble, first: bool
    ) -> None:
        """Process a chunk of data.

        This method should be implemented in subclasses to perform the actual
        classification on a chunk of data.

        Parameters
        ----------
        start
            The starting index of the chunk.

        end
            The ending index of the chunk.

        data
            The data chunk to be processed.

        first
            True if this is the first chunk, False otherwise.
        """
        raise NotImplementedError(
            f"{self.name}._process_chunk is not implemented"
        )  # pragma: no cover

    def _do_chunk_output(
        self, class_id: TableLike, start: int, end: int, first: bool
    ) -> None:
        """Handle the output of a processed chunk.

        Parameters
        ----------
        class_id
            The classification results for the chunk.

        start
            The starting index of the chunk.

        end
            The ending index of the chunk.

        first
            True if this is the first chunk, False otherwise.
        """
        if first:
            self._output_handle = self.add_handle("output", data=class_id)
            assert self._output_handle is not None
            self._output_handle.initialize_write(
                self._input_length, communicator=self.comm
            )
        assert self._output_handle is not None
        self._output_handle.set_data(class_id, partial=True)
        self._output_handle.write_chunk(start, end)

    def run(self) -> None:
        """Processes the input data in chunks and performs classification.

        This method iterates over chunks of the input data, calling the
        _process_chunk method for each chunk to perform the actual classification.

        The _process_chunk method should be implemented by subclasses to define
        the specific classification logic.
        """
        test_data = self.get_data("input")

        iterator = self.input_iterator("input")
        first = True

        for start, end, test_data in iterator:
            # print(f"Process {self.rank} running estimator on chunk {start} - {end}")
            self._process_chunk(start, end, test_data, first)
            first = False
            # Running garbage collection manually seems to be needed
            # to avoid memory growth for some estimators
            gc.collect()
        self._finalize_run()
