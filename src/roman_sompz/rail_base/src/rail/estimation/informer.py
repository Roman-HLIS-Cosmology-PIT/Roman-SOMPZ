"""
Abstract base classes for Informers.
These superstages ingest prior information, including training sets and explicit
priors, and prepare a model that can be used to produce photo-z data products.
They are distinguished by their input data types, and the models they output can
be used for their corresponding Estimator, Summarizer, or Classifier stages.
"""

from typing import Any, Generator
import numpy as np

import qp

from rail.core.common_params import SHARED_PARAMS
from rail.core.data import (DataHandle, ModelHandle, QPHandle, TableHandle,
                            TableLike)
from rail.core.stage import RailStage
import tables_io


class CatInformer(RailStage):
    """The base class for informing models used to make photo-z data products
    from catalog-like inputs (i.e., tables with fluxes in photometric bands among
    the set of columns).

    Estimators use a generic "model", the details of which depends on the sub-class.
    Most estimators will have associated Informer classes, which can be used to inform
    those models.

    (Note, "Inform" is more generic than "Train" as it also applies to algorithms that
    are template-based rather than machine learning-based.)

    Informer will produce as output a generic "model", the details of which depends on the sub-class.

    They take as "input" catalog-like tabular data, which is used to "inform" the model.
    """

    name = "CatInformer"
    config_options = RailStage.config_options.copy()
    inputs = [("input", TableHandle)]
    outputs = [("model", ModelHandle)]
    config_options.update(
        hdf5_groupname=SHARED_PARAMS,
    )

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Initialize Informer that can inform models for redshift estimation"""
        super().__init__(args, **kwargs)
        self.model = None

    def inform(self, training_data: TableLike) -> DataHandle:
        """The main interface method for Informers

        This will attach the input_data to this `Informer`
        (for introspection and provenance tracking).

        Then it will call the run(), validate() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the model that it creates to this Estimator
        by using `self.add_data('model', model)`.

        Finally, this will return a ModelHandle providing access to the trained model.

        Parameters
        ----------
        input_data
            dictionary of all input data, or a `TableHandle` providing access to it

        Returns
        -------
        ModelHandle
            Handle providing access to trained model
        """

        self.set_data("input", training_data)
        self.validate()
        self.run()
        self.finalize()
        return self.get_handle("model")

    def _convert_table_format(self, data: TableLike, out_fmt_str: str="numpyDict") -> TableLike: # pragma: no cover
        """
        Utility function to convert existing Tabular data to a numpy dictionary,
        ingestable for most informer and estimators.
        To be called in run().
        """
        # required format for informer/estimator
        out_fmt = tables_io.types.TABULAR_FORMAT_NAMES[out_fmt_str] 
        out_data = tables_io.convert(data, out_fmt)
        # overwrite set_data
        return out_data
        

class PzInformer(RailStage):
    """The base class for informing models used to make photo-z data products from
    existing ensembles of p(z) distributions.

    PzInformer can use a generic "model", the details of which depends on the sub-class.
    Some summarizer will have associated PzInformer classes, which can be used to inform
    those models.

    (Note, "Inform" is more generic than "Train" as it also applies to algorithms that
    are template-based rather than machine learning-based.)

    PzInformer will produce as output a generic "model", the details of which depends on the sub-class.

    They take as "input" a qp.Ensemble of per-galaxy p(z) data, which is used to "inform" the model.
    """

    name = "PzInformer"
    config_options = RailStage.config_options.copy()
    config_options.update(
        hdf5_groupname=SHARED_PARAMS,
        chunk_size=SHARED_PARAMS,
    )
        
    inputs = [("input", QPHandle), ("truth", TableHandle)]
    outputs = [("model", ModelHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Initialize Informer that can inform models for redshift estimation"""
        super().__init__(args, **kwargs)
        self.model = None
        self.model_handle: ModelHandle | None = None

    def _setup_iterator(self) -> Generator:

        itrs = []
        input_itr = self.input_iterator("input", groupname="")
        truth_itr = self.input_iterator("truth", groupname=self.config.hdf5_groupname)

        if input_itr:
            itrs.append(input_itr)
        else:  # pragma: no cover
            return []
        if truth_itr:  # pragma: no cover
            itrs.append(truth_itr)
            
        for it in zip(*itrs):
            first = True
            for s, e, d in it:
                true_redshift: np.ndarray | None = None
                if first:
                    start = s
                    end = e
                    qp_ens = d
                    first = False
                else:  # pragma: no cover
                    true_redshift = d[self.config.redshift_col]

            yield start, end, qp_ens, true_redshift

    def inform(
        self, training_data: qp.Ensemble | str="None", truth_data: TableLike | str = "None",
    ) -> dict[str, DataHandle]:
        """The main interface method for Informers

        This will attach the input_data to this `Informer`
        (for introspection and provenance tracking).

        Then it will call the run(), validate() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the model that it creates to this Estimator
        by using `self.add_data('model', model)`.

        Finally, this will return a ModelHandle providing access to the trained model.

        Parameters
        ----------
        input_data
            Per-galaxy p(z), and any ancilary data associated with it

        truth_data
            Table with the true redshifts

        Returns
        -------
        dict[str, DataHandle]
            Handle providing access to trained model
        """
        self.set_data("input", training_data)
        self.set_data("truth", truth_data)
        self.validate()
        self.run()
        self.finalize()
        self._model_handle = self.get_handle("model")
        return dict(
            model=self._model_handle,
        )

    def run(self) -> None:
        iterator = self._setup_iterator()
        first = True
        self._initialize_run()
        self._output_handle = None
        for s, e, test_data, truth_data in iterator:
            print(f"Process {self.rank} running estimator on chunk {s:,} - {e:,}")
            self._process_chunk(s, e, test_data, truth_data, first)
            first = False
        self._finalize_run()

    def _initialize_run(self) -> None:
        self._model_handle = None

    def _finalize_run(self) -> None:
        assert self.model is not None
        self.add_data("model", self.model)
        self._model_handle = self.get_handle("model")        
        assert self._model_handle is not None        
        self._model_handle.write()

    def _process_chunk(
        self, start: int, end: int, data: qp.Ensemble, truth_data: np.ndarray, first: bool
    ) -> None:
        return
        
