"""Abstract base classes defining a Creator.

A Creator will create synthetic photometric data and a PosteriorCalculator, 
which can calculate posteriors for the data with respect to the distribution 
defined by the Creator.
"""

from typing import Any

from ceci.config import StageParameter as Param

from rail.core.data import (DataHandle, ModelHandle, ModelLike, QPHandle,
                            TableHandle, TableLike)
from rail.core.stage import RailStage


class Modeler(RailStage):  # pragma: no cover
    """Base class for creating a model of redshift and photometry."""

    name = "Modeler"
    config_options = RailStage.config_options.copy()
    config_options.update(
        seed=Param(int, default=12345, msg="Random number seed"),
    )
    inputs = [("input", DataHandle)]
    outputs = [("model", ModelHandle)]

    def __init__(self, args: Any, **kwargs: Any):
        """Initialize Modeler"""
        super().__init__(args, **kwargs)
        self.model = None

    def fit_model(self) -> ModelHandle:
        """Produce a creation model from which photometry and redshifts can be
        generated.

        Returns
        -------
        ModelHandle:
            This will definitely be a wrapper around a File,
            but the filetype and format depend entirely on the
            modeling approach
        """
        self.validate()
        self.run()
        self.finalize()
        return self.get_handle("model")


class Creator(RailStage):  # pragma: no cover
    """Base class for Creators that generate synthetic photometric data from a
    model.

    ``Creator`` will output a table of photometric data. The details will depend
    on the particular engine.
    """

    name = "Creator"
    config_options = RailStage.config_options.copy()
    config_options.update(
        n_samples=Param(int, required=True, msg="Number of samples to create"),
        seed=Param(int, default=12345, msg="Random number seed"),
    )
    inputs = [("model", ModelHandle)]
    outputs = [("output", TableHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Initialize Creator"""
        super().__init__(args, **kwargs)
        self.model = None
        if not isinstance(args, dict):  # pragma: no cover
            args = vars(args)
        self.open_model(**args)

    def sample(
        self, n_samples: int, seed: int | None = None, **kwargs: Any
    ) -> DataHandle:
        """Draw samples from the model specified in the configuration.

        This is a method for running a Creator in interactive mode. In pipeline
        mode, the subclass ``run`` method will be called by itself.

        Parameters
        ----------
        n_samples
            The number of samples to draw

        seed
            The random seed to control sampling

        **kwargs:
            Used to update the configuration

        Returns
        -------
        DataHandle
            DataHandle wrapping the newly created samples

        Notes
        -----
        This method puts ``n_samples`` and ``seed`` into the stage configuration
        data, which makes them available to other methods.

        It then calls the ``run`` method, which must be defined by a subclass.

        Finally, the ``DataHandle`` associated to the ``output`` tag is returned.
        """
        self.config["n_samples"] = n_samples
        self.config["seed"] = seed
        self.config.update(**kwargs)
        self.run()
        self.finalize()
        return self.get_handle("output")


class PosteriorCalculator(RailStage):  # pragma: no cover
    """Base class for object that calculates the posterior distribution of a
    particular field in a table of photometric data (typically the redshift).

    The posteriors will be contained in a qp Ensemble.
    """

    name = "PosteriorCalculator"
    config_options = RailStage.config_options.copy()
    config_options.update(
        column=Param(str, required=True, msg="Column to compute posterior for"),
    )
    inputs = [
        ("model", ModelHandle),
        ("input", TableHandle),
    ]
    outputs = [("output", QPHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Initialize PosteriorCalculator"""
        super().__init__(args, **kwargs)
        self.model = None
        if not isinstance(args, dict):  # pragma: no cover
            args = vars(args)
        self.open_model(**args)

    def get_posterior(self, input_data: TableLike, **kwargs: Any) -> DataHandle:
        """Return posteriors for the given column.

        This is a method for running a Creator in interactive mode. In pipeline
        mode, the subclass ``run`` method will be called by itself.

        Parameters
        ----------
        data
            A table of the galaxies for which posteriors are calculated

        **kwargs
            Used to update configuration

        Returns
        -------
        DataHandle
            Posterior Estimate

        Notes
        -----
        This will put the ``data`` argument input this Stages the DataStore
        using this stages ``input`` tag.

        This will put the additional functional arguments into this Stages
        configuration data.

        It will then call ``self.run()`` and return the ``DataHandle``
        associated to the ``output`` tag.
        """
        self.set_data("input", input_data)
        self.config.update(**kwargs)
        self.run()
        self.finalize()
        return self.get_handle("output")
