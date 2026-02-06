"""Abstract base class defining a degrader.

The key feature is that the __call__ method takes a pandas DataFrame and a seed,
and returns a pandas DataFrame, and wraps the run method.
"""

from ceci.config import StageParameter as Param

from rail.core.data import DataHandle, PqHandle, TableLike
from rail.core.stage import RailStage


class Degrader(RailStage):  # pragma: no cover
    """Base class Degraders, which apply various degradations to synthetic
    photometric data.

    Degraders take "input" data in the form of pandas dataframes in Parquet
    files and provide as "output" another pandas dataframes written to Parquet
    files.
    """

    name = "Degrader"
    config_options = RailStage.config_options.copy()
    config_options.update(
        seed=Param(int, default=12345, msg="Random number seed"),
    )
    inputs = [("input", PqHandle)]
    outputs = [("output", PqHandle)]

    def __call__(self, sample: TableLike, seed: int | None = None) -> DataHandle:
        """The main interface method for ``Degrader``.

        Applies degradation.

        This will attach the sample to this `Degrader` (for introspection and
        provenance tracking).

        Then it will call the run() and finalize() methods, which need to be
        implemented by the sub-classes.

        The run() method will need to register the data that it creates to this
        Estimator by using ``self.add_data('output', output_data)``.

        Finally, this will return a PqHandle providing access to that output
        data.

        Parameters
        ----------
        sample
            The sample to be degraded

        seed
            An integer to set the numpy random seed

        Returns
        -------
        DataHandle
            A handle giving access to a table with degraded sample
        """
        if seed is not None:
            self.config.seed = seed
        self.set_data("input", sample)
        self.run()
        self.finalize()
        return self.get_handle("output")
