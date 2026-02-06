"""Abstract base class defining a selector.

The key feature here is make selection to either the photometric or spectroscopic catalog. 
Intended subclasses spectroscopic selection, probability selection on a grid for the photometry, 
or pure photometric selection. 
"""

from ceci.config import StageParameter as Param

from rail.core.data import DataHandle, PqHandle, TableLike
from rail.core.stage import RailStage


class Selector(RailStage):
    """Base class Selector, which makes selection to the catalog

    Selector take "input" data in the form of pandas dataframes in Parquet
    files and provide as "output" another pandas dataframes written to Parquet
    files.
    """

    name = "Selector"
    config_options = RailStage.config_options.copy()
    config_options.update(
        drop_rows=Param(bool, True, msg="Drop selected rows from output table"),
        seed=Param(
            default=None,
            required=False,
            msg="Set to an `int` to force reproducible results.",
        ),
    )
    inputs = [("input", PqHandle)]
    outputs = [("output", PqHandle)]

    def __call__(self, sample: TableLike) -> DataHandle:
        """The main interface method for ``Selector``.

        Adds noise to the input catalog

        This will attach the input to this `Selector`

        Then it will call the select() which add a flag column to the catalog. flag=1 means
        selected, 0 means dropped.

        If dropRows = True, the dropped rows will not be presented in the output catalog,
        otherwise, all rows will be presented.

        Finally, this will return a PqHandle providing access to that output
        data.

        Parameters
        ----------
        sample:
            The sample to be selected

        Returns
        -------
        DataHandle
            A handle giving access to a table with selected sample
        """
        self.set_data("input", sample)
        self.run()
        self.finalize()
        return self.get_handle("output")

    def run(self) -> None:
        data = self.get_data("input")
        selection_mask = self._select()
        if self.config["drop_rows"]:
            out_data = data[selection_mask.astype(bool)]
        else:
            out_data = data.copy()
            out_data.insert(0, "flag", selection_mask)
        self.add_data("output", out_data)

    def _select(self) -> TableLike:  # pragma: no cover
        raise NotImplementedError("Selector._select()")
