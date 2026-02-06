from typing import Any

import qp


class MetricEvaluator:
    """A superclass for metrics evaluations"""

    def __init__(self, qp_ens: qp.Ensemble) -> None:
        """Class constructor.
        Parameters
        ----------
        qp_ens: qp.Ensemble object
            PDFs as qp.Ensemble
        """
        self._qp_ens = qp_ens

    def evaluate(self) -> Any:  # pragma: no cover
        """
        Evaluates the metric a function of the truth and prediction

        Returns
        -------
        metric: dictionary
            value of the metric and statistics thereof
        """
        raise NotImplementedError
