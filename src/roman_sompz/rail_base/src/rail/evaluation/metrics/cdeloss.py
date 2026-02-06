import numpy as np
import qp

from rail.evaluation.stats_groups import stat_and_pval

from .base import MetricEvaluator


class CDELoss(MetricEvaluator):
    """Conditional density loss"""

    def __init__(
        self, qp_ens: qp.Ensemble, zgrid: np.ndarray, ztrue: np.ndarray
    ) -> None:
        """Class constructor"""
        super().__init__(qp_ens)

        self._pdfs = qp_ens.pdf(zgrid)
        self._xvals = zgrid
        self._ztrue = ztrue
        self._npdf = qp_ens.npdf

    def evaluate(self) -> stat_and_pval:
        """Evaluate the estimated conditional density loss described in
        Izbicki & Lee 2017 (arXiv:1704.08095).

        Notes
        -----
        """

        # Calculate first term E[\int f*(z | X)^2 dz]
        term1 = np.mean(np.trapz(self._pdfs**2, x=self._xvals))
        # z bin closest to ztrue
        nns = [np.argmin(np.abs(self._xvals - z)) for z in self._ztrue]
        # Calculate second term E[f*(Z | X)]
        term2 = np.mean(self._pdfs[range(self._npdf), nns])
        cdeloss = term1 - 2 * term2
        return stat_and_pval(cdeloss, np.nan)
