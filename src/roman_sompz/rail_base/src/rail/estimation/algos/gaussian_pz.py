"""
A summarizer that simple makes a histogram of a point estimate
"""

from typing import Any, Generator

import numpy as np
import qp
from ceci.config import StageParameter as Param

from rail.core.data import DataHandle, QPHandle, TableHandle, TableLike
from rail.estimation.informer import PzInformer
from rail.estimation.estimator import PzEstimator


class GaussianPzInformer(PzInformer):
    """Placeholder Informer"""

    name = "GaussianPzInformer"
    config_options = PzInformer.config_options.copy()

    def _finalize_run(self) -> None:
        self.model = np.array([None])
        PzInformer._finalize_run(self)


class GaussianPzEstimator(PzEstimator):
    """Estimator which converts to Gaussian reps"""

    name = "GaussianPzEstimator"
    config_options = PzEstimator.config_options.copy()

    def _process_chunk(
        self, start: int, end: int, data: qp.Ensemble, first: bool
    ) -> None:

        mean = np.squeeze(data.mean())
        std = np.squeeze(data.std())
        std = np.where(np.isfinite(std), std, 0.01)
        
        qp_dstn = qp.Ensemble(qp.stats.norm, data=dict(loc=mean, scale=std))
        self._do_chunk_output(qp_dstn, start, end, first)

        
