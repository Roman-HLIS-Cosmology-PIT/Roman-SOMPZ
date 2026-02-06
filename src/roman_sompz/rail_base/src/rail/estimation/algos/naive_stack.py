"""
A summarizer that simple makes a histogram of a point estimate
"""

from typing import Any, Generator

import numpy as np
import qp
from ceci.config import StageParameter as Param

from rail.core.data import DataHandle, QPHandle, TableHandle, TableLike
from rail.estimation.informer import PzInformer
from rail.estimation.summarizer import PZSummarizer


class NaiveStackInformer(PzInformer):
    """Placeholder Informer"""

    name = "NaiveStackInformer"
    config_options = PzInformer.config_options.copy()

    def _finalize_run(self) -> None:
        self.model = np.array([None])
        PzInformer._finalize_run(self)


class NaiveStackSummarizer(PZSummarizer):
    """Summarizer which stacks individual P(z)"""

    name = "NaiveStackSummarizer"
    config_options = PZSummarizer.config_options.copy()
    config_options.update(
        zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
        zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
        nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
        seed=Param(int, 87, msg="random seed"),
        nsamples=Param(int, 1000, msg="Number of sample distributions to create"),
    )
    inputs = [("input", QPHandle)]
    outputs = [("output", QPHandle), ("single_NZ", QPHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        super().__init__(args, **kwargs)
        self.zgrid: np.ndarray | None = None

    def _setup_iterator(self) -> Generator:
        itr = self.input_iterator("input")
        for s, e, d in itr:
            yield s, e, d, np.ones(e - s, dtype=bool)

    def run(self) -> None:
        handle = self.get_handle("input", allow_missing=True)
        self._input_length = handle.size()
        iterator = self._setup_iterator()
        self.zgrid = np.linspace(
            self.config.zmin, self.config.zmax, self.config.nzbins + 1
        )
        assert self.zgrid is not None
        # Initiallizing the stacking pdf's
        yvals = np.zeros((1, len(self.zgrid)))
        bvals = np.zeros((self.config.nsamples, len(self.zgrid)))
        bootstrap_matrix = self._broadcast_bootstrap_matrix()

        first = True
        for s, e, test_data, mask in iterator:
            print(f"Process {self.rank} running estimator on chunk {s:,} - {e:,}")
            self._process_chunk(
                s, e, test_data, mask, first, bootstrap_matrix, yvals, bvals
            )
            first = False
        if self.comm is not None:  # pragma: no cover
            bvals, yvals = self._join_histograms(bvals, yvals)

        if self.rank == 0:
            sample_ens = qp.Ensemble(
                qp.interp, data=dict(xvals=self.zgrid, yvals=bvals)
            )
            qp_d = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=yvals))
            self.add_data("output", sample_ens)
            self.add_data("single_NZ", qp_d)

    def _process_chunk(
        self,
        start: int,
        end: int,
        data: qp.Ensemble,
        mask: np.ndarray,
        _first: bool,
        bootstrap_matrix: np.ndarray,
        yvals: np.ndarray,
        bvals: np.ndarray,
    ) -> None:
        assert self.zgrid is not None
        pdf_vals = data.pdf(self.zgrid)
        squeeze_mask = np.squeeze(mask)
        yvals += np.expand_dims(
            np.sum(
                np.where(
                    np.isfinite(pdf_vals[squeeze_mask, :]), pdf_vals[squeeze_mask], 0.0
                ),
                axis=0,
            ),
            0,
        )
        # qp_d is the normalized probability of the stack, we need to know how many galaxies were
        for i in range(self.config.nsamples):
            bootstrap_draws = bootstrap_matrix[:, i]
            # Neither all of the bootstrap_draws are in this chunk nor the index starts at "start"
            chunk_mask = (bootstrap_draws >= start) & (bootstrap_draws < end)
            bootstrap_draws = bootstrap_draws[chunk_mask] - start
            zarr = np.where(squeeze_mask, pdf_vals.T, 0.0).T[bootstrap_draws]
            bvals[i] += np.sum(zarr, axis=0)


class NaiveStackMaskedSummarizer(NaiveStackSummarizer):
    name = "NaiveStackMaskedSummarizer"
    config_options = NaiveStackSummarizer.config_options.copy()
    config_options.update(
        selected_bin=Param(int, -1, msg="bin to use"),
    )
    inputs = [("input", QPHandle), ("tomography_bins", TableHandle)]
    outputs = [("output", QPHandle), ("single_NZ", QPHandle)]

    def _setup_iterator(self) -> Generator:
        selected_bin = self.config.selected_bin
        if self.config.tomography_bins in ["none", None]:
            selected_bin = -1

        if selected_bin == -1:
            itrs = [self.input_iterator("input")]
        else:
            itrs = [
                self.input_iterator("input"),
                self.input_iterator("tomography_bins"),
            ]

        for it in zip(*itrs):
            first = True
            mask = None
            for s, e, d in it:
                if first:
                    start = s
                    end = e
                    pz_data = d
                    first = False
                else:
                    mask = d["class_id"] == self.config.selected_bin
            if mask is None:
                mask = np.ones(pz_data.npdf, dtype=bool)
            yield start, end, pz_data, mask

    def summarize(
        self, input_data: qp.Ensemble, tomo_bins: TableLike | None = None
    ) -> DataHandle:
        """Override the Summarizer.summarize() method to take tomo bins
        as an additional input

        Parameters
        ----------
        input_data
            Per-galaxy p(z), and any ancilary data associated with it

        tomo_bins
            Tomographic bins file

        Returns
        -------
        DataHandle
            Ensemble with n(z), and any ancilary data
        """
        self.set_data("input", input_data)
        if tomo_bins is None:
            self.config.tomography_bins = None
        else:
            self.set_data("tomography_bins", tomo_bins)
        self.run()
        self.finalize()
        return self.get_handle("output")
