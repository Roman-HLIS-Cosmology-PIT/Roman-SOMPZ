import numpy as np
from scipy import stats
from ceci.config import StageParameter as Param

from rail.core.common_params import SHARED_PARAMS
from rail.core.data import TableLike, DataHandle, Hdf5Handle, TableHandle
from rail.core.stage import RailStage


class KDEBinOverlap(RailStage):
    name = "KDEBinOverlap"
    inputs = [("truth", TableHandle), ("bin_index", Hdf5Handle)]
    outputs = [("output", Hdf5Handle)]

    config_options = RailStage.config_options.copy()
    config_options.update(
        hdf5_groupname=Param(
            str, "", required=False, msg="HDF5 Groupname for truth table."
        ),
        redshift_col=SHARED_PARAMS,
        bin_name=Param(
            str,
            "class_id",
            required=False,
            msg="Groupname for the tomographic bin index in the hdf5 handle",
        ),
    )
    # metric_base_class = Evaluator

    def evaluate(self, bin_index: TableLike, truth: TableLike) -> DataHandle:
        self.set_data("bin_index", bin_index)
        self.set_data("truth", truth)

        self.run()
        return self.get_handle("output")

    def run(self) -> None:
        if self.config.hdf5_groupname is None: # pragma: no cover
            true_redshifts = self.get_handle("truth").data[
                self.config.redshift_col
            ]  # 1D array of redshifts
        else: 
            true_redshifts = self.get_handle("truth").data[self.config.hdf5_groupname][
                self.config.redshift_col
            ]  # 1D array of redshifts
        bin_indices = self.get_handle("bin_index").data[
            self.config.bin_name
        ]  # 1D array of bin indices

        unique_bins = np.unique(bin_indices)
        unique_bins = unique_bins[unique_bins >= 0]  # to exclude the -99
        N = len(unique_bins)
        overlap_matrix = np.zeros((N, N))

        kde_dict = {}
        for bin_id in unique_bins:
            redshifts_in_bin = true_redshifts[bin_indices == bin_id]
            if len(redshifts_in_bin) > 1:
                kde = stats.gaussian_kde(redshifts_in_bin)
                kde_dict[bin_id] = kde

        for i, bin_i in enumerate(unique_bins):
            overlap_matrix[i][i] = 1.0
            for j, bin_j in enumerate(unique_bins):
                if j >= i:
                    continue

                kde_i = kde_dict.get(bin_i, None)
                kde_j = kde_dict.get(bin_j, None)

                if kde_i is not None and kde_j is not None:
                    # Define the evaluation grid based on the union of both bin samples
                    eval_grid = np.linspace(
                        min(true_redshifts), max(true_redshifts), 1000
                    )
                    p_i = kde_i(eval_grid)
                    p_j = kde_j(eval_grid)

                    # Compute overlap as the integral of the minimum of both distributions
                    overlap = np.trapz(np.minimum(p_i, p_j), eval_grid)
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap  # Symmetric matrix

        self.add_handle("output", data=overlap_matrix)
