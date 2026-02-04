"""
Port of SOMPZ
"""
import os
import numpy as np
import qp
from ceci.config import StageParameter as Param
from rail.core.data import TableHandle, ModelHandle, QPHandle, Hdf5Handle
from rail.estimation.estimator import CatEstimator, CatInformer
import rail.estimation.algos.som as somfuncs
from rail.core.common_params import SHARED_PARAMS
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import pickle
import gc


class Pickableclassify:  # pragma: no cover
    def __init__(self, som, flux, fluxerr, inds):
        self.som = som
        self.flux = flux
        self.inds = inds
        self.flux_err = fluxerr

    def __call__(self, ind):
        cells_test, dist_test = self.som.classify(self.flux[self.inds[ind]], self.flux_err[self.inds[ind]])
        return cells_test, dist_test


def_bands = ["u", "g", "r", "i", "z", "y"]
default_bin_edges = [0.0, 0.405, 0.665, 0.96, 2.0]
default_input_names = []
default_err_names = []
default_zero_points = []
for band in def_bands:
    default_input_names.append(f"mag_{band}_lsst")
    default_err_names.append(f"mag_err_{band}_lsst")
    default_zero_points.append(30.)


def mag2flux(mag, zero_pt=30):
    # zeropoint: M = 30 <=> f = 1
    exponent = (mag - zero_pt) / (-2.5)
    val = 1 * 10 ** (exponent)
    return val


def magerr2fluxerr(magerr, flux):
    coef = np.log(10) / -2.5
    return np.abs(coef * magerr * flux)


def calculate_pcchat(deep_som_size, wide_som_size, cell_deep_assign, cell_wide_assign, overlap_weight):
    pcchat_num = np.zeros((deep_som_size, wide_som_size))
    np.add.at(pcchat_num,
              (cell_deep_assign, cell_wide_assign),
              overlap_weight)

    pcchat_denom = pcchat_num.sum(axis=0)
    pcchat = pcchat_num / pcchat_denom[None]

    # any nonfinite in pcchat are to be treated as 0 probabilty
    pcchat = np.where(np.isfinite(pcchat), pcchat, 0)

    return pcchat


def get_deep_histograms(data, deep_data, key, cells, overlap_weighted_pzc, bins, overlap_key='overlap_weight',
                        deep_som_size=64 * 64, deep_map_shape=(64 * 64,), interpolate_kwargs={}):
    """Return individual deep histograms for each cell. Can interpolate for empty cells.

    Parameters
    ----------
    deep_data             : cosmos data used here for Y3
    key                   : Parameter to extract from dataframe
    cells                 : A list of deep cells to return sample from, or a single int.
    overlap_weighted_pzc  : Use overlap_weights in p(z|c) histogram if True. Also required if you want to bin conditionalize
    overlap_key           : column name for the overlap weights in the dataframe, default to 'overlap_weight'
    bins                  : Bins we histogram the values into
    interpolate_kwargs    : arguments to pass in for performing interpolation between cells for redshift hists using a 2d gaussian of sigma scale_length out to max_length cells away.
    The two kwargs are    : 'scale_length' and 'max_length'
    Returns
    -------
    hists : a histogram of the values from self.data[key] for each deep cell
    """

    if len(interpolate_kwargs) > 0:  # pragma: no cover
        cells_keep = cells
        cells = np.arange(deep_som_size)
    else:
        cells_keep = cells

    hists = []
    missing_cells = []
    populated_cells = []

    for ci, c in enumerate(cells):
        try:
            df = deep_data.groupby('cell_deep').get_group(c)
            if type(key) is str:
                z = df[key].values
                if overlap_weighted_pzc:  # pragma: no cover
                    # print("WARNING: You are using a deprecated point estimate Z. No overlap weighting enabled.
                    # You're on your own now.")#suppress
                    weights = df[overlap_key].values
                else:
                    weights = np.ones(len(z))
                hist = np.histogram(z, bins, weights=weights, density=True)[
                    0]  # make weighted histogram by overlap weights
                populated_cells.append([ci, c])
            elif type(key) is list:  # pragma: no cover
                # use full p(z)
                assert (bins is not None)
                # ##histogram_from_fullpz CURRENTLY UNDEFINED!
                hist = histogram_from_fullpz(df, key, overlap_weighted=overlap_weighted_pzc, bin_edges=bins)
            hists.append(hist)
        except KeyError as e:
            missing_cells.append([ci, c])
            hists.append(np.zeros(len(bins) - 1))
    hists = np.array(hists)

    if len(interpolate_kwargs) > 0:  # pragma: no cover
        # print('Interpolating {0} missing histograms'.format(len(missing_cells)))
        missing_cells = np.array(missing_cells)
        populated_cells = np.array(populated_cells)
        hist_conds = np.isin(cells, populated_cells[:, 1]) & np.all(np.isfinite(hists), axis=1)
        for ci, c in missing_cells:
            if c not in cells_keep:
                # don't worry about interpolating cells we won't use anyways
                continue

            central_index = np.zeros(len(deep_map_shape), dtype=int)
            # unravel_index(c, deep_map_shape, central_index)  # fills central_index
            cND = np.zeros(len(deep_map_shape), dtype=int)
            weight_map = np.zeros(deep_som_size)
            # gaussian_rbf(weight_map, central_index, cND, deep_map_shape, **interpolate_kwargs)  # fills weight_map
            hists[ci] = np.sum(hists[hist_conds] * (weight_map[hist_conds] / weight_map[hist_conds].sum())[:, None],
                               axis=0)

        # purge hists back to the ones we care about
        hists = hists[cells_keep]

    return hists


def histogram(data, deep_data, key, cells, cell_weights, pcchat, overlap_weighted_pzc, deep_som_size=64 * 64, bins=None,
              individual_chat=False, interpolate_kwargs={}):
    """Return histogram from values that live in specified wide cells by querying deep cells that contribute

    Parameters
    ----------
    key                  : Parameter(s) to extract from dataframe
    cells                : A list of wide cells to return sample from, or a single int.
    cell_weights         : How much we weight each wide cell. This is the array p(chat | sample)
    overlap_weighted_pzc : Weight contribution of galaxies within c by overlap_weight, if True. Weighting for p(c|chat) is done using stored transfer matrix.
    bins                 : Bins we histogram the values into
    individual_chat      : If True, compute p(z|chat) for each individual cell in cells. If False, compute a single p(z|{chat}) for all cells.
    interpolate_kwargs   : arguments to pass in for performing interpolation between cells for redshift hists using a 2d gaussian of sigma scale_length out to max_length cells away. The two kwargs are: 'scale_length' and 'max_length'

    Returns
    -------
    hist : a histogram of the values from self.data[key]

    Notes
    -----
    This method tries to marginalize wide assignments into what deep assignments it has

    """
    # get sample, p(z|c)
    all_cells = np.arange(deep_som_size)
    hists_deep = get_deep_histograms(data, deep_data, key=key, cells=all_cells,
                                     overlap_weighted_pzc=overlap_weighted_pzc,
                                     bins=bins, interpolate_kwargs=interpolate_kwargs)
    if individual_chat:  # then compute p(z|chat) for each individual cell in cells and return histograms
        hists = []
        for i, (cell, cell_weight) in enumerate(zip(cells, cell_weights)):
            # p(c|chat,s)p(chat|s) = p(c,chat|s)
            possible_weights = pcchat[:, [cell]] * np.array([cell_weight])[None]  # (n_deep_cells, 1)
            # sum_chat p(c,chat|s) = p(c|s)
            weights = np.sum(possible_weights, axis=-1)
            conds = (weights != 0) & np.all(np.isfinite(hists_deep), axis=1)
            # sum_c p(z|c) p(c|s) = p(z|s)
            hist = np.sum((hists_deep[conds] * weights[conds, None]), axis=0)

            dx = np.diff(bins)
            normalization = np.sum(dx * hist)
            if normalization != 0:
                hist = hist / normalization
            hists.append(hist)
        return hists
    else:  # compute p(z|{chat}) and return histogram
        # p(c|chat,s)p(chat|s) = p(c,chat|s)
        possible_weights = pcchat[:, cells] * cell_weights[None]  # (n_deep_cells, n_cells)
        # sum_chat p(c,chat|s) = p(c|s)
        weights = np.sum(possible_weights, axis=-1)
        conds = (weights != 0) & np.all(np.isfinite(hists_deep), axis=1)
        # sum_c p(z|c) p(c|s) = p(z|s)
        hist = np.sum((hists_deep[conds] * weights[conds, None]), axis=0)

        dx = np.diff(bins)
        normalization = np.sum(dx * hist)
        if normalization != 0:
            hist = hist / normalization
        return hist


def histogram_from_fullpz(df, key, overlap_weighted, bin_edges, full_pz_end=6.00, full_pz_npts=601):  # pragma: no cover
    """Preserve bins from Laigle"""
    dz_laigle = full_pz_end / (full_pz_npts - 1)
    condition = np.sum(~np.equal(bin_edges, np.arange(0 - dz_laigle / 2.,
                                                      full_pz_end + dz_laigle,
                                                      dz_laigle)))
    assert condition == 0

    single_cell_hists = np.zeros((len(df), len(key)))

    overlap_weights = np.ones(len(df))
    if overlap_weighted:
        overlap_weights = df['overlap_weight'].values

    single_cell_hists[:, :] = df[key].values

    # normalize sompz p(z) to have area 1
    dz = 0.01
    area = np.sum(single_cell_hists, axis=1) * dz
    area[area == 0] = 1  # some galaxies have pz with only one non-zero point. set these galaxies' histograms to have
    # area 1
    area = area.reshape(area.shape[0], 1)
    single_cell_hists = single_cell_hists / area

    # response weight normalized p(z)
    single_cell_hists = np.multiply(overlap_weights, single_cell_hists.transpose()).transpose()

    # sum individual galaxy p(z) to single cell p(z)
    hist = np.sum(single_cell_hists, axis=0)

    # renormalize p(z|c)
    area = np.sum(hist) * dz
    hist = hist / area

    return hist


def redshift_distributions_wide(data,
                                deep_data,
                                overlap_weighted_pchat,
                                overlap_weighted_pzc,
                                bins,
                                pcchat,
                                deep_som_size=64 * 64,
                                tomo_bins={},
                                key='Z',
                                force_assignment=True,
                                interpolate_kwargs={}, **kwargs):
    """Returns redshift distribution for sample

    Parameters
    ----------
    data :  Data sample of interest with wide data
    deep_data: cosmos data
    overlap_weighted_pchat  : If True, use overlap weights for p(chat)
    overlap_weighted_pzc : If True, use overlap weights for p(z|c)
                Note that whether p(c|chat) is overlap weighted depends on how you built pcchat earlier.
    bins :      bin edges for redshift distributions data[key]
    tomo_bins : Which cells belong to which tomographic bins. First column is
                cell id, second column is an additional reweighting of galaxies in cell.
                If nothing is passed in, then we by default just use all cells
    key :       redshift key
    force_assignment : Calculate cell assignments. If False, then will use whatever value is in the cell_key field of data. Default: True
    interpolate_kwargs : arguments to pass in for performing interpolation
    between cells for redshift hists using a 2d gaussian of sigma
    scale_length out to max_length cells away. The two kwargs are:
    'scale_length' and 'max_length'

    Returns
    -------
    hists : Either a single array (if no tomo_bins) or multiple arrays

    """
    if len(tomo_bins) == 0:  # pragma: no cover
        cells, cell_weights = get_cell_weights_wide(data, overlap_weighted_pchat=overlap_weighted_pchat,
                                                    force_assignment=force_assignment, **kwargs)
        if cells.size == 0:
            hist = np.zeros(len(bins) - 1)
        else:
            hist = histogram(data, deep_data, key=key, cells=cells, cell_weights=cell_weights,
                             overlap_weighted_pzc=overlap_weighted_pzc, bins=bins,
                             interpolate_kwargs=interpolate_kwargs)
        return hist
    else:
        cells, cell_weights = get_cell_weights_wide(data, overlap_weighted_pchat,
                                                    force_assignment=force_assignment, **kwargs)
        cellsort = np.argsort(cells)
        cells = cells[cellsort]
        cell_weights = cell_weights[cellsort]

        # break up hists into the different bins
        hists = []
        for tomo_key in tomo_bins:
            cells_use = tomo_bins[tomo_key][:, 0]
            cells_binweights = tomo_bins[tomo_key][:, 1]
            cells_conds = np.searchsorted(cells, cells_use, side='left')
            if len(cells_conds) == 0:  # pragma: no cover
                hist = np.zeros(len(bins) - 1)
            else:
                hist = histogram(data, deep_data, key=key, cells=cells[cells_conds],
                                 cell_weights=cell_weights[cells_conds] * cells_binweights,
                                 pcchat=pcchat,
                                 deep_som_size=deep_som_size,
                                 overlap_weighted_pzc=overlap_weighted_pzc,
                                 bins=bins,
                                 interpolate_kwargs=interpolate_kwargs)
            hists.append(hist)
        hists = np.array(hists)
        return hists


def get_cell_weights(data, overlap_weighted, key):
    """Given data, get cell weights and indices

    Parameters
    ----------
    data :  Dataframe we extract parameters from
    overlap_weighted : If True, use mean overlap weights of cells.
    key :   Which key we are grabbing

    Returns
    -------
    cells :         The names of the cells
    cell_weights :  The fractions of the cells
    """
    if overlap_weighted:  # pragma: no cover
        cws = data.groupby(key)['overlap_weight'].sum()
    else:
        cws = data.groupby(key).size()

    cells = cws.index.values.astype(int)
    cws = cws / cws.sum()

    cell_weights = cws.values
    return cells, cell_weights


def get_cell_weights_wide(data, overlap_weighted_pchat, cell_key='cell_wide', force_assignment=False, **kwargs):
    """Given data, get cell weights p(chat) and indices from wide SOM

    Parameters
    ----------
    data             : Dataframe we extract parameters from
    overlap_weighted_pchat : If True, use mean overlap weights of wide cells in p(chat)
    cell_key         : Which key we are grabbing. Default: cell_wide
    force_assignment : Calculate cell assignments. If False, then will use whatever value is in the cell_key field of data. Default: True

    Returns
    -------
    cells        :  The names of the cells
    cell_weights :  The fractions of the cells
    """
    # if force_assignment:
    #     data[cell_key] = self.assign_wide(data, **kwargs)
    return get_cell_weights(data, overlap_weighted_pchat, cell_key)


def bin_assignment_spec(spec_data, deep_som_size, wide_som_size, bin_edges,
                        key_z='Z', key_cells_wide='cell_wide_unsheared'):
    # assign gals in redshift sample to bins
    xlabels = []
    nbins = len(bin_edges) - 1
    for ii in range(nbins):
        xlabels.append(ii)
    spec_data['tomo_bin'] = pd.cut(spec_data[key_z], bin_edges, labels=xlabels)

    ncells_with_spec_data = len(np.unique(spec_data[key_cells_wide].values))
    cell_bin_assignment = np.ones(wide_som_size, dtype=int) * -1
    cells_with_spec_data = np.unique(spec_data[key_cells_wide].values)

    groupby_obj_value_counts = spec_data.groupby(key_cells_wide)['tomo_bin'].value_counts()

    for c in cells_with_spec_data:
        bin_assignment = groupby_obj_value_counts.loc[c].index[0]
        cell_bin_assignment[c] = bin_assignment

    # reformat bins into dict
    tomo_bins_wide = {}
    nbins = len(bin_edges) - 1
    for i in range(nbins):
        tomo_bins_wide[i] = np.where(cell_bin_assignment == i)[0]

    return tomo_bins_wide


def tomo_bins_wide_2d(tomo_bins_wide_dict):
    tomo_bins_wide = tomo_bins_wide_dict.copy()
    for k in tomo_bins_wide:
        if tomo_bins_wide[k].ndim == 1:
            tomo_bins_wide[k] = np.column_stack((tomo_bins_wide[k], np.ones(len(tomo_bins_wide[k]))))
        renorm = 1. / np.average(tomo_bins_wide[k][:, 1])
        tomo_bins_wide[k][:, 1] *= renorm  # renormalize so the mean weight is 1; important for bin conditioning
    return tomo_bins_wide


def plot_nz(hists, zbins, outfile, xlimits=(0, 2), ylimits=(0, 3.25)):  # pragma: no cover
    plt.figure(figsize=(16., 9.))
    for i in range(len(hists)):
        plt.plot((zbins[1:] + zbins[:-1]) / 2., hists[i], label='bin ' + str(i))
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$p(z)$')
    plt.legend()
    plt.title('n(z)')
    plt.savefig(outfile)
    plt.close()


class SOMPZInformer(CatInformer):
    """Inform stage for SOMPZEstimator
    """
    name = "SOMPZInformer"
    config_options = CatInformer.config_options.copy()
    config_options.update(redshift_col=SHARED_PARAMS,
                          #hdf5_groupname=None,
                          nproc=Param(int, 1, msg="number of processors to use"),
                          # groupname=Param(str, "photometry", msg="hdf5_groupname for ata"),
                          inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for data"),
                          input_errs=Param(list, default_err_names, msg="list of the names of columns containing errors on inputs for data"),
                          zero_points=Param(list, default_zero_points, msg="zero points for converting mags to fluxes for data, if needed"),
                          som_shape=Param(list, [32, 32], msg="shape for the som, must be a 2-element tuple"),
                          som_minerror=Param(float, 0.01, msg="floor placed on observational error on each feature in som"),
                          som_wrap=Param(bool, False, msg="flag to set whether the SOM has periodic boundary conditions"),
                          som_take_log=Param(bool, True, msg="flag to set whether to take log of inputs (i.e. for fluxes) for som"),
                          convert_to_flux=Param(bool, False, msg="flag for whether to convert input columns to fluxes for data"
                                                "set to true if inputs are mags and to False if inputs are already fluxes"),
                          set_threshold=Param(bool, False, msg="flag for whether to replace values below a threshold with a set number"),
                          thresh_val=Param(float, 1.e-5, msg="threshold value for set_threshold for data"),
                          thresh_val_err=Param(float, 1.e-5, msg="threshold value for set_threshold for data error"))

    inputs = [('input_data', TableHandle),
              ]
    outputs = [('model', ModelHandle),
               ]

    def run(self):
        
        # note: hdf5_groupname is a SHARED_PARAM defined in the parent class!
        data = self.get_data('input_data').to_pandas()                 
        num_inputs = len(self.config.inputs)
        ngal = len(data[self.config.inputs[0]])
        print(f"{ngal} galaxies in sample")

        d_input = np.zeros([ngal, num_inputs])
        d_errs = np.zeros([ngal, num_inputs])

        # assemble data
        for i, (col, errcol) in enumerate(zip(self.config.inputs, self.config.input_errs)):
            if self.config.convert_to_flux:
                d_input[:, i] = mag2flux(data[col], self.config.zero_points[i])
                d_errs[:, i] = magerr2fluxerr(data[errcol], d_input[:, i])
            else:  # pragma: no cover
                d_input[:, i] = data[col]
                d_errs[:, i] = data[errcol]

        # put a temporary threshold bit in. TODO fix this up later...
        if self.config.set_threshold:
            for i in range(num_inputs):
                mask = (d_input[:, i] < self.config.thresh_val)
                d_input[:, i][mask] = self.config.thresh_val
                errmask = (d_errs[:, i] < self.config.thresh_val_err)
                d_errs[:, i][errmask] = self.config.thresh_val_err

        sommetric = somfuncs.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
        learn_func = somfuncs.hFunc(ngal, sigma=(30, 1))

        # if 'pool' in self.config.keys():
        #     self.pool, self.nprocess = self.config["pool"]
        # else:
        #     print("pool not specified, setting pool to None")
        #     self.pool = None
        #     self.nprocess = 0
        #     self.config.pool = (None, 1)
        pool = Pool(self.config.nproc)
        nproc = self.config.nproc
        pooltuple = (pool, nproc)

        print(f"Training SOM of shape {self.config.som_shape}...", flush=True)

        som = somfuncs.NoiseSOM(sommetric, d_input, d_errs, learn_func,
                                shape=self.config.som_shape, minError=self.config.som_minerror,
                                wrap=self.config.som_wrap, logF=self.config.som_take_log, pool=pooltuple)
        model = dict(som=som, columns=self.config.inputs,
                     err_columns=self.config.input_errs)
        self.add_data('model', model)

    def inform(self, input_data):
        self.set_data('input_data', input_data)
        self.run()
        self.finalize()
        return self.model


class SOMPZEstimator(CatEstimator):  # pragma: no cover
    """CatEstimator subclass to compute redshift PDFs for SOMPZ
    """
    name = "SOMPZEstimator"
    config_options = CatEstimator.config_options.copy()
    config_options.update(redshift_col=SHARED_PARAMS,
                          bin_edges=Param(list, default_bin_edges, msg="list of edges of tomo bins"),
                          zbins_min=Param(float, 0.0, msg="minimum redshift for output grid"),
                          zbins_max=Param(float, 6.0, msg="maximum redshift for output grid"),
                          zbins_dz=Param(float, 0.01, msg="delta z for defining output grid"),
                          # data_path=Param(str, "directory", msg="directory for output files"),
                          spec_groupname=Param(str, "", msg="hdf5_groupname for spec_data"),
                          balrog_groupname=Param(str, "", msg="hdf5_groupname for balrog_data"),
                          wide_groupname=Param(str, "", msg="hdf5_groupname for wide_data"),
                          specz_name=Param(str, "redshift", msg="column name for true redshift in specz sample"),
                          inputs_deep=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          input_errs_deep=Param(list, default_err_names, msg="list of the names of columns containing errors on inputs for deep data"),
                          inputs_wide=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for wide data"),
                          input_errs_wide=Param(list, default_err_names, msg="list of the names of columns containing errors on inputs for wide data"),
                          zero_points_deep=Param(list, default_zero_points, msg="zero points for converting mags to fluxes for deep data, if needed"),
                          zero_points_wide=Param(list, default_zero_points, msg="zero points for converting mags to fluxes for wide data, if needed"),
                          som_shape_deep=Param(list, [32, 32], msg="shape for the deep som, must be a 2-element tuple"),
                          som_shape_wide=Param(list, [32, 32], msg="shape for the wide som, must be a 2-element tuple"),
                          som_minerror_deep=Param(float, 0.01, msg="floor placed on observational error on each feature in deep som"),
                          som_minerror_wide=Param(float, 0.01, msg="floor placed on observational error on each feature in wide som"),
                          som_wrap_deep=Param(bool, False, msg="flag to set whether the deep SOM has periodic boundary conditions"),
                          som_wrap_wide=Param(bool, False, msg="flag to set whether the wide SOM has periodic boundary conditions"),
                          som_take_log_deep=Param(bool, True, msg="flag to set whether to take log of inputs (i.e. for fluxes) for deep som"),
                          som_take_log_wide=Param(bool, True, msg="flag to set whether to take log of inputs (i.e. for fluxes) for wide som"),
                          convert_to_flux_deep=Param(bool, False, msg="flag for whether to convert input columns to fluxes for deep data"
                                                     "set to true if inputs are mags and to False if inputs are already fluxes"),
                          convert_to_flux_wide=Param(bool, False, msg="flag for whether to convert input columns to fluxes for wide data"),
                          set_threshold_deep=Param(bool, False, msg="flag for whether to replace values below a threshold with a set number"),
                          thresh_val_deep=Param(float, 1.e-5, msg="threshold value for set_threshold for deep data"),
                          set_threshold_wide=Param(bool, False, msg="flag for whether to replace values below a threshold with a set number"),
                          thresh_val_wide=Param(float, 1.e-5, msg="threshold value for set_threshold for wide data"),
                          debug=Param(bool, False, msg="boolean reducing dataset size for quick debuggin"))

    inputs = [('deep_model', ModelHandle),
              ('wide_model', ModelHandle),
              ('spec_data', TableHandle),
              ('balrog_data', TableHandle),
              ('wide_data', TableHandle)]
    outputs = [('nz', QPHandle),
               ('spec_data_deep_assignment', Hdf5Handle),
               ('spec_data_wide_assignment', Hdf5Handle),
               ('balrog_data_deep_assignment', Hdf5Handle),
               ('balrog_data_wide_assignment', Hdf5Handle),
               ('wide_data_assignment', Hdf5Handle),
               ('pz_c', Hdf5Handle),
               ('pz_chat', Hdf5Handle),
               ('pc_chat', Hdf5Handle),
               ]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        if 'pool' in self.config.keys():
            self.pool, self.nprocess = self.config["pool"]
        else:
            self.pool = None
            self.nprocess = 0
        # check on bands, errs, and prior band
        if len(self.config.inputs_deep) != len(self.config.input_errs_deep):  # pragma: no cover
            raise ValueError("Number of inputs_deep specified in inputs_deep must be equal to number of mag errors specified in input_errs_deep!")
        if len(self.config.inputs_wide) != len(self.config.input_errs_wide):  # pragma: no cover
            raise ValueError("Number of inputs_wide specified in inputs_wide must be equal to number of mag errors specified in input_errs_wide!")

    def open_model(self, **kwargs):
        """Load the model and/or attach it to this Creator.

        Keywords
        --------
        model : object, str or ModelHandle
            Either an object with a trained model, a path pointing to a file
            that can be read to obtain the trained model, or a ``ModelHandle``
            providing access to the trained model

        Returns
        -------
        self.model : object
            The object encapsulating the trained model
        """
        deep_model = kwargs.get("deep_model", None)
        wide_model = kwargs.get("wide_model", None)
        if deep_model is None or deep_model == "None":  # pragma: no cover
            self.deep_model = None
        else:
            if isinstance(deep_model, str):  # pragma: no cover
                self.deep_model = self.set_data("deep_model", data=None, path=deep_model)
                self.config["deep_model"] = deep_model
            else:
                if isinstance(deep_model, ModelHandle):  # pragma: no cover
                    if deep_model.has_path:
                        self.config["deep_model"] = deep_model.path
                self.deep_model = self.set_data("deep_model", deep_model)

        if wide_model is None or wide_model == "None":  # pragma: no cover
            self.wide_model = None
        else:
            if isinstance(wide_model, str):  # pragma: no cover
                self.wide_model = self.set_data("wide_model", data=None, path=wide_model)
                self.config["wide_model"] = wide_model
            else:
                if isinstance(wide_model, ModelHandle):  # pragma: no cover
                    if wide_model.has_path:
                        self.config["wide_model"] = wide_model.path
                self.wide_model = self.set_data("wide_model", wide_model)
        return self.deep_model, self.wide_model

    def _assign_som(self, flux, flux_err, somstr):
        if somstr == 'deep':
            som_dim = self.config.som_shape_deep[0]
        elif somstr == 'wide':
            som_dim = self.config.som_shape_wide[0]

        # output_path = './'  # TODO make kwarg
        nTrain = flux.shape[0]
        # som_weights = np.load(infile_som, allow_pickle=True)
        if somstr == "deep":
            som_weights = self.deep_model['som'].weights
        elif somstr == "wide":
            som_weights = self.wide_model['som'].weights
        else:
            # assert (0)
            raise ValueError(f"valid SOM values are 'deep' and 'wide', {somstr} is not valid")
        hh = somfuncs.hFunc(nTrain, sigma=(30, 1))
        metric = somfuncs.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
        som = somfuncs.NoiseSOM(metric, None, None,
                                learning=hh,
                                shape=(som_dim, som_dim),
                                wrap=False, logF=True,
                                initialize=som_weights,
                                minError=0.02)
        subsamp = 1

        # Now we classify the objects into cells and save these cells
        if self.pool is not None:
            inds = np.array_split(np.arange(len(flux)), self.nprocess)
            pickableclassify = Pickableclassify(som, flux, flux_err, inds)
            result = self.pool.map(pickableclassify, range(self.nprocess))
            cells_test = np.concatenate([r[0] for r in result])
            dist_test = np.concatenate([r[1] for r in result])
            del pickableclassify
        else:
            cells_test, dist_test = som.classify(flux[::subsamp, :], flux_err[::subsamp, :])

        # take out numpy savez
        # outfile = os.path.join(output_path, "som_{0}_{1}x{1}_assign.npz".format(somstr,som_dim))
        # np.savez(outfile, cells=cells_test, dist=dist_test)

        return cells_test, dist_test

    def _estimate_pdf(self,):
        zbins = np.arange(self.config.zbins_min - self.config.zbins_dz / 2.,
                          self.config.zbins_max + self.config.zbins_dz,
                          self.config.zbins_dz)
        self.bincents = 0.5 * (zbins[1:] + zbins[:-1])
        # TODO: improve file i/o
        # output_path = './'
        deep_som_size = np.product(self.deep_model['som'].shape)
        wide_som_size = np.product(self.wide_model['som'].shape)

        all_deep_cells = np.arange(deep_som_size)
        # key = 'specz_redshift'
        key = self.config.specz_name

        # load spec_data to access redshifts to histogram

        # TODO: this code block is repeated in run. refactor to avoid repeating
        if self.config.spec_groupname:
            spec_data = self.get_data('spec_data')[self.config.spec_groupname]
        else:  # pragma: no cover
            # DEAL with hdf5_groupname stuff later, just assume it's in the top level for now!
            spec_data = self.get_data('spec_data')

        if self.config.debug:
            spec_data = spec_data[:2000]
        # spec_data = self.get_data('spec_data')
        # balrog_data = self.get_data('balrog_data')
        # wide_data = self.get_data('wide_data')

        cell_deep_spec_data = self.deep_assignment['spec_data'][0]
        cell_wide_spec_data = self.wide_assignment['spec_data'][0]
        # pdb.set_trace()
        spec_data_for_pz = pd.DataFrame({key: spec_data[key],
                                         'cell_deep': cell_deep_spec_data,
                                         'cell_wide': cell_wide_spec_data})

        # compute p(z|c), redshift histograms of deep SOM cells
        pz_c = np.array(get_deep_histograms(None,  # this arg is not currently used in get_deep_histograms
                                            spec_data_for_pz,
                                            key=key,
                                            cells=all_deep_cells,
                                            overlap_weighted_pzc=False,
                                            bins=zbins))
        # compute p(c|chat,etc.), the deep-wide transfer function
        pc_chat = calculate_pcchat(deep_som_size,
                                   wide_som_size,
                                   self.deep_assignment['balrog_data'][0],  # balrog_data['cell_deep'],#.values,
                                   self.wide_assignment['balrog_data'][0],  # balrog_data['cell_wide'],#.values,
                                   np.ones(len(self.deep_assignment['balrog_data'][0])))
        pcchatdict = dict(pc_chat=pc_chat)
        self.add_data('pc_chat', pcchatdict)
        # use to write pc_chat out to file, leave in temporarily for cross checks
        # outfile = os.path.join(output_path, 'pcchat.npy')
        # np.savez(outfile, pc_chat=pc_chat)

        # compute p(chat), occupation in wide SOM cells
        all_wide_cells = np.arange(wide_som_size)
        cell_wide_wide_data = self.wide_assignment['wide_data'][0]
        wide_data_for_pz = pd.DataFrame({'cell_wide': cell_wide_wide_data})

        # compute p(z|chat) \propto sum_c p(z|c) p(c|chat)
        pz_chat = np.array(histogram(wide_data_for_pz,
                                     spec_data_for_pz,
                                     key=key,
                                     pcchat=pc_chat,
                                     cells=all_wide_cells,
                                     cell_weights=np.ones(len(all_wide_cells)),
                                     deep_som_size=deep_som_size,
                                     overlap_weighted_pzc=False,
                                     bins=zbins,
                                     individual_chat=True))
        # note: used to write out pz_chat to np, leave in temporarily for cross-checks
        # outfile = os.path.join(output_path, 'pzchat.npy')
        # np.savez(outfile, pz_chat=pz_chat)
        pzchatdict = dict(pz_chat=pz_chat)
        self.add_data('pz_chat', pzchatdict)

        # assign sample to tomographic bins
        # bin_edges = [0.0, 0.405, 0.665, 0.96, 2.0] # this is now a config input
        # n_bins = len(self.config.bin_edges) - 1
        tomo_bins_wide_dict = bin_assignment_spec(spec_data_for_pz,
                                                  deep_som_size,
                                                  wide_som_size,
                                                  bin_edges=self.config.bin_edges,
                                                  key_z=key,
                                                  key_cells_wide='cell_wide')
        tomo_bins_wide = tomo_bins_wide_2d(tomo_bins_wide_dict)
        # compute number of galaxies per tomographic bin (diagnostic info)
        # cell_occupation_info = wide_data_for_pz.groupby('cell_wide')['cell_wide'].count()
        # bin_occupation_info = {'bin' + str(i) : np.sum(cell_occupation_info.loc[tomo_bins_wide_dict[i]].values) for i in range(n_bins)}
        # print(bin_occupation_info)

        # calculate n(z)
        nz = redshift_distributions_wide(data=wide_data_for_pz,
                                         deep_data=spec_data_for_pz,
                                         overlap_weighted_pchat=False,
                                         overlap_weighted_pzc=False,
                                         bins=zbins,
                                         deep_som_size=deep_som_size,
                                         pcchat=pc_chat,
                                         tomo_bins=tomo_bins_wide,
                                         key=key,
                                         force_assignment=False,
                                         cell_key='cell_wide')

        return tomo_bins_wide, pz_c, pc_chat, nz

    def _find_wide_tomo_bins(self, tomo_bins_wide):
        """the current code has a map of the wide galaxies to the wide SOM cells
        (in wide_assign), and the mapping of which wide som cells map to which tomo
        bin (in tomo_bins_wide, passed as an arg to this function), but not a direct
        map of which tomo bin each wide galaxy gets mapped to.  This function will
        return a dictionary with two entries, one that contains an array of the
        same length as the number of input galaxies that contains an integer
        corresponding to which tomographic bin the galaxy belongs to, and the other
        (weight) that corresponds to the weight associated with that bin in the
        tomo_bins_wide file (it looks to always be one, but I'll copy it just in
        case there are situations where it is not 1.0)

        Inputs: tomo_bins_wide (returned by estimate pdf)
        Returns: wide_tomo_bins (dict)
        """
        # This code does not handle the fact that som of the SOM has no balrog sample covered, thereby should not be included in the  calculation
        # assert (0)
        raise ValueError("this code is no longer used")
        wide_assign = self.widedict['cells']
        # print(tomo_bins_wide)

        # nbins = len(self.config.bin_edges)-1
        # ngal = len(wide_assign)
        # tomo_mask = np.zeros(ngal, dtype=int)
        tmp_cells = np.concatenate([tomo_bins_wide[nbin][:, 0].astype(np.int32) for nbin in tomo_bins_wide])
        tmp_weights = np.concatenate([tomo_bins_wide[nbin][:, 1] for nbin in tomo_bins_wide])
        tmp_bins = np.concatenate([(np.ones(len(tomo_bins_wide[nbin][:, 0])) * nbin).astype(int) for nbin in tomo_bins_wide])
        sortidx = np.argsort(tmp_cells)
        indices = sortidx[np.searchsorted(tmp_cells, wide_assign, sorter=sortidx)]
        tomo_bins = tmp_bins[indices]
        tomo_weights = tmp_weights[indices]

        tmask_dict = dict(bin=tomo_bins, weight=tomo_weights)
        return tmask_dict

    def _initialize_run(self):
        """
        code that gets run once
        """

        self._output_handle = None

    def _do_chunk_output(self):
        """
        code that gets run once
        """
        print('TODO')
        # assert False
        raise NotImplementedError("_do_chunk_output not yet implemented")

    def _finalize_run(self):
        self._output_handle.finalize_write()

    def _process_chunk(self, start, end, data, first):
        """
        Run SOMPZ on a chunk of data
        """
        ngal_wide = len(data[self.config.inputs_wide[0]])
        num_inputs_wide = len(self.config.inputs_wide)
        data_wide = np.zeros([ngal_wide, num_inputs_wide])
        data_err_wide = np.zeros([ngal_wide, num_inputs_wide])
        for j, (col, errcol) in enumerate(zip(self.config.inputs_wide, self.config.input_errs_wide)):
            if self.config.convert_to_flux_wide:
                data_wide[:, j] = mag2flux(np.array(data[col], dtype=np.float32), self.config.zero_points_wide[j])
                data_err_wide[:, j] = magerr2fluxerr(np.array(data[errcol], dtype=np.float32), data_wide[:, j])
            else:
                data_wide[:, j] = np.array(data[col], dtype=np.float32)
                data_err_wide[:, j] = np.array(data[errcol], dtype=np.float32)

        if self.config.set_threshold_wide:
            truncation_value = self.config.thresh_value_wide
            for j in range(num_inputs_wide):
                mask = (data_wide[:, j] < self.config.thresh_val_wide)
                data_wide[:, j][mask] = truncation_value
                errmask = (data_err_wide[:, j] < self.config.thresh_val_wide)
                data_err_wide[:, j][errmask] = truncation_value

        data_wide_ndarray = np.array(data_wide, copy=False)
        flux_wide = data_wide_ndarray.view()
        data_err_wide_ndarray = np.array(data_err_wide, copy=False)
        flux_err_wide = data_err_wide_ndarray.view()

        cells_wide, dist_wide = self._assign_som(flux_wide, flux_err_wide, 'wide')
        print('TODO store this info')
        output_handle = None
        self._do_chunk_output(output_handle, start, end, first)

    def run(self,):
        self.deep_model, self.wide_model = self.open_model(**self.config)  # None
        print('initialized model', self.deep_model, self.wide_model)
        if self.config.spec_groupname:
            spec_data = self.get_data('spec_data')[self.config.spec_groupname]
        else:  # pragma: no cover
            spec_data = self.get_data('spec_data').to_pandas()

        if self.config.balrog_groupname:
            balrog_data = self.get_data('balrog_data')[self.config.balrog_groupname]
        else:  # pragma: no cover
            balrog_data = self.get_data('balrog_data').to_pandas()

        if self.config.wide_groupname:
            wide_data = self.get_data('wide_data')[self.config.wide_groupname]
        else:  # pragma: no cover
            wide_data = self.get_data('wide_data').to_pandas()

        # iterator = self.input_iterator("wide_data")
        # first = True
        # self._initialize_run() # TODO implement
        # self._output_handle = None # TODO consider handle for dict to store all outputs
        # for s, e, data_chunk in iterator:
        #    if self.rank == 0:
        #        print(f"Process {self.rank} running estimator on chunk {s} - {e}")
        #    self._process_chunk(s, e, data_chunk, first)
        #    first = False
        #    gc.collect()

        # print('You need to do spec_data and balrog_data')
        # self._finalize_run()
        # assert False,'below this line is code that needs to be updated'

        samples = [spec_data, balrog_data, wide_data]
        # NOTE: DO NOT CHANGE NAMES OF 'labels' below! They are used
        # in the naming of the outputs of the stage!
        labels = ['spec_data', 'balrog_data', 'wide_data']
        # output_path = './' # make kwarg
        # assign samples to SOMs
        # TODO: handle case of sample already having been assigned
        self.deep_assignment = {}
        self.wide_assignment = {}
        for i, (data, label) in enumerate(zip(samples, labels)):
            print("Working on {0}\n".format(label), flush=True)
            if i <= 1:
                outlabel = f"{label}_deep_assignment"
                if os.path.isfile(self.config[outlabel]):
                    temp = h5py.File(self.config[outlabel], 'r')
                    cells_deep, dist_deep = temp['cells'][:], temp['dist'][:]
                    self.deep_assignment[label] = (cells_deep, dist_deep)
                    tmpdict = dict(cells=cells_deep, dist=dist_deep)
                    self.add_data(outlabel, tmpdict)
                    temp.close()
                else:
                    # print(self.config.inputs_deep)
                    # #######
                    #  REDO how subset of data is copied so that it works for hdf5
                    # data_deep = data[self.config.inputs_deep]
                    # data_deep_ndarray = np.array(data_deep,copy=False)
                    # Flux_deep = data_deep_ndarray.view((np.float32,
                    #                                    len(self.config.inputs_deep)))
                    ngal_deep = len(data[self.config.inputs_deep[0]])
                    num_inputs_deep = len(self.config.inputs_deep)
                    data_deep = np.zeros([ngal_deep, num_inputs_deep])
                    data_err_deep = np.zeros([ngal_deep, num_inputs_deep])
                    for j, (col, errcol) in enumerate(zip(self.config.inputs_deep, self.config.input_errs_deep)):
                        if self.config.convert_to_flux_deep:
                            data_deep[:, j] = mag2flux(np.array(data[col], dtype=np.float32), self.config.zero_points_deep[j])
                            data_err_deep[:, j] = magerr2fluxerr(np.array(data[errcol], dtype=np.float32), data_deep[:, j])
                        else:
                            data_deep[:, j] = np.array(data[col], dtype=np.float32)
                            data_err_deep[:, j] = np.array(data[errcol], dtype=np.float32)

                    # ### TRY PUTTING IN THRESHOLD FROM INFORM!
                    if self.config.set_threshold_deep:
                        truncation_value = self.config.thresh_val_deep
                        for j in range(num_inputs_deep):
                            mask = (data_deep[:, j] < self.config.thresh_val_deep)
                            data_deep[:, j][mask] = truncation_value
                            errmask = (data_err_deep[:, j] < self.config.thresh_val_deep)
                            data_err_deep[:, j][errmask] = truncation_value

                    data_deep_ndarray = np.array(data_deep, copy=False)
                    flux_deep = data_deep_ndarray.view()

                    # data_deep = data[self.config.err_inputs_deep]
                    # data_deep_ndarray = np.array(data_deep,copy=False)
                    # flux_err_deep = data_deep_ndarray.view((np.float32,
                    #                                         len(self.config.err_inputs_deep)))
                    data_err_deep_ndarray = np.array(data_err_deep, copy=False)
                    flux_err_deep = data_err_deep_ndarray.view()
                    cells_deep, dist_deep = self._assign_som(flux_deep, flux_err_deep, 'deep')

                    self.deep_assignment[label] = (cells_deep, dist_deep)
                    # take out numpy savez
                    # outfile = os.path.join(output_path, label + '_deep.npz')
                    # np.savez(outfile, cells=cells_deep, dist=dist_deep)
                    tmpdict = dict(cells=cells_deep, dist=dist_deep)
                    self.add_data(outlabel, tmpdict)
            else:
                cells_deep, dist_deep = None, None

            ngal_wide = len(data[self.config.inputs_wide[0]])
            num_inputs_wide = len(self.config.inputs_wide)
            data_wide = np.zeros([ngal_wide, num_inputs_wide])
            data_err_wide = np.zeros([ngal_wide, num_inputs_wide])
            for j, (col, errcol) in enumerate(zip(self.config.inputs_wide, self.config.input_errs_wide)):
                if self.config.convert_to_flux_wide:
                    data_wide[:, j] = mag2flux(np.array(data[col], dtype=np.float32), self.config.zero_points_wide[j])
                    data_err_wide[:, j] = magerr2fluxerr(np.array(data[errcol], dtype=np.float32), data_wide[:, j])
                else:
                    data_wide[:, j] = np.array(data[col], dtype=np.float32)
                    data_err_wide[:, j] = np.array(data[errcol], dtype=np.float32)

            # ## PUT IN THRESHOLD!
            if self.config.set_threshold_wide:
                truncation_value = self.config.thresh_value_wide
                for j in range(num_inputs_wide):
                    mask = (data_wide[:, j] < self.config.thresh_val_wide)
                    data_wide[:, j][mask] = truncation_value
                    errmask = (data_err_wide[:, j] < self.config.thresh_val_wide)
                    data_err_wide[:, j][errmask] = truncation_value

            # data_wide = data[self.config.input_errs_wide]
            data_wide_ndarray = np.array(data_wide, copy=False)
            flux_wide = data_wide_ndarray.view()
            data_err_wide_ndarray = np.array(data_err_wide, copy=False)
            flux_err_wide = data_err_wide_ndarray.view()

            cells_wide, dist_wide = self._assign_som(flux_wide, flux_err_wide, 'wide')
            if i > 1:
                widelabel = f"{label}_assignment"
            else:
                widelabel = f"{label}_wide_assignment"

            self.wide_assignment[label] = (cells_wide, dist_wide)
            self.widedict = dict(cells=cells_wide, dist=dist_wide)
            self.add_data(widelabel, self.widedict)

        tomo_bins_wide, pz_c, pc_chat, nz = self._estimate_pdf()  # *samples
        with open(self.config['tomo_bin_mask_wide_data'], 'wb') as f:
            pickle.dump(tomo_bins_wide, f)

        # Add in computation of which tomo bin each wide galaxy is mapped to
        # wide_tomo_bin_dict = self._find_wide_tomo_bins(tomo_bins_wide)
        # self.add_data("tomo_bin_mask_wide_data", wide_tomo_bin_dict)

        # self.nz = nz
        tomo_ens = qp.Ensemble(qp.interp, data=dict(xvals=self.bincents, yvals=nz))
        self.add_data('nz', tomo_ens)

        pzcdict = dict(pz_c=pz_c)
        self.add_data('pz_c', pzcdict)  # wide_data_cells_wide)

    def estimate(self,
                 spec_data,
                 balrog_data,
                 wide_data,):
        self.set_data("spec_data", spec_data)
        self.set_data("balrog_data", balrog_data)
        self.set_data("wide_data", wide_data)
        self.run()
        self.finalize()
        return


class SOMPZPzc(CatEstimator):
    """Calcaulate pzc
    """
    name = "SOMPZPzc"
    config_options = CatEstimator.config_options.copy()
    config_options.update(inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          redshift_col=SHARED_PARAMS,
                          deep_groupname=Param(str, "photometry", msg="hdf5_groupname for deep file"),
                          bin_edges=Param(list, default_bin_edges, msg="list of edges of tomo bins"),
                          zbins_min=Param(float, 0.0, msg="minimum redshift for output grid"),
                          zbins_max=Param(float, 6.0, msg="maximum redshift for output grid"),
                          zbins_dz=Param(float, 0.01, msg="delta z for defining output grid"),
                          )
    inputs = [('spec_data', TableHandle),
              ('cell_deep_spec_data', TableHandle),]
    outputs = [('pz_c', Hdf5Handle)]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        # check on bands, errs, and prior band

    def run(self):
        spec_data = self.get_data('spec_data')
        cell_deep_spec_data = self.get_data('cell_deep_spec_data')
        self.deep_som_size = int(cell_deep_spec_data['som_size'][0])
        key = self.config.redshift_col
        zbins = np.arange(self.config.zbins_min - self.config.zbins_dz / 2., self.config.zbins_max + self.config.zbins_dz, self.config.zbins_dz)
        spec_data_for_pz = pd.DataFrame({key: spec_data[key],
                                         'cell_deep': cell_deep_spec_data['cells']})
        all_deep_cells = np.arange(self.deep_som_size)
        pz_c = np.array(get_deep_histograms(None,  # this arg is not currently used in get_deep_histograms
                                            spec_data_for_pz,
                                            key=key,
                                            cells=all_deep_cells,
                                            overlap_weighted_pzc=False,
                                            bins=zbins))
        pzcdict = dict(pz_c=pz_c)
        self.add_data('pz_c', pzcdict)

    def estimate(self, spec_data, cell_deep_spec_data):
        spec_data = self.set_data('spec_data', spec_data)
        cell_deep_spec_data = self.set_data('cell_deep_spec_data', cell_deep_spec_data)
        self.run()
        self.finalize()


class SOMPZPzchat(CatEstimator):
    """Calcaulate pzchat
    """
    name = "SOMPZPzchat"
    config_options = CatEstimator.config_options.copy()
    config_options.update(inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          redshift_col=SHARED_PARAMS,
                          bin_edges=Param(list, default_bin_edges, msg="list of edges of tomo bins"),
                          zbins_min=Param(float, 0.0, msg="minimum redshift for output grid"),
                          zbins_max=Param(float, 6.0, msg="maximum redshift for output grid"),
                          zbins_dz=Param(float, 0.01, msg="delta z for defining output grid"),
                          )
    inputs = [('spec_data', TableHandle),
              ('cell_deep_spec_data', TableHandle),
              ('cell_wide_wide_data', TableHandle),
              ('pz_c', Hdf5Handle),
              ('pc_chat', Hdf5Handle),
              ]
    outputs = [('pz_chat', Hdf5Handle)]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        # check on bands, errs, and prior band

    def run(self):
        spec_data = self.get_data('spec_data')
        cell_wide_wide_data = self.get_data('cell_wide_wide_data')
        cell_deep_spec_data = self.get_data('cell_deep_spec_data')
        self.wide_som_size = int(cell_wide_wide_data['som_size'][0])
        self.deep_som_size = int(cell_deep_spec_data['som_size'][0])
        pc_chat = self.get_data('pc_chat')['pc_chat'][:]
        key = self.config.redshift_col
        zbins = np.arange(self.config.zbins_min - self.config.zbins_dz / 2., self.config.zbins_max + self.config.zbins_dz, self.config.zbins_dz)
        spec_data_for_pz = pd.DataFrame({key: spec_data[key],
                                         'cell_deep': cell_deep_spec_data['cells']})

        wide_data_for_pz = pd.DataFrame({'cell_wide': cell_wide_wide_data['cells']})

        all_wide_cells = np.arange(self.wide_som_size)
        all_deep_cells = np.arange(self.deep_som_size)

        pz_chat = np.array(histogram(wide_data_for_pz,
                                     spec_data_for_pz,
                                     key=key,
                                     pcchat=pc_chat,
                                     cells=all_wide_cells,
                                     cell_weights=np.ones(len(all_wide_cells)),
                                     deep_som_size=self.deep_som_size,
                                     overlap_weighted_pzc=False,
                                     bins=zbins,
                                     individual_chat=True))
        pzchatdict = dict(pz_chat=pz_chat)
        self.add_data('pz_chat', pzchatdict)

    def estimate(self, spec_data, cell_deep_spec_data, cell_wide_wide_data, pz_c, pc_chat):
        self.set_data("spec_data", spec_data)
        self.set_data("cell_deep_spec_data", cell_deep_spec_data)
        self.set_data("cell_wide_wide_data", cell_wide_wide_data)
        self.set_data("pz_c", pz_c)
        self.set_data("pc_chat", pc_chat)
        self.run()
        self.finalize()


class SOMPZPc_chat(CatEstimator):
    """Calcaulate p(c|chat)
    """
    name = "SOMPZPc_chat"
    config_options = CatEstimator.config_options.copy()
    config_options.update(inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          )
    inputs = [('cell_deep_balrog_data', TableHandle),
              ('cell_wide_balrog_data', TableHandle),
              ]
    outputs = [('pc_chat', Hdf5Handle)]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        # check on bands, errs, and prior band

    def run(self):
        cell_deep_balrog_data = self.get_data('cell_deep_balrog_data')
        cell_wide_balrog_data = self.get_data('cell_wide_balrog_data')
        self.deep_som_size = int(cell_deep_balrog_data['som_size'][0])
        self.wide_som_size = int(cell_wide_balrog_data['som_size'][0])
        pc_chat = calculate_pcchat(self.deep_som_size,
                                   self.wide_som_size,
                                   cell_deep_balrog_data['cells'],  # balrog_data['cell_deep'],#.values,
                                   cell_wide_balrog_data['cells'],  # balrog_data['cell_wide'],#.values,
                                   np.ones(len(cell_wide_balrog_data['cells'])))
        pcchatdict = dict(pc_chat=pc_chat)
        self.add_data('pc_chat', pcchatdict)

    def estimate(self, cell_deep_balrog_data, cell_wide_balrog_data):
        self.set_data('cell_deep_balrog_data', cell_deep_balrog_data)
        self.set_data('cell_wide_balrog_data', cell_wide_balrog_data)
        self.run()
        self.finalize()


class SOMPZTomobin(CatEstimator):
    """Calcaulate tomobin
    """
    name = "SOMPZTomobin"
    config_options = CatEstimator.config_options.copy()
    config_options.update(inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          redshift_col=SHARED_PARAMS,
                          bin_edges=Param(list, default_bin_edges, msg="list of edges of tomo bins"),
                          zbins_min=Param(float, 0.0, msg="minimum redshift for output grid"),
                          zbins_max=Param(float, 6.0, msg="maximum redshift for output grid"),
                          zbins_dz=Param(float, 0.01, msg="delta z for defining output grid"),
                          )
    inputs = [('spec_data', TableHandle),
              ('cell_deep_spec_data', TableHandle),
              ('cell_wide_spec_data', TableHandle),
              ]
    outputs = [('tomo_bins_wide', Hdf5Handle)]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        # check on bands, errs, and prior band

    def run(self):
        spec_data = self.get_data('spec_data')
        cell_deep_spec_data = self.get_data('cell_deep_spec_data')
        cell_wide_spec_data = self.get_data('cell_wide_spec_data')
        self.wide_som_size = int(cell_wide_spec_data['som_size'][0])
        self.deep_som_size = int(cell_deep_spec_data['som_size'][0])
        # pc_chat = self.get_data('pc_chat')['pc_chat'][:]
        key = self.config.redshift_col

        spec_data_for_pz = pd.DataFrame({key: spec_data[key],
                                         'cell_deep': cell_deep_spec_data['cells'],
                                         'cell_wide': cell_wide_spec_data['cells']})
        tomo_bins_wide_dict = bin_assignment_spec(spec_data_for_pz,
                                                  self.deep_som_size,
                                                  self.wide_som_size,
                                                  bin_edges=self.config['bin_edges'],
                                                  key_z=key,
                                                  key_cells_wide='cell_wide')
        tomobinswide = tomo_bins_wide_2d(tomo_bins_wide_dict)
        tomobinsmapping = -1 * np.ones((self.wide_som_size, 2))
        for key in tomobinswide:
            tomobinsmapping[tomobinswide[key][:, 0].astype(int), 0] = key
            tomobinsmapping[tomobinswide[key][:, 0].astype(int), 1] = tomobinswide[key][:, 1]

        self.add_data('tomo_bins_wide', dict(tomo_bins_wide=tomobinsmapping))

    def estimate(self, spec_data, cell_deep_spec_data, cell_wide_spec_data):
        self.set_data('spec_data', spec_data)
        self.set_data('cell_deep_spec_data', cell_deep_spec_data)
        self.set_data('cell_wide_spec_data', cell_wide_spec_data)
        self.run()
        self.finalize()


class SOMPZnz(CatEstimator):
    """Calcaulate nz
    """
    name = "SOMPZnz"
    config_options = CatEstimator.config_options.copy()
    config_options.update(inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          redshift_col=SHARED_PARAMS,
                          bin_edges=Param(list, default_bin_edges, msg="list of edges of tomo bins"),
                          zbins_min=Param(float, 0.0, msg="minimum redshift for output grid"),
                          zbins_max=Param(float, 6.0, msg="maximum redshift for output grid"),
                          zbins_dz=Param(float, 0.01, msg="delta z for defining output grid"),
                          )
    inputs = [('spec_data', TableHandle),
              ('cell_deep_spec_data', TableHandle),
              ('cell_wide_wide_data', TableHandle),
              ('tomo_bins_wide', Hdf5Handle),
              ('pc_chat', Hdf5Handle),
              ]
    outputs = [('nz', QPHandle)]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        # check on bands, errs, and prior band

    def run(self):
        spec_data = self.get_data('spec_data')
        cell_wide_wide_data = self.get_data('cell_wide_wide_data')
        cell_deep_spec_data = self.get_data('cell_deep_spec_data')
        self.wide_som_size = int(cell_wide_wide_data['som_size'][0])
        self.deep_som_size = int(cell_deep_spec_data['som_size'][0])
        tomo_bins_wide_in = self.get_data('tomo_bins_wide')['tomo_bins_wide'][:]
        tomo_bins_wide = {}
        for i in np.unique(tomo_bins_wide_in[:, 0]):
            if i < 0:
                continue
            inarr1 = np.where(tomo_bins_wide_in[:, 0] == i)[0]
            inarr2 = tomo_bins_wide_in[inarr1, 1]
            tomo_bins_wide[i] = np.array([inarr1, inarr2]).T
        pc_chat = self.get_data('pc_chat')['pc_chat'][:]
        key = self.config.redshift_col
        zbins = np.arange(self.config.zbins_min - self.config.zbins_dz / 2., self.config.zbins_max + self.config.zbins_dz, self.config.zbins_dz)
        spec_data_for_pz = pd.DataFrame({key: spec_data[key],
                                         'cell_deep': cell_deep_spec_data['cells']})

        wide_data_for_pz = pd.DataFrame({'cell_wide': cell_wide_wide_data['cells']})

        all_wide_cells = np.arange(self.wide_som_size)
        all_deep_cells = np.arange(self.deep_som_size)
        nz = redshift_distributions_wide(data=wide_data_for_pz,
                                         deep_data=spec_data_for_pz,
                                         overlap_weighted_pchat=False,
                                         overlap_weighted_pzc=False,
                                         bins=zbins,
                                         deep_som_size=self.deep_som_size,
                                         pcchat=pc_chat,
                                         tomo_bins=tomo_bins_wide,
                                         key=key,
                                         force_assignment=False,
                                         cell_key='cell_wide')
        self.bincents = 0.5 * (zbins[1:] + zbins[:-1])
        tomo_ens = qp.Ensemble(qp.interp, data=dict(xvals=self.bincents, yvals=nz))
        self.add_data('nz', tomo_ens)

    def estimate(self, spec_data, cell_deep_spec_data, cell_wide_wide_data, tomo_bins_wide, pc_chat):
        spec_data = self.set_data('spec_data', spec_data)
        cell_deep_spec_data = self.set_data('cell_deep_spec_data', cell_deep_spec_data)
        cell_wide_wide_data = self.set_data('cell_wide_wide_data', cell_wide_wide_data)
        tomo_bins_wide = self.set_data('tomo_bins_wide', tomo_bins_wide)
        pc_chat = self.set_data('pc_chat', pc_chat)
        self.run()
        self.finalize()


class SOMPZEstimatorBase(CatEstimator):
    """CatEstimator subclass to compute redshift PDFs for SOMPZ
    """
    name = "SOMPZEstimatorBase"
    config_options = CatEstimator.config_options.copy()
    config_options.update(chunk_size=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS, #Param(str, "redshift", msg="name of redshift column"),
                          hdf5_groupname=None,  #Param(str, "photometry", msg="hdf5_groupname for data"),
                          #groupname=SHARED_PARAMS, #Param(str, "", msg="hdf5_groupname for data"),
                          inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          input_errs=Param(list, default_err_names, msg="list of the names of columns containing errors on inputs for deep data"),
                          zero_points=Param(list, default_zero_points, msg="zero points for converting mags to fluxes for deep data, if needed"),
                          som_shape=Param(list, [32, 32], msg="shape for the deep som, must be a 2-element tuple"),
                          som_minerror=Param(float, 0.01, msg="floor placed on observational error on each feature in deep som"),
                          som_wrap=Param(bool, False, msg="flag to set whether the deep SOM has periodic boundary conditions"),
                          som_take_log=Param(bool, True, msg="flag to set whether to take log of inputs (i.e. for fluxes) for deep som"),
                          convert_to_flux=Param(bool, False, msg="flag for whether to convert input columns to fluxes for deep data"
                                                     "set to true if inputs are mags and to False if inputs are already fluxes"),
                          set_threshold=Param(bool, False, msg="flag for whether to replace values below a threshold with a set number"),
                          thresh_val=Param(float, 1.e-5, msg="threshold value for set_threshold for deep data"),
                          debug=Param(bool, False, msg="boolean reducing dataset size for quick debuggin"))

    inputs = [('model', ModelHandle),
              ('data', TableHandle),]
    outputs = [
        ('assignment', Hdf5Handle),
    ]

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do SOMPZ specific setup
        """
        super().__init__(args, **kwargs)
        # check on bands, errs, and prior band
        if len(self.config.inputs) != len(self.config.input_errs):  # pragma: no cover
            raise ValueError("Number of inputs_deep specified in inputs_deep must be equal to number of mag errors specified in input_errs_deep!")
        if len(self.config.som_shape) != 2:  # pragma: no cover
            raise ValueError(f"som_shape must be a list with two integers specifying the SOM shape, not len {len(self.config.som_shape)}")

    def _assign_som(self, flux, flux_err):
        # som_dim = self.config.som_shape[0]
        s0 = int(self.config.som_shape[0])
        s1 = int(self.config.som_shape[1])
        self.som_size = np.array([int(s0 * s1)])
        # output_path = './'  # TODO make kwarg
        nTrain = flux.shape[0]
        # som_weights = np.load(infile_som, allow_pickle=True)
        som_weights = self.model['som'].weights
        hh = somfuncs.hFunc(nTrain, sigma=(30, 1))
        metric = somfuncs.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
        som = somfuncs.NoiseSOM(metric, None, None,
                                learning=hh,
                                shape=(s0, s1),
                                wrap=False, logF=True,
                                initialize=som_weights,
                                minError=0.02)
        subsamp = 1
        cells_test, dist_test = som.classify(flux[::subsamp, :], flux_err[::subsamp, :])

        return cells_test, dist_test

    def _process_chunk(self, start, end, data, first):
        """
        Run SOMPZ on a chunk of data
        """
        ngal_wide = len(data[self.config.inputs[0]])
        num_inputs_wide = len(self.config.inputs)
        data_wide = np.zeros([ngal_wide, num_inputs_wide])
        data_err_wide = np.zeros([ngal_wide, num_inputs_wide])
        for j, (col, errcol) in enumerate(zip(self.config.inputs, self.config.input_errs)):
            if self.config.convert_to_flux:
                data_wide[:, j] = mag2flux(np.array(data[col], dtype=np.float32), self.config.zero_points[j])
                data_err_wide[:, j] = magerr2fluxerr(np.array(data[errcol], dtype=np.float32), data_wide[:, j])
            else:  # pragma: no cover
                data_wide[:, j] = np.array(data[col], dtype=np.float32)
                data_err_wide[:, j] = np.array(data[errcol], dtype=np.float32)

        if self.config.set_threshold:
            truncation_value = self.config.thresh_val
            for j in range(num_inputs_wide):
                mask = (data_wide[:, j] < self.config.thresh_val)
                data_wide[:, j][mask] = truncation_value
                errmask = (data_err_wide[:, j] < self.config.thresh_val)
                data_err_wide[:, j][errmask] = truncation_value

        data_wide_ndarray = np.array(data_wide, copy=False)
        flux_wide = data_wide_ndarray.view()
        data_err_wide_ndarray = np.array(data_err_wide, copy=False)
        flux_err_wide = data_err_wide_ndarray.view()

        cells_wide, dist_wide = self._assign_som(flux_wide, flux_err_wide)
        output_chunk = dict(cells=cells_wide, dist=dist_wide)
        self._do_chunk_output(output_chunk, start, end, first)

    def _do_chunk_output(self, output_chunk, start, end, first):
        """

        Parameters
        ----------
        output_chunk
        start
        end
        first

        Returns
        -------

        """
        if first:
            self._output_handle = self.add_handle('assignment', data=output_chunk)
            self._output_handle.initialize_write(self._input_length, communicator=self.comm)
        self._output_handle.set_data(output_chunk, partial=True)
        self._output_handle.write_chunk(start, end)

    def run(self):
        self.model = None
        self.model = self.open_model(**self.config)  # None
        first = True
        iter1 = self.input_iterator('data')
        # iter1 = self.input_iterator('data', groupname=self.config.hdf5_groupname)
        # iter1 = self.input_iterator('data')
        self._output_handle = None
        for s, e, test_data in iter1:
            print(f"Process {self.rank} running creator on chunk {s} - {e}", flush=True)
            self._process_chunk(s, e, test_data, first)
            first = False
            gc.collect()
        if self.comm:  # pragma: no cover
            self.comm.Barrier()
        self._finalize_run()

    def estimate(self, data):
        self.set_data("data", data)
        self.run()
        self.finalize()
        return

    def _finalize_run(self):
        """

        Returns
        -------

        """
        tmpdict = dict(som_size=self.som_size)
        self._output_handle.finalize_write(**tmpdict)


class SOMPZEstimatorWide(SOMPZEstimatorBase):
    """CatEstimator subclass to compute redshift PDFs for SOMPZ
    """
    name = "SOMPZEstimatorWide"

    inputs = [('wide_model', ModelHandle),
              ('data', TableHandle),]

    def open_model(self, **kwargs):
        """Load the model and/or attach it to this Creator.

        Keywords
        --------
        model : object, str or ModelHandle
            Either an object with a trained model, a path pointing to a file
            that can be read to obtain the trained model, or a ``ModelHandle``
            providing access to the trained model

        Returns
        -------
        self.model : object
            The object encapsulating the trained model
        """
        model = kwargs.get("model", kwargs.get('wide_model', None))
        if model is None or model == "None":  # pragma: no cover
            self.model = None
        else:
            if isinstance(model, str):
                self.model = self.set_data("wide_model", data=None, path=model)
                self.config["model"] = model
            else:  # pragma: no cover
                if isinstance(model, ModelHandle):  # pragma: no cover
                    if model.has_path:
                        self.config["model"] = model.path
                self.model = self.set_data("wide_model", model)

        return self.model


class SOMPZEstimatorDeep(SOMPZEstimatorBase):
    """CatEstimator subclass to compute redshift PDFs for SOMPZ
    """
    name = "SOMPZEstimatorDeep"
    inputs = [('deep_model', ModelHandle),
              ('data', TableHandle),]

    def open_model(self, **kwargs):
        """Load the model and/or attach it to this Creator.

        Keywords
        --------
        model : object, str or ModelHandle
            Either an object with a trained model, a path pointing to a file
            that can be read to obtain the trained model, or a ``ModelHandle``
            providing access to the trained model

        Returns
        -------
        self.model : object
            The object encapsulating the trained model
        """
        model = kwargs.get("model", kwargs.get('deep_model', None))
        if model is None or model == "None":  # pragma: no cover
            self.model = None
        else:
            if isinstance(model, str):
                self.model = self.set_data("deep_model", data=None, path=model)
                self.config["model"] = model
            else:  # pragma: no cover
                if isinstance(model, ModelHandle):  # pragma: no cover
                    if model.has_path:
                        self.config["model"] = model.path
                self.model = self.set_data("deep_model", model)

        return self.model
