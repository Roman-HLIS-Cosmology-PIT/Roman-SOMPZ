from scm_pipeline import PipelineStage
from rail.core.data import TableHandle, ModelHandle, QPHandle, Hdf5Handle
from rail.estimation.estimator import CatEstimator, CatInformer
from ceci.config import StageParameter as Param
import rail.estimation.algos.som as somfuncs
from rail.core.common_params import SHARED_PARAMS
import numpy as np
import gc
#from ceci_example.types import NpyFile, HDFFile
# We need to define NpyFile etc we want to use inside ceci_example.types, Or not??? get_input don't use this function and we open in run ourselves
def_bands = ["u", "g", "r", "i", "z", "y"]
default_bin_edges = [0.0, 0.405, 0.665, 0.96, 2.0]
default_input_names = []
default_err_names = []
default_zero_points = []

# assign_som_deep_ZPU_mpi4py_ceci.py
class PhotozDeepZeroPointPipe(CatEstimator):
    """CatEstimator subclass to compute redshift PDFs for SOMPZ
    """
    name = "PhotozDeepZeroPointPipe"
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

    inputs = [('deep_model', ModelHandle), ('lhc_samples', TableHandle),
              ('balrogdata', TableHandle)]
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

    def _process_chunk(self, start, end, data, first, total_LHCsamples):
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
        if first:
            output_chunk = {}
            output_chunk['cells'] = cells_wide
            output_chunk['dist'] = dist_wide
            for ind in range(total_LHCsamples):
                output_chunk['cells_LHC_id_{0}'.format(ind)] = cells_wide
                output_chunk['dist_LHC_id_{0}'.format(ind)] = dist_wide
        else:
            output_chunk = dict(cells=cells_wide, dist=dist_wide)
        self._do_chunk_output(output_chunk, start, end, first)

    def _process_chunk_perturb(self, start, end, data, first, LHC_id, LHC_sample):
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
            #Here we assume only one deep field 
            data_wide[:, j] = data_wide[:, j]* 10**LHC_sample[j]
            data_err_wide[:, j] = data_wide[:, j]* 10**LHC_sample[j]

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
        output_chunk = {}
        output_chunk['cells_LHC_id_{0}'.format(LHC_id)] = cells_wide
        output_chunk['dist_LHC_id_{0}'.format(LHC_id)] = dist_wide
        self._do_chunk_output(output_chunk, start, end, first)

    def _do_chunk_output(self, output_chunk, start, end, first, purturbnum=-1):
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
            name = "assignment"
            self._output_handle = self.add_handle(name, data=output_chunk)
            self._output_handle.initialize_write(self._input_length, communicator=self.comm)
        self._output_handle.set_data(output_chunk, partial=True)
        self._output_handle.write_chunk(start, end)

    def run(self):
        self.model = None
        self.model = self.open_model(**self.config)  # None
        first = True
        iter1 = self.input_iterator('balrogdata') # here we assume no deep galaxy duplicates  (Will deal with this later)
        LHC_samples = self.get_data('lhc_samples').view(np.ndarray)['samples']
        print(LHC_samples.shape)
        self._output_handle = None
        for s, e, test_data in iter1:
            print(f"Process {self.rank} running creator on chunk {s} - {e}", flush=True)
            self._process_chunk(s, e, test_data, first, len(LHC_samples))
            first = False
            for LHC_id, LHC_sample in enumerate(LHC_samples):
                self._process_chunk_perturb(s, e, test_data, first, LHC_id, LHC_sample)
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



