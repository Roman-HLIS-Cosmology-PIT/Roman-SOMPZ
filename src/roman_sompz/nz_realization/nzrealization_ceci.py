from scm_pipeline import PipelineStage
from rail.core.stage import RailStage
from rail.core.data import TableHandle, ModelHandle, QPHandle, Hdf5Handle
from rail.estimation.estimator import CatEstimator, CatInformer
from ceci.config import StageParameter as Param
import roman_sompz.rail_sompz.src.rail.estimation.algos.som as somfuncs
from rail.core.common_params import SHARED_PARAMS
import numpy as np
import gc
from roman_sompz.nz_realization.generate_LHC_sample_ceci import generate_LHC_points
import astropy.table as apTable
import tables_io
from ceci.config import StageParameter
import numpy as np
import healpy as hp
import os
from astropy.io import fits
from roman_sompz.nz_realization.samplevariance import *
from roman_sompz.nz_realization.Roman_selection_nz_final_rushift_zpshift_ceci import get_realizations
#from ceci_example.types import NpyFile, HDFFile
# We need to define NpyFile etc we want to use inside ceci_example.types, Or not??? get_input don't use this function and we open in run ourselves
def_bands = ["u", "g", "r", "i", "z", "y"]
default_bin_edges = [0.0, 0.405, 0.665, 0.96, 2.0]
default_input_names = []
default_err_names = []
default_zero_points = []

#from ceci_example.types import NpyFile, HDFFile
# We need to define NpyFile etc we want to use inside ceci_example.types, Or not??? get_input don't use this function and we open in run ourselves
#If we use RailStage and use self.get_data for the dust map, we have to put the maps as input in yaml file. And I hesitate doing this.
class PreparePZRealizationsPipe(RailStage):
    """
    Generates LHC sampling points for the uncertainty beside sample variance and shot noise.
    Saves to a NPZfile with LHCsampling points.
    """

    name = "PreparePZRealizationsPipe"
    inputs = ("sfd_data",TableHandle), ("lss_error_data",TableHandle)
    outputs = [("LHC_samples_deep_zp",TableHandle), ("LHC_samples_wide_zp",TableHandle),  ("LHC_samples_sky",TableHandle)]
    parallel = False

    config_options = {
        "redshift_sample_uncertainty": StageParameter(bool, False,
            msg="The bias and uncertainty due to the use of photometric redshfit calibration sample" ), #If we use spec-only, don't need this
        "photometric_zeropoint_deep": StageParameter(bool, False,
            msg="If you want to add photometric zero point uncertainty due to the deep field zero point offsets " ),
        "photometric_zeropoint_wide": StageParameter(bool, False,
            msg="If you want to add photometric zero point uncertainty due to the wide field zero point offsets " ),  #not implemented
        "photometric_skybackground_deep": StageParameter(bool, False,
            msg="If you want to add skybackground uncertainty on the deep field photometry " ),    
        "photometric_skybackground_wide": StageParameter(bool, False,
            msg="If you want to add skybackground uncertainty on the wide field photometry " ),      
        "deepfield_zeropoint_data": StageParameter(list, [], msg="zero point uncertainty for deep field" ), 
        "widefield_zeropoint_data": StageParameter(list, [], msg="zero point uncertainty for wide field" ), 
        "num_lhc_points":StageParameter(int, 100,msg="number of lhc points we want to sample")
    }


    def run(self):
        redshift_sample_uncertainty = self.config["redshift_sample_uncertainty"]
        photometric_zeropoint_deep = self.config["photometric_zeropoint_deep"]
        photometric_zeropoint_wide = self.config["photometric_zeropoint_wide"]
        photometric_skybackground_deep = self.config["photometric_skybackground_deep"]
        photometric_skybackground_wide = self.config["photometric_skybackground_wide"]
        deepfield_zeropoint_data =  self.config["deepfield_zeropoint_data"]
        widefield_zeropoint_data =  self.config["widefield_zeropoint_data"]
        sfd_map = self.get_data('sfd_data').view(np.ndarray)['map']
        lss_error_map = self.get_data('lss_error_data').view(np.ndarray)['map']
        num_lhc_points = self.config["num_lhc_points"]

    
        lhc_samples = generate_LHC_points(np.array(deepfield_zeropoint_data), np.array(widefield_zeropoint_data), sfd_map, lss_error_map, photometric_zeropoint_deep, redshift_sample_uncertainty, photometric_zeropoint_wide, photometric_skybackground_deep, photometric_skybackground_wide, num_lhc_points)
        deep_zp_samples=lhc_samples['deep_zp']
        wide_zp_samples=lhc_samples['wide_zp']
        sky_samples=lhc_samples['sky']

        #I am saving to multiple tables since they will eventually have different length.
        if photometric_zeropoint_deep:
            #deep_zp_filename = self.get_output(self.outputs[0][0])
            deep_zp_t = np.zeros(len(deep_zp_samples), dtype=[("samples", '>f8', deep_zp_samples.shape[-1])])
            deep_zp_t['samples'] = deep_zp_samples
            deep_zp_table = apTable.Table(deep_zp_t)
            deep_zp_table = tables_io.convert(deep_zp_table, tables_io.types.NUMPY_FITS)
            #deep_zp_filename, deep_zp_file_extension = os.path.splitext(deep_zp_filename)
            #print(deep_zp_table, deep_zp_filename, deep_zp_file_extension)
            #tables_io.write(deep_zp_table, deep_zp_filename, deep_zp_file_extension[1:])

    

        if photometric_zeropoint_wide: 
            #wide_zp_filename = self.get_output(self.outputs[1][0])
            wide_zp_t = np.zeros(len(wide_zp_samples), dtype=[("samples", '>f8', wide_zp_samples.shape[-1])])
            wide_zp_t['samples'] = wide_zp_samples
            wide_zp_table = apTable.Table(wide_zp_t)
            wide_zp_table = tables_io.convert(wide_zp_table, tables_io.types.NUMPY_FITS)
            #wide_zp_filename, wide_zp_file_extension = os.path.splitext(wide_zp_filename)
            #tables_io.write(wide_zp_table, wide_zp_filename, wide_zp_file_extension[1:])

        # We need to use the same dust error map for the deep and wide field, assuming they are both corrected using the csdf map
        if photometric_skybackground_deep or photometric_skybackground_wide: 
            #sky_filename = self.get_output(self.outputs[2][0])
            sky_t = np.zeros(len(sky_samples), dtype=[("samples", '>f8', sky_samples.shape[-1])])
            sky_t['samples'] = sky_samples
            sky_table = apTable.Table(sky_t)
            sky_table = tables_io.convert(sky_table, tables_io.types.NUMPY_FITS)
            #sky_filename, sky_file_extension = os.path.splitext(sky_filename)
            #tables_io.write(sky_table, sky_filename, sky_file_extension[1:])
            
        self.add_data("LHC_samples_deep_zp", deep_zp_table)
        self.add_data("LHC_samples_wide_zp", wide_zp_table)
        self.add_data("LHC_samples_sky", sky_table)
        
# assign_som_deep_ZPU_mpi4py_ceci.py
class PhotozZpDustPipe(CatEstimator):
    """CatEstimator subclass to assign pertubed photometry from zp and dust on SOM
    """
    name = "PhotozZpDustPipe"
    config_options = CatEstimator.config_options.copy()
    config_options.update(chunk_size=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS, #Param(str, "redshift", msg="name of redshift column"),
                          hdf5_groupname=None,  #Param(str, "photometry", msg="hdf5_groupname for data"),
                          #groupname=SHARED_PARAMS, #Param(str, "", msg="hdf5_groupname for data"),
                          inputs=Param(list, default_input_names, msg="list of the names of columns to be used as inputs for deep data"),
                          input_errs=Param(list, default_err_names, msg="list of the names of columns containing errors on inputs for deep data"),
                          input_ra_col = Param(str, "ra", msg="column name for ra"),
                          input_dec_col = Param(str, "dec", msg="column name for dec"),
                          zero_points=Param(list, default_zero_points, msg="zero points for converting mags to fluxes for deep data, if needed"),
                          som_shape=Param(list, [32, 32], msg="shape for the deep som, must be a 2-element tuple"),
                          som_minerror=Param(float, 0.01, msg="floor placed on observational error on each feature in deep som"),
                          som_wrap=Param(bool, False, msg="flag to set whether the deep SOM has periodic boundary conditions"),
                          som_take_log=Param(bool, True, msg="flag to set whether to take log of inputs (i.e. for fluxes) for deep som"),
                          convert_to_flux=Param(bool, False, msg="flag for whether to convert input columns to fluxes for deep data"
                                                     "set to true if inputs are mags and to False if inputs are already fluxes"),
                          set_threshold=Param(bool, False, msg="flag for whether to replace values below a threshold with a set number"),
                          thresh_val=Param(float, 1.e-5, msg="threshold value for set_threshold for deep data"),
                          debug=Param(bool, False, msg="boolean reducing dataset size for quick debuggin"),
                          photometric_zeropoint = Param(bool, False, msg="photometric zero point uncertainty" ),
                          photometric_skybackground = Param(bool, False, msg="photometric dust uncertainty" )
                         )
    inputs = [('deep_model', ModelHandle), ('lhc_samples_zp', TableHandle), ('lhc_samples_sky', TableHandle),
              ('data', TableHandle)]
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

    def _process_chunk_perturb(self, start, end, data, first, LHC_id, LHC_sample_zp, LHC_sample_sky):
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
            if self.config.photometric_zeropoint:
                data_wide[:, j] = data_wide[:, j] * 10**LHC_sample_zp[j]

            #Also add the dust uncertainty here
            #Honestly I think all col are float 64, I use float32 here to be coherent with rail codes
            if self.config.photometric_skybackground:
                ra_col = self.config.input_ra_col
                dec_col = self.config.input_dec_col
                ra = np.array(data[ra_col], dtype=np.float32)
                dec = np.array(data[dec_col], dtype=np.float32)
                
                nside = hp.get_nside(LHC_sample_sky)

                theta = np.radians(90.0 - dec)
                phi = np.radians(ra)
                
                pixel_ids = hp.ang2pix(nside, theta, phi, nest=False)
                data_wide[:, j] = data_wide[:, j] * 10**LHC_sample_sky[pixel_ids]

                

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
        # --- Boyan: START PATCH ---
        # --- There is chunck padding and actual data mismatch ---
        # --- This is brute force solution to it. Problem probably lie in table_io ---
        # 1. If the iterator hands us a chunk completely past the file limit, ignore it.
        print('PATCH with resize chunk')
        print(self._input_length)
        
        if start >= self._input_length:
            return
            
        # 2. If the iterator's end index overshoots the file limit, trim the data to fit.
        if end > self._input_length:
            true_end = self._input_length
            true_size = true_end - start
            for key in output_chunk.keys():
                truncated_data = output_chunk[key][true_size:]
                print(f"Truncating from '{key}': {truncated_data}")
                
                output_chunk[key] = output_chunk[key][:true_size]
            end = true_end
        
        # --- END PATCH ---
        
        if first:
            name = "assignment"
            self._output_handle = self.add_handle(name, data=output_chunk)
            self._output_handle.initialize_write(self._input_length, communicator=self.comm)
        self._output_handle.set_data(output_chunk, partial=True)
        self._output_handle.write_chunk(start, end)

    def run(self):
        self.model = None
        self.model = self.open_model(**self.config)  # None
        #If we are doing neither zp nor dust map uncertainty, no need to assign soms
        if not self.config.photometric_zeropoint and not self.config.photometric_skybackground:
            return
        first = True
        iter1 = self.input_iterator('data') # here we assume no deep galaxy duplicates  (Will deal with this later)
        LHC_samples_zp = self.get_data('lhc_samples_zp').view(np.ndarray)['samples']
        LHC_samples_sky = self.get_data('lhc_samples_sky').view(np.ndarray)['samples']
        print(LHC_samples_zp.shape)
        print(LHC_samples_sky.shape)
        self._output_handle = None
        for s, e, test_data in iter1:
            print(f"Process {self.rank} running creator on chunk {s} - {e}", flush=True)
            self._process_chunk(s, e, test_data, first, np.max((len(LHC_samples_zp), len(LHC_samples_sky))))
            first = False
            for LHC_id, (LHC_sample_zp, LHC_sample_sky) in enumerate(zip(LHC_samples_zp, LHC_samples_sky)):
                print(LHC_sample_zp.shape)
                print(LHC_sample_sky.shape)
                self._process_chunk_perturb(s, e, test_data, first, LHC_id, LHC_sample_zp, LHC_sample_sky)
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

        
class Samplevariance(PipelineStage):

    name = "Samplevariance"
    inputs = []
    outputs = [("Sample_variance", TableHandle)] 
    parallel = False

    config_options = {
        "cosmoparameters": StageParameter(list, [70.0, 0.02242, 0.11933, 0.0, 0.0561, 0.0, 2.15E-9, 0.9665, 0.0],
            msg="" ),
         "zbins_min": StageParameter(float, 0.0, msg="minimum redshift for output grid"),
         "zbins_max": StageParameter(float, 5.0, msg="max redshift for output grid"),
         "zbins_dz": StageParameter(float, 0.01, msg="redshift difference for output grid"),
         "area": StageParameter(float, 20.0, msg="size of deep field in deg^2"),
         "num_points_z":StageParameter(int, 50, msg="number of point z"),
    
    }
    def run(self):
        camb_pars = camb.CAMBparams()
        cosmoparameters = self.config.cosmoparameters
        camb_pars.set_cosmology(H0=cosmoparameters[0],
                               ombh2=cosmoparameters[1],
                               omch2=cosmoparameters[2],
                               mnu=cosmoparameters[3],
                               tau=cosmoparameters[4],
                               omk=cosmoparameters[5])
        camb_pars.InitPower.set_params(As=cosmoparameters[6], 
                                       ns=cosmoparameters[7], 
                                       r=cosmoparameters[8])
        camb_pars.set_for_lmax(3500, lens_potential_accuracy=0)
        zbins = np.arange(self.config.zbins_min - self.config.zbins_dz / 2., self.config.zbins_max + self.config.zbins_dz, self.config.zbins_dz)
        zbins[zbins<0]=0.0
        camb_pars.set_matter_power(redshifts=np.sort(zbins)[::-1], kmax=100.0)
        camb_pars.NonLinear = model.NonLinear_both
        camb_pars.NonLinearModel.set_params(halofit_version='takahashi')
        
        # Set CAMB source terms (only density for simplicity)
        camb_pars.SourceTerms.counts_density = True
        camb_pars.SourceTerms.counts_evolve = False
        camb_pars.SourceTerms.counts_redshift = False
        camb_pars.SourceTerms.counts_velocity = False
        camb_pars.SourceTerms.counts_lensing = False
        camb_pars.SourceTerms.counts_radial = False
        camb_pars.SourceTerms.counts_timedelay = False
        camb_pars.SourceTerms.counts_ISW = False
        camb_pars.SourceTerms.counts_potential = False
        area_deg2 = self.config.area
        area_rad2 = area_deg2 / (180./np.pi)**2
        theta = np.arccos(1. - 0.5*area_rad2/np.pi)
        thetas = np.array([theta])
        z_binsc, sample_var, sample_var_var = sample_variance(
                camb_pars, 
                thetas, 
                zbins,
                nz_function=simple_galaxy_counts,
                bias_function=simple_bias, area_deg2=area_deg2, num_points_z = self.config.num_points_z 
        )
        filename = self.get_output(self.outputs[0][0])
        t= np.zeros(len(sample_var[0]), dtype=[("sample_var", '>f8', sample_var.shape[-1])])
        t['sample_var'] = sample_var
        table = apTable.Table(t)
        table = tables_io.convert(table, tables_io.types.NUMPY_FITS)
        filename, file_extension = os.path.splitext(filename)
        tables_io.write(table,filename, file_extension[1:])




#selection_nz_final_rushift_zpshift.py call functions_nzrealizations_Roman_pointz.py
class RunPZRealizationsPipe(CatEstimator):
    """
    Generates redshift realizations, where each is a possible redshift distribution given the uncertainty.
    Use SOM products and modeled uncertainty and saves to a HDFFile.
    """

    name = "RunPZRealizationsPipe"
    parallel = False
    config_options = CatEstimator.config_options.copy()
    config_options.update( 
        shot_noise = StageParameter(bool, False,
            msg="If you want to add shot noise from the limited number of galaxies in redshift and deep sample "),
        sample_variance= StageParameter(bool, False,
            msg="If you want to add sample variance from the limited area of redshift and deep field. Must add shot nose if you add sample variance" ),
        photometric_zeropoint_deep =  StageParameter(bool, False,
            msg="If you want to add photometric zero point uncertainty due to the deep field zero point offsets " ),
        photometric_zeropoint_wide =  StageParameter(bool, False,
            msg="If you want to add photometric zero point uncertainty due to the wide field zero point offsets " ), 
        photometric_skybackground_deep = StageParameter(bool, False,
            msg="If you want to add skybackground uncertainty on the deep field photometry " ),     
        photometric_skybackground_wide = StageParameter(bool, False,
            msg="If you want to add skybackground uncertainty on the wide field photometry " ),       
        redshift_sample_uncertainty =  StageParameter(bool, False,
            msg="The bias and uncertainty due to the use of photometric redshfit calibration sample" ), #If we use spec-only, don't need this
        num_lhc_points = StageParameter(int, 100,msg="number of lhc points we want to sample"),
        num_3sdir = StageParameter(int, 100,msg="number of 3sdir sampling we want to do"),
        bands = StageParameter(list, [], msg="photometric bands"),
        zbins_min=StageParameter(float, 0.0, msg="minimum redshift for output grid"),
        zbins_max=StageParameter(float, 6.0, msg="maximum redshift for output grid"),
        zbins_dz=StageParameter(float, 0.01, msg="delta z for defining output grid"),

    )
    inputs = [("deep_balrog_file", TableHandle), ("redshift_deep_balrog_file", TableHandle), ("deep_som", ModelHandle), ("wide_som", ModelHandle), ("pchat", Hdf5Handle), ("pcchat", Hdf5Handle), ("tomo_bin_assignment", Hdf5Handle), ("deep_cells_assignment_balrog_files_withzp", TableHandle), ("wide_cells_assignment_balrog_files_withzp", TableHandle), ("wide_cells_assignment_wide_files_withzp", TableHandle), ("sv_redshift_file", TableHandle),("sv_deep_file", TableHandle), ("deep_cells_assignment_balrog_files", TableHandle),("wide_cells_assignment_balrog_files", TableHandle), ("deep_cells_assignment_spec_files", TableHandle), ("wide_cells_assignment_spec_files", TableHandle)]
    outputs = [("photoz_realizations", TableHandle)]
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




    def run(self):
            deep_som = self.set_data("deep_som", self.config["deep_som"])
            wide_som = self.set_data("wide_som", self.config["wide_som"])
            sv_redshift_data = self.get_data('sv_redshift_file').view(np.ndarray)['sample_var']
            sv_deep_data = self.get_data('sv_deep_file').view(np.ndarray)['sample_var']
            shot_noise = self.config["shot_noise"]
            sample_variance = self.config["sample_variance"]
            redshift_sample_uncertainty = self.config["redshift_sample_uncertainty"]
            photometric_zeropoint_deep = self.config["photometric_zeropoint_deep"]
            photometric_zeropoint_wide = self.config["photometric_zeropoint_wide"]
            photometric_skybackground_deep = self.config["photometric_skybackground_deep"]
            photometric_skybackground_wide = self.config["photometric_skybackground_wide"]
            num_lhc_points = self.config["num_lhc_points"]
            num_3sdir = self.config["num_3sdir"]
            deep_balrog_data = self.get_data('deep_balrog_file').to_pandas()
            redshift_deep_balrog_data = self.get_data('redshift_deep_balrog_file').to_pandas()
            deep_cells_assignment_balrog_files_withzp = self.get_data('deep_cells_assignment_balrog_files_withzp')
            wide_cells_assignment_balrog_files_withzp = self.get_data('wide_cells_assignment_balrog_files_withzp')
            wide_cells_assignment_wide_files_withzp = self.get_data('wide_cells_assignment_wide_files_withzp')
            tomo_bin_assignment = self.get_data('tomo_bin_assignment')
            deep_cells_assignment_balrog = self.get_data("deep_cells_assignment_balrog_files")
            wide_cells_assignment_balrog = self.get_data("wide_cells_assignment_balrog_files")
            deep_cells_assignment_spec = self.get_data("deep_cells_assignment_spec_files")
            wide_cells_assignment_spec = self.get_data("wide_cells_assignment_spec_files")
            pchat = np.squeeze(self.get_data('pchat')['pchat'])
            pcchat = self.get_data('pcchat')['pc_chat']
            print(np.shape(pchat),pchat)
            print(np.shape(pcchat),pcchat)
            assert(pcchat.shape[1]==len(pchat))
            bands = self.config.bands 
            deep_som_size = int(deep_cells_assignment_balrog['som_size'][0])
            wide_som_size = int(wide_cells_assignment_balrog['som_size'][0])

            if shot_noise == False:
                print('Sorry must have shot noise for now.')
            if redshift_sample_uncertainty == True:
                print('Sorry dont have redshift_sample_uncertainty yet.')
            #if photometric_zeropoint_wide == True:
            #    print('Sorry dont have photometric_zeropoint_wide yet.')
            #if photometric_skybackground_wide == True:
            #    print('Sorry dont have photometric_skybackground_wide yet.')
            zbins = np.arange(self.config.zbins_min - self.config.zbins_dz / 2., self.config.zbins_max + self.config.zbins_dz, self.config.zbins_dz)
                
            nz_realizations = get_realizations(sv_redshift_data, sv_deep_data, shot_noise, sample_variance, redshift_sample_uncertainty, photometric_zeropoint_deep, photometric_zeropoint_wide, photometric_skybackground_deep, photometric_skybackground_wide, num_lhc_points, num_3sdir, deep_balrog_data, redshift_deep_balrog_data, deep_som_size, wide_som_size, pchat, pcchat, tomo_bin_assignment, deep_cells_assignment_balrog_files_withzp, wide_cells_assignment_balrog_files_withzp, wide_cells_assignment_wide_files_withzp, deep_cells_assignment_balrog, wide_cells_assignment_balrog, deep_cells_assignment_spec, wide_cells_assignment_spec, bands, self.config.redshift_col, zbins)
            photoz_realizations = {}
            for idx, nz_r in enumerate(nz_realizations):
                photoz_realizations["LHC_id_{0}".format(idx)]= nz_r
            self.add_data('photoz_realizations', photoz_realizations)
    def estimate(self, spec_data, cell_deep_spec_data):
        self.run()
        self.finalize()


