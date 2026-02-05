from scm_pipeline import PipelineStage
from rail.core.data import TableHandle, ModelHandle, QPHandle, Hdf5Handle
from rail.estimation.estimator import CatEstimator, CatInformer
from roman_sompz.nz_realization.Roman_selection_nz_final_rushift_zpshift_ceci import get_realizations
from ceci.config import StageParameter
import numpy as np
#from ceci_example.types import NpyFile, HDFFile
# We need to define NpyFile etc we want to use inside ceci_example.types, Or not??? get_input don't use this function and we open in run ourselves


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
        redshift_sample_uncertainty =  StageParameter(bool, False,
            msg="The bias and uncertainty due to the use of photometric redshfit calibration sample" ), #If we use spec-only, don't need this
        photometric_zeropoint_wide =  StageParameter(bool, False,
            msg="If you want to add photometric zero point uncertainty due to the wide field zero point offsets " ),  #not implemented
        photometric_skybackground_wide = StageParameter(bool, False,
            msg="If you want to add skybackground uncertainty on the wide field photometry " ), #not implemented       
        num_lhc_points = StageParameter(int, 100,msg="number of lhc points we want to sample"),
        num_3sdir = StageParameter(int, 100,msg="number of 3sdir sampling we want to do"),
        bands = StageParameter(list, [], msg="photometric bands"),
          zbins_min=StageParameter(float, 0.0, msg="minimum redshift for output grid"),
          zbins_max=StageParameter(float, 6.0, msg="maximum redshift for output grid"),
          zbins_dz=StageParameter(float, 0.01, msg="delta z for defining output grid"),

    )
    inputs = [("deep_balrog_file", TableHandle), ("redshift_deep_balrog_file", TableHandle), ("deep_som", ModelHandle), ("wide_som", ModelHandle), ("pchat", Hdf5Handle), ("pcchat", Hdf5Handle), ("tomo_bin_assignment", Hdf5Handle), ("deep_cells_assignment_balrog_files_withzp", TableHandle), ("sv_redshift_file", TableHandle),("sv_deep_file", TableHandle), ("deep_cells_assignment_balrog_files", TableHandle),("wide_cells_assignment_balrog_files", TableHandle)]
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
            photometric_zeropoint_deep = self.config["photometric_zeropoint_deep"]
            redshift_sample_uncertainty = self.config["redshift_sample_uncertainty"]
            photometric_zeropoint_wide = self.config["photometric_zeropoint_wide"]
            photometric_skybackground_wide = self.config["photometric_skybackground_wide"]
            num_lhc_points = self.config["num_lhc_points"]
            num_3sdir = self.config["num_3sdir"]
            deep_balrog_data = self.get_data('deep_balrog_file').to_pandas()
            redshift_deep_balrog_data = self.get_data('redshift_deep_balrog_file').to_pandas()
            deep_cells_assignment_balrog_files_withzp = self.get_data('deep_cells_assignment_balrog_files_withzp')
            tomo_bin_assignment = self.get_data('tomo_bin_assignment')
            deep_cells_assignment_balrog = self.get_data("deep_cells_assignment_balrog_files")
            wide_cells_assignment_balrog = self.get_data("wide_cells_assignment_balrog_files")
            pchat = np.squeeze(self.get_data('pchat')['pz_chat'])
            pcchat = self.get_data('pcchat')['pc_chat']
            print(pchat.shape, pcchat.shape)
            bands = self.config.bands 
            deep_som_size = int(deep_cells_assignment_balrog['som_size'][0])
            wide_som_size = int(wide_cells_assignment_balrog['som_size'][0])

            if shot_noise == False:
                print('Sorry must have shot noise for now.')
            if redshift_sample_uncertainty == True:
                print('Sorry dont have redshift_sample_uncertainty yet.')
            if photometric_zeropoint_wide == True:
                print('Sorry dont have photometric_zeropoint_wide yet.')
            if photometric_skybackground_wide == True:
                print('Sorry dont have photometric_skybackground_wide yet.')
            zbins = np.arange(self.config.zbins_min - self.config.zbins_dz / 2., self.config.zbins_max + self.config.zbins_dz, self.config.zbins_dz)
                
            nz_realizations = get_realizations(sv_redshift_data, sv_deep_data, shot_noise, sample_variance, photometric_zeropoint_deep, redshift_sample_uncertainty, photometric_zeropoint_wide, photometric_skybackground_wide, num_lhc_points, num_3sdir, deep_balrog_data, redshift_deep_balrog_data, deep_som_size, wide_som_size, pchat, pcchat, tomo_bin_assignment, deep_cells_assignment_balrog_files_withzp, deep_cells_assignment_balrog, wide_cells_assignment_balrog, bands, self.config.redshift_col, zbins)
            photoz_realizations = {}
            for idx, nz_r in enumerate(nz_realizations):
                photoz_realizations["LHC_id_{0}".format(idx)]= nz_r
            self.add_data('photoz_realizations', photoz_realizations)
    def estimate(self, spec_data, cell_deep_spec_data):
        self.run()
        self.finalize()


