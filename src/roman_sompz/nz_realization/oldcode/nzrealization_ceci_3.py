from scm_pipeline import PipelineStage
from rail.core.data import TableHandle, ModelHandle, QPHandle, Hdf5Handle
from rail.estimation.estimator import CatEstimator, CatInformer
#from ceci_example.types import NpyFile, HDFFile
# We need to define NpyFile etc we want to use inside ceci_example.types, Or not??? get_input don't use this function and we open in run ourselves


#selection_nz_final_rushift_zpshift.py call functions_nzrealizations_Roman_pointz.py
class RunPZRealizationsPipe(PipelineStage):
    """
    Generates redshift realizations, where each is a possible redshift distribution given the uncertainty.
    Use SOM products and modeled uncertainty and saves to a HDFFile.
    """

    name = "RunPZRealizationsPipe"
    inputs = [("deep_balrog_file", ParquetFile), ("redshift_deep_balrog_file", ParquetFile), ("deep_som", PklFile), ("wide_som", PklFile), ("pchat", HDFFile), ("pcchat", HDFFile), ("tomo_bin_assignment", HDFFile), ("deep_cells_assignment_balrog_files_withzp", HDFFile), ("sv_redshift_file", NPZFile),("sv_deep_file", NPZFile)]
    outputs = [("photoz_realizations", ParquetFile)]
    parallel = False
    
    config_options = {   
        "shot_noise": StageParameter(bool, False,
            msg="If you want to add shot noise from the limited number of galaxies in redshift and deep sample "),
        "sample_variance": StageParameter(bool, False,
            msg="If you want to add sample variance from the limited area of redshift and deep field. Must add shot nose if you add sample variance" ),
        "photometric_zeropoint_deep": StageParameter(bool, False,
            msg="If you want to add photometric zero point uncertainty due to the deep field zero point offsets " ),
        "redshift_sample_uncertainty": StageParameter(bool, False,
            msg="The bias and uncertainty due to the use of photometric redshfit calibration sample" ), #If we use spec-only, don't need this
        "photometric_zeropoint_wide": StageParameter(bool, False,
            msg="If you want to add photometric zero point uncertainty due to the wide field zero point offsets " ),  #not implemented
        "photometric_skybackground_wide": StageParameter(bool, False,
            msg="If you want to add skybackground uncertainty on the wide field photometry " ), #not implemented       
        "num_lhc_points":StageParameter(int, 100,msg="number of lhc points we want to sample"),
        "num_3sdir":StageParameter(int, 100,msg="number of 3sdir sampling we want to do")
    }


    def run(self):
        for inp, _ in self.inputs:
            sv_redshift_filename = self.get_input("sv_redshift_file")
            sv_redshift_data = np.load(sv_redshift_filename)
            
            sv_deep_filename = self.get_input("sv_deep_file")
            sv_deep_data = np.load(sv_deep_filename)
            
            
            shot_noise = self.config["shot_noise"]
            sample_variance = self.config["sample_variance"]
            photometric_zeropoint_deep = self.config["photometric_zeropoint_deep"]
            redshift_sample_uncertainty = self.config["redshift_sample_uncertainty"]
            photometric_zeropoint_wide = self.config["photometric_zeropoint_wide"]
            photometric_skybackground_wide = self.config["photometric_skybackground_wide"]
            
            num_lhc_points = self.config["num_lhc_points"]
            num_3sdir = self.config["num_3sdir"]
            

            if shot_noise == False:
                print('Sorry must have shot noise for now.')
            if redshift_sample_uncertainty == True:
                print('Sorry dont have redshift_sample_uncertainty yet.')
            if photometric_zeropoint_wide == True:
                print('Sorry dont have photometric_zeropoint_wide yet.')
            if photometric_skybackground_wide == True:
                print('Sorry dont have photometric_skybackground_wide yet.')
                
            nz_realizations = get_realizations(sv_redshift_data, sv_deep_data, shot_noise, sample_variance, photometric_zeropoint_deep, redshift_sample_uncertainty, photometric_zeropoint_wide, photometric_skybackground_wide, num_lhc_points, num_3sdir, deep_balrog_data, redshift_deep_balrog_data, deep_som, wide_som, pchat, pcchat, tomo_bin_assignment, deep_cells_assignment_balrog_files_withzp)


