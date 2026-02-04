from scm_pipeline import PipelineStage
from rail.core.data import TableHandle, ModelHandle, QPHandle, Hdf5Handle
from rail.estimation.estimator import CatEstimator, CatInformer
#from ceci_example.types import NpyFile, HDFFile
# We need to define NpyFile etc we want to use inside ceci_example.types, Or not??? get_input don't use this function and we open in run ourselves

class PreparePZRealizationsPipe(PipelineStage):
    """
    Generates LHC sampling points for the uncertainty beside sample variance and shot noise.
    Saves to a NPZfile with LHCsampling points.
    """

    name = "PreparePZRealizationsPipe"
    inputs = [("deepfield_zeropoint", NpyFile)]
    outputs = [("LHC_samples", NpyFile)]
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
        "num_3sdir":StageParameter(int, 100,msg="number of 3sdir sampling we want to do"),
    }


    def run(self):
        for inp, _ in self.inputs:
            deepfield_zeropoint_filename = self.get_input("deepfield_zeropoint")
            deepfield_zeropoint_data = np.load(deepfield_zeropoint_filename)
            
            shot_noise = self.config["shot_noise"]
            sample_variance = self.config["sample_variance"]
            photometric_zeropoint_deep = self.config["photometric_zeropoint_deep"]
            redshift_sample_uncertainty = self.config["redshift_sample_uncertainty"]
            photometric_zeropoint_wide = self.config["photometric_zeropoint_wide"]
            photometric_skybackground_wide = self.config["photometric_skybackground_wide"]
            
            num_lhc_points = self.config["num_lhc_points"]
            num_3sdir = self.config["num_3sdir"]
            
            generate_LHC_points(deepfield_zeropoint_data, shot_noise, sample_variance, photometric_zeropoint_deep, redshift_sample_uncertainty, photometric_zeropoint_wide, photometric_skybackground_wide, num_lhc_points )
            
            #
            save_LHC_points here
            num_3sdir 
