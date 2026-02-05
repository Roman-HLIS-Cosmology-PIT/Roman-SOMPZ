from scm_pipeline import PipelineStage
from rail.core.data import TableHandle, ModelHandle, QPHandle, Hdf5Handle
from rail.estimation.estimator import CatEstimator, CatInformer
from roman_sompz.nz_realization.generate_LHC_sample_ceci import generate_LHC_points
import astropy.table as apTable
import tables_io
from ceci.config import StageParameter 
import numpy as np
import os
#from ceci_example.types import NpyFile, HDFFile
# We need to define NpyFile etc we want to use inside ceci_example.types, Or not??? get_input don't use this function and we open in run ourselves
class PreparePZRealizationsPipe(PipelineStage):
    """
    Generates LHC sampling points for the uncertainty beside sample variance and shot noise.
    Saves to a NPZfile with LHCsampling points.
    """

    name = "PreparePZRealizationsPipe"
    inputs = []
    outputs = [("LHC_samples",TableHandle)]
    parallel = False
    
    config_options = {
        "photometric_zeropoint_deep": StageParameter(bool, False,
            msg="If you want to add photometric zero point uncertainty due to the deep field zero point offsets " ),
        "redshift_sample_uncertainty": StageParameter(bool, False,
            msg="The bias and uncertainty due to the use of photometric redshfit calibration sample" ), #If we use spec-only, don't need this
        "photometric_zeropoint_wide": StageParameter(bool, False,
            msg="If you want to add photometric zero point uncertainty due to the wide field zero point offsets " ),  #not implemented
        "photometric_skybackground_wide": StageParameter(bool, False,
            msg="If you want to add skybackground uncertainty on the wide field photometry " ), #not implemented       
        "deepfield_zeropoint_data": StageParameter(list, [], msg="zero point uncertainty" ), #not implemented
        "num_lhc_points":StageParameter(int, 100,msg="number of lhc points we want to sample")
    }


    def run(self):
            photometric_zeropoint_deep = self.config["photometric_zeropoint_deep"]
            redshift_sample_uncertainty = self.config["redshift_sample_uncertainty"]
            photometric_zeropoint_wide = self.config["photometric_zeropoint_wide"]
            photometric_skybackground_wide = self.config["photometric_skybackground_wide"]
            deepfield_zeropoint_data=  self.config["deepfield_zeropoint_data"]
            
            num_lhc_points = self.config["num_lhc_points"]
            
            samples = generate_LHC_points(np.array(deepfield_zeropoint_data), photometric_zeropoint_deep, redshift_sample_uncertainty, photometric_zeropoint_wide, photometric_skybackground_wide, num_lhc_points)
            filename = self.get_output(self.outputs[0][0])
            t= np.zeros(len(samples), dtype=[("samples", '>f8', samples.shape[-1])])
            t['samples'] = samples
            table = apTable.Table(t)
            table = tables_io.convert(table, tables_io.types.NUMPY_FITS)
            filename, file_extension = os.path.splitext(filename)
            tables_io.write(table,filename, file_extension[1:])

            
         
