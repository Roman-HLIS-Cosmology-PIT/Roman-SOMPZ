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
    inputs = [("deep_balrog_file", ParquetFile), ("redshift_deep_balrog_file", ParquetFile), ("deep_som", PklFile), ("wide_som", PklFile), ("pchat", HDFFile), ("pcchat", HDFFile), ("tomo_bin_assignment", HDFFile), ("deep_cells_assignment_balrog_file", HDFFile)]
    outputs = [("photoz_realizations", ParquetFile)]
    parallel = False
    
    config_options = {   
    }


    def run(self):
        for inp, _ in self.inputs:
            filename = self.get_input(inp)
            print(f"    PZRealizationPipe reading from {filename}")
            open(filename)

            #open all the input files

            nz_realizations = get_realizations(input_files)


