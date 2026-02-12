from scm_pipeline import PipelineStage
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
import os
from .types import TextFile, YamlFile
import pyarrow.parquet as pq





class Inputcheck(PipelineStage):

    name = "Inputcheck"
    inputs = [("catfile", TableHandle)]
    outputs = [("checked",TextFile)] 
    parallel = False

    config_options = {
                "inputs": Param(list, [], msg="list of the names of columns to be used as inputs for data"),
                "input_errs": Param(list, [], msg="list of the names of columns containing errors on inputs for data"),  
    }
    def run(self):
        for inp, _ in self.inputs:
            filename = self.get_input(inp)
            print(f"check {filename}")
            t= pq.read_table(filename)
            t= t.to_pandas()
            bad = False
            for item in self.config.inputs:
                cond = np.isfinite(np.min(np.array(t[item]))*np.max(np.array(t[item])))
                print(item, np.min(np.array(t[item])), np.max(np.array(t[item])), cond)
                cond = cond&(np.all(t[item]>0))
                if cond ==False:
                    bad =True
                break
            if bad:
                break
        if not bad:
            filename = self.get_output(self.outputs[0][0])
            open(filename, "w").write("True")

        else:
            assert(0)
