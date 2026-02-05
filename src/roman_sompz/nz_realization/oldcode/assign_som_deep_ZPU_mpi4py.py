import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('/global/cfs/cdirs/des/boyan/sompz_y6/sompz')
import NoiseSOM as ns
import multiprocessing as mp
import time
import warnings
import os
import yaml
from functions_nzrealizations import*
from mpi4py import MPI


outpath = '/global/cfs/cdirs/des/boyan/sompz_output/y6_data_10000Tile_final_unblind/nz_realizations/sub_files/ZPU_assignment/'


cfgfile = '/global/cfs/cdirs/des/boyan/sompz_y6/y6_sompz_10000Tile_v6_final_noshear_unblind.cfg'
with open(cfgfile, 'r') as fp:
    cfg = yaml.safe_load(fp)

path_cats_src = cfg['out_dir']
balrog_file= cfg['deep_balrog_file']
deep_file=cfg['deep_file']

bands = ['U','G','R','I','Z','J','H','K']

LHC_samples = np.load("./LHC_samples.npy")

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###########################################################################################
### Load flux and SOM
###########################################################################################

if rank == 0:
    fluxes, fluxerrs, fields, deep_id = get_fluxes(balrog_file, deep_file)
    #unique for each deep galaxy, will be matched later to deep_balrog file when generating realizations
    #Shape should be less than 1472127 (all deep before matching)
    np.save(outpath + 'deep_data_ID.npy', deep_id)
else:
    fluxes, fluxerrs = None, None


# Load SOM, No training included    
metric = ns.AsinhMetric(lnScaleSigma=0.4,lnScaleStep=0.03)
som_weights = np.load("%s/som_deep_64_64.npy"%path_cats_src,allow_pickle=True)  #(4096, 8)
som = ns.NoiseSOM(metric,None,None, \
    learning = None, \
    shape=(64,64), \
    wrap=False,logF=True, \
    initialize=som_weights, \
    minError=0.02)




###########################################################################################
### Fiducial cell assignment
########################################################################################### 


filename = "%s/som_deep_64x64_default.npz"%(outpath)
    
if not os.path.exists(filename):
    if rank == 0:
        print("Running Default Assignment")

    # Broadcast data to all processes
    fluxes = comm.bcast(fluxes, root=0)
    fluxerrs = comm.bcast(fluxerrs, root=0)
    
    # Split the data among processes
    fluxes_chunks = np.array_split(fluxes, size)
    fluxerrs_chunks = np.array_split(fluxerrs, size)

    # Each process gets its own chunk of data
    local_fluxes = fluxes_chunks[rank]
    local_fluxerrs = fluxerrs_chunks[rank]

    # Each process runs the classify function on its local data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local_cells_default = som.classify(local_fluxes, local_fluxerrs)[0] 

    # Gather results from all processes
    cells_default = comm.gather(local_cells_default, root=0)

    # Combine results if rank 0
    if rank == 0:
        cells_default = np.concatenate(cells_default)
        np.savez(filename,cells=cells_default)



###########################################################################################
### Perturb fluxes in all fields besides COSMOS and run assignments in parallel
###########################################################################################

if rank == 0:
    iscosmos = (fields == 'COSMOS')
    indices = np.arange(len(fluxes))

    fluxes_notcosmos = fluxes[~iscosmos]
    fluxerrs_notcosmos = fluxerrs[~iscosmos]
    fields_notcosmos = fields[~iscosmos]
    indices_notcosmos = indices[~iscosmos]




    #Cell assignment of deep fields
    #This should be in the same order as the flux and flexerr in this file, and this contributes to the cell assignment of COSMOS
    cells_default = np.load("%s/som_deep_64x64_default.npz"%outpath)['cells'] 





# Generate photometric uncertainty shifts and re-assign on deep SOM
for LHC_id, LHC_sample in enumerate(LHC_samples):             # LHC_samples : (100, 28)
    
    filename = "%s/som_deep_64x64_assign_LHC%d.npz"%(outpath, LHC_id)
    
    if os.path.exists(filename):
        continue
    
    t0 = time.time()
    
    if rank == 0:
        print("Running LHC sample %d"%LHC_id)
        
        fluxes = np.zeros((len(fluxes_notcosmos),len(bands)))
        fluxerrs = np.zeros((len(fluxes_notcosmos),len(bands)))

        ['C3', 'COSMOS', 'E2', 'X3']
        zp_vals_C3 = LHC_sample[0:8]
        zp_vals_X3 = LHC_sample[8:16]
        zp_vals_E2 = LHC_sample[16:24]

        for i,band in enumerate(bands):
            print(i,band)
            _mask = (fields_notcosmos == 'C3')
            fluxes[_mask,i] = fluxes_notcosmos[_mask,i] * 10**zp_vals_C3[i]
            fluxerrs[_mask,i] = fluxerrs_notcosmos[_mask,i] * 10**zp_vals_C3[i]

            _mask = (fields_notcosmos == 'X3')
            fluxes[_mask,i] = fluxes_notcosmos[_mask,i] * 10**zp_vals_X3[i]
            fluxerrs[_mask,i] = fluxerrs_notcosmos[_mask,i] * 10**zp_vals_X3[i]

            _mask = (fields_notcosmos == 'E2')
            fluxes[_mask,i] = fluxes_notcosmos[_mask,i] * 10**zp_vals_E2[i]
            fluxerrs[_mask,i] = fluxerrs_notcosmos[_mask,i] * 10**zp_vals_E2[i]
    else:
        fluxes, fluxerrs = None, None
     

    # Broadcast data to all processes
    fluxes = comm.bcast(fluxes, root=0)
    fluxerrs = comm.bcast(fluxerrs, root=0)
    
    # Split the data among processes
    fluxes_chunks = np.array_split(fluxes, size)
    fluxerrs_chunks = np.array_split(fluxerrs, size)

    # Each process gets its own chunk of data
    local_fluxes = fluxes_chunks[rank]
    local_fluxerrs = fluxerrs_chunks[rank]

    # Each process runs the classify function on its local data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local_cells_zp = som.classify(local_fluxes, local_fluxerrs)[0] 

    # Gather results from all processes
    cells_zp = comm.gather(local_cells_zp, root=0)

    # Combine results if rank 0
    if rank == 0:
        cells_zp= np.concatenate(cells_zp)
        
        # COSMOS BMU assignment from cells_default
        # non-COSMOS from cell_test after adding ZPU
        cells_final = np.zeros_like(cells_default)
        cells_final[iscosmos] = cells_default[iscosmos]
        cells_final[indices_notcosmos] = cells_zp

        print(cells_zp.shape)
        print(cells_final.shape)
    
        np.savez(filename,cells=cells_final)
        
        t1 = time.time()
        print(t1-t0)


