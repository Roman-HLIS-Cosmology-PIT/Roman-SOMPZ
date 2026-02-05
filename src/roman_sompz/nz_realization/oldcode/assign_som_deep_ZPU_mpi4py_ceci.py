import numpy as np

def assign_som_zpu(deep_data, balrog_data, deep_som, LHC_samples, output_file):

    bands = ['LSST_u','LSST_g','LSST_r','LSST_i','LSST_z','LSST_y', 'Y', 'J', 'H']
    LHC_samples_ZPU = LHC_samples['ZPU']


    
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    ###########################################################################################
    ### Load flux and SOM
    ###########################################################################################
    
    if rank == 0:
        fluxes, fluxerrs, fields, deep_id = get_fluxes(balrog_data, deep_data)
        #unique for each deep galaxy, will be matched later to deep_balrog file when generating realizations
        #Shape should be less than 1472127 (all deep before matching)
        np.save(outpath + 'deep_data_ID.npy', deep_id)
    else:
        fluxes, fluxerrs = None, None


    # Load SOM, No training included    
    metric = ns.AsinhMetric(lnScaleSigma=0.4,lnScaleStep=0.03)
    som = ns.NoiseSOM(metric,None,None, \
        learning = None, \
        shape=(64,64), \
        wrap=False,logF=True, \
        initialize=deep_som, \
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
    ### Perturb fluxes in all fields and run assignments in parallel
    ###########################################################################################
    
    
    # Generate photometric uncertainty shifts and re-assign on deep SOM
    for LHC_id, LHC_sample in enumerate(LHC_samples):             # LHC_samples : (100, 28)
        
        filename = "%s/som_deep_64x64_assign_LHC%d.npz"%(outpath, LHC_id)
        
        if os.path.exists(filename):
            continue
        
        t0 = time.time()
        
        if rank == 0:
            print("Running LHC sample %d"%LHC_id)
            
            fluxes_p = np.zeros((len(fluxes),len(bands)))
            fluxerrs_p = np.zeros((len(fluxes),len(bands)))
    
            #['COSMOS', 'C3', 'E2', 'X3']
            zp_vals_COSMOS = LHC_samples_ZPU['COSMOS']
            zp_vals_C3 = LHC_samples_ZPU['C3'] #LHC_sample[0:8]
            zp_vals_X3 = LHC_samples_ZPU['X3'] #LHC_sample[8:16]
            zp_vals_E2 = LHC_samples_ZPU['E3'] #LHC_sample[16:24]
    
            for i, band in enumerate(bands):
                print(i,band)
                _mask = (fields_notcosmos == 'COSMOS')
                fluxes_p[_mask,i] = fluxes[_mask,i] * 10**zp_vals_COSMOS[i]
                fluxerrs_p[_mask,i] = fluxerrs[_mask,i] * 10**zp_vals_COSMOS[i]
                
                _mask = (fields_notcosmos == 'C3')
                fluxes_p[_mask,i] = fluxes[_mask,i] * 10**zp_vals_C3[i]
                fluxerrs_p[_mask,i] = fluxerrs[_mask,i] * 10**zp_vals_C3[i]
    
                _mask = (fields_notcosmos == 'X3')
                fluxes_p[_mask,i] = fluxes[_mask,i] * 10**zp_vals_X3[i]
                fluxerrs_p[_mask,i] = fluxerrs[_mask,i] * 10**zp_vals_X3[i]
    
                _mask = (fields_notcosmos == 'E2')
                fluxes_p[_mask,i] = fluxes[_mask,i] * 10**zp_vals_E2[i]
                fluxerrs_p[_mask,i] = fluxerrs[_mask,i] * 10**zp_vals_E2[i]
        else:
            fluxes_p, fluxerrs_p = None, None
         
    
        # Broadcast data to all processes
        fluxes_p = comm.bcast(fluxes_p, root=0)
        fluxerrs_p = comm.bcast(fluxerrs_p, root=0)
        
        # Split the data among processes
        fluxes_chunks = np.array_split(fluxes_p, size)
        fluxerrs_chunks = np.array_split(fluxerrs_p, size)
    
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
            
            cells_final = cells_zp
    
            print(cells_zp.shape)
            print(cells_final.shape)
        
            np.savez(filename,cells=cells_final)
            
            t1 = time.time()
            print(t1-t0)
    
    
