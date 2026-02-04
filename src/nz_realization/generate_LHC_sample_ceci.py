import numpy as np

def generate_LHC_points(deepfield_zeropoint_data, photometric_zeropoint_deep, redshift_sample_uncertainty, photometric_zeropoint_wide, photometric_skybackground_wide, num_lhc_points, LHC_samples_filename):
    #Currently just use Gaussian random, will change to Latin Hypercube sampling later
    if photometric_zeropoint_deep:
        ZPU_result = np.random.normal(loc=0, scale=deepfield_zeropoint_data[:, np.newaxis], size=(len(deepfield_zeropoint_data), num_lhc_points))
    else:
        ZPU_result = np.zeros(1)
    np.save('LHC_samples.npy', {'ZPU': ZPU_result})
