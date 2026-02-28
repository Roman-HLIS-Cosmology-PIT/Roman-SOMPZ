import healpy as hp
import numpy as np

#Currently just use Gaussian random, will change to Latin Hypercube sampling later
def generate_LHC_points(deepfield_zeropoint_data, widefield_zeropoint_data, sfd_map, lss_error_map, photometric_zeropoint_deep, redshift_sample_uncertainty, photometric_zeropoint_wide, photometric_skybackground_deep, photometric_skybackground_wide, num_lhc_points):
    print('photometric_zeropoint_wide', photometric_zeropoint_wide)
    LHC_sample = {'deep_zp': None, 'wide_zp': None, 'sky': None}
    
    if photometric_zeropoint_deep:
        deep_zp_result = np.random.normal(loc=0, scale=deepfield_zeropoint_data[np.newaxis, :], size=(num_lhc_points, len(deepfield_zeropoint_data)))
        LHC_sample['deep_zp'] = np.array(deep_zp_result)
    else:
        LHC_sample['deep_zp'] = np.zeros((0,len(deepfield_zeropoint_data)))
        
    if photometric_zeropoint_wide:
        wide_zp_result = np.random.normal(loc=0, scale=widefield_zeropoint_data[np.newaxis, :], size=(num_lhc_points, len(widefield_zeropoint_data)))
        LHC_sample['wide_zp'] = np.array(wide_zp_result)
    else:
        LHC_sample['wide_zp'] = np.zeros((0,len(widefield_zeropoint_data)))
        
    #Shall use same dust uncertainty map for deep and wide
    if photometric_skybackground_deep or photometric_skybackground_wide:        
        gauss_samples = np.random.normal(loc=0, scale=1, size=num_lhc_points)
        # This 10% uncertainty is a coherent uncertainty
        sky_uncertainty = np.array([g * 0.1 * sfd_map for g in gauss_samples])
        # This uncertainty is due to dust map lss correction
        lss_uncertainty = np.random.normal(loc=0, scale=lss_error_map, size=(num_lhc_points, len(lss_error_map)))
        print(np.shape(sky_uncertainty), np.shape(lss_uncertainty))
        sky_result = sky_uncertainty + lss_uncertainty
        LHC_sample['sky'] = np.array(sky_result)
    else:
        LHC_sample['sky'] = np.zeros((0,len(lss_error_map)))

    return LHC_sample