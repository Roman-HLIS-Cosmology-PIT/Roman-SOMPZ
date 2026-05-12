import healpy as hp
import numpy as np
from scipy.stats import qmc
from scipy.stats import norm

def generate_LHC_points(deepfield_zeropoint_data, widefield_zeropoint_data, sfd_map, lss_error_map, photometric_zeropoint_deep, redshift_sample_uncertainty, photometric_zeropoint_wide, photometric_skybackground_deep, photometric_skybackground_wide, num_lhc_points):
    LHC_sample = {'deep_zp': None, 'wide_zp': None, 'sky': None}

    len_deepzp = 0
    len_widezp = 0
    len_sky = 0

    ## Gathering the info we have
    LHC_std = []
    if photometric_zeropoint_deep:
        LHC_std += deepfield_zeropoint_data.tolist()
        len_deepzp = len(deepfield_zeropoint_data)
    if photometric_zeropoint_wide:
        LHC_std += widefield_zeropoint_data.tolist()
        len_widezp = len(widefield_zeropoint_data)
    if photometric_skybackground_deep or photometric_skybackground_wide:   
        LHC_std += [1.0]
        len_sky = 1
        
    LHC_std = np.array(LHC_std)
    
    num_parameters = len(LHC_std)

    if num_parameters!=0:
        #LHC generation
        sampler = qmc.LatinHypercube(d=num_parameters)
        sample = sampler.random(n=num_lhc_points)
        
        # Take this uniform distribution as cdf
        # Transform to pdf
        sample_gaussian = norm.ppf(sample, loc=0, scale=LHC_std)
        
        
    if photometric_zeropoint_deep:
        LHC_sample['deep_zp'] = np.array(sample_gaussian[:,0:len_deepzp])
    else:
        LHC_sample['deep_zp'] = np.zeros((num_lhc_points,len_deepzp))
    
    if photometric_zeropoint_wide:
        LHC_sample['wide_zp'] = np.array(sample_gaussian[:,len_deepzp : len_deepzp+len_widezp])
    else:
        LHC_sample['wide_zp'] = np.zeros((num_lhc_points,len_widezp))
    
    if photometric_skybackground_deep or photometric_skybackground_wide:   
        gauss_samples = np.array(sample_gaussian[:, len_deepzp+len_widezp: len_deepzp+len_widezp+len_sky].flatten())
        # This 10% uncertainty is a coherent uncertainty
        sky_uncertainty = np.array([g * 0.1 * sfd_map for g in gauss_samples])
        # This uncertainty is due to dust map lss correction
        # Skipping LHS here: at ~100,000 dimensions, LHS loses its space-filling 
        # advantage and becomes a massive, unnecessary computational bottleneck.
        lss_uncertainty = np.random.normal(loc=0, scale=lss_error_map, size=(num_lhc_points, len(lss_error_map)))
        print(np.shape(sky_uncertainty), np.shape(lss_uncertainty))
        sky_result = sky_uncertainty + lss_uncertainty
        LHC_sample['sky'] = np.array(sky_result)
    else:
        LHC_sample['sky'] = np.zeros((num_lhc_points,len(lss_error_map)))

    return LHC_sample
