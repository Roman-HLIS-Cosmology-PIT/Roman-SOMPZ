"""
Minimal working example for sample variance calculation
Area: 20 deg^2
"""

import numpy as np
import scipy.integrate
import scipy.special
from scipy.interpolate import UnivariateSpline
import sys
import os
from tqdm import tqdm

import camb
from camb import model
from camb.sources import SplinedSourceWindow

################################################################################
# Helper functions
################################################################################

def smooth_tophat_function(x, y0, y1, mean_x, width, delta_x):
    """Smoothed top hat function"""
    def smooth_step(x, y0, y1, xt, delta_x):
        return (y1-y0)/(1.+np.exp(-(x-xt)/delta_x)) + y0
    
    return (y1-y0)*(smooth_step(x, y0=0., y1=1, xt=mean_x-width/2., delta_x=delta_x) -
                    smooth_step(x, y0=0., y1=1, xt=mean_x+width/2., delta_x=delta_x)) + y0

def get_Cls(camb_results):
    """Extract Cls from CAMB results"""
    spectra_dict = camb_results.get_source_cls_dict(raw_cl=True)
    spectra_results = []
    for i in range(1, len(camb_results.Params.SourceWindows)+1):
        temp = []
        for j in range(1, len(camb_results.Params.SourceWindows)+1):
            temp.append(spectra_dict['W'+str(i)+'x'+'W'+str(j)][2:])
        spectra_results.append(temp)
    spectra_results = np.array(spectra_results)
    spectra_results = np.swapaxes(spectra_results, 0, -1)
    ell = np.arange(2, spectra_results.shape[0]+2)
    return ell, spectra_results

def harmonic_smooth_filter(theta, ell):
    """Top-hat filter weights"""
    x = np.cos(theta)
    A_theta = 2.*np.pi*(1.-x)
    W_ell = np.array([scipy.special.eval_legendre(l-1, x) - 
                      scipy.special.eval_legendre(l+1, x) for l in ell])
    W_ell = 2.*np.pi/A_theta * W_ell/(2.*ell+1.)
    W_ell = W_ell**2 * (2.*ell+1.)/(4.*np.pi)
    return W_ell

################################################################################
# Simple galaxy count model (power law in mass)
################################################################################

def simple_galaxy_counts(z, log_M_min=12.0, slope=0.5):
    """
    Simple model for galaxy number density as function of redshift
    This is a placeholder - replace with your actual model
    """
    # Simple power law model n(z) ~ z^slope * exp(-z/z0)
    z0 = 1.5
    normalization = 10.0  # galaxies per steradian
    return normalization * (z**slope) * np.exp(-z/z0)

def simple_bias(z, b0=1.0, alpha=0.5):
    """
    Simple bias model b(z) = b0 * (1+z)^alpha
    """
    return b0 * (1. + z)**alpha

################################################################################
# Sample variance calculation
################################################################################

def sample_variance(camb_params, thetas, redshift_bins, 
                   nz_function=None, bias_function=None,
                   Cls_mask=None, area_deg2=20.0,num_points_z = 50):
    """
    Calculate sample variance for given areas and redshift bins
    
    Parameters:
    -----------
    camb_params : CAMB parameters object
    thetas : array of angles (in radians) corresponding to survey areas
    redshift_bins : edges of redshift bins
    nz_function : function n(z) returning number density (optional)
    bias_function : function b(z) returning bias (optional)
    Cls_mask : mask coupling matrix (optional)
    
    Returns:
    --------
    z_binsc : center of redshift bins
    sample_covariance : sample variance for each theta and redshift bin
    sample_cov_cov : variance of sample variance
    """
    
    # Copy CAMB params
    _camb_params = camb_params.copy()
    
    # Get redshift bin centers
    z_bins = np.array(redshift_bins)
    z_binsc = 0.5*(z_bins[1:] + z_bins[:-1])
    
    # Create n(z) and bias models
    if nz_function is None:
        nz_function = simple_galaxy_counts
    if bias_function is None:
        bias_function = simple_bias
    
    # Create interpolators
    window_z = np.linspace(0.9*np.amin(z_bins), 1.1*np.amax(z_bins), 100)
    model_nz = UnivariateSpline(window_z, nz_function(window_z), s=0., ext=1)
    model_bias = UnivariateSpline(window_z, bias_function(window_z), s=0., ext=1)
    
    # Prepare window functions
    num_windows = len(z_binsc)
    windows = []
    
    for ind in tqdm(range(num_windows)):
        delta_x = 0.003
        z = np.linspace(z_bins[ind]-20.*delta_x, z_bins[ind+1]+20.*delta_x, 
                       num_points_z)
        P = smooth_tophat_function(z, 0., 1., 
                                   0.5*(z_bins[ind]+z_bins[ind+1]), 
                                   z_bins[ind+1]-z_bins[ind], delta_x)
        temp = P * model_nz(z)
        amp = scipy.integrate.simpson(temp, z)
        windows.append(SplinedSourceWindow(bias=model_bias(z_binsc[ind]), 
                                          dlog10Ndm=0.0,  # no magnification bias
                                          z=z, 
                                          W=temp/amp, 
                                          source_type='counts'))
    
    # Get Cls from CAMB
    _camb_params.SourceWindows = windows
    temp_results = camb.get_results(_camb_params)
    ell, model_Cls = get_Cls(temp_results)
    max_ell = np.amax(ell)
    
    # Apply mask if provided
    if Cls_mask is not None:
        assert(0)
    
    # Compute sample covariance
    harmonic_filter = []
    for theta in thetas:
        harmonic_filter.append(harmonic_smooth_filter(theta, ell))
    harmonic_filter = np.array(harmonic_filter)
    
    # Multiply by Cls
    sample_covariance = np.tensordot(harmonic_filter, model_Cls, (1, 0))
    
    # Compute sample covariance of covariance
    temp = 2. * harmonic_filter**2 / (2.*ell + 1.)
    sample_cov_cov = np.dot(temp, 
                           np.diagonal(model_Cls, axis1=1, axis2=2)**2)
    
    return z_binsc, sample_covariance, sample_cov_cov

