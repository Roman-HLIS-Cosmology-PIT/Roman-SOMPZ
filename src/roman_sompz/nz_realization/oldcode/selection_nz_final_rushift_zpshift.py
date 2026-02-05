import numpy as np
import pickle
import pandas as pd
colors = ['k', 'teal', 'orange', 'powderblue','tomato'] 
#from astropy.coordinates import SkyCoord
#from astropy import units as u
import healpy as hp
from astropy.io import fits
import time
import h5py
from scipy.interpolate import interp1d
import multiprocessing as mp
import os 
import yaml
from functions_nzrealizations import*
import sys
sys.path.append('/global/cfs/cdirs/des/boyan/sompz_y6/sompz/')
from functions_sompz import *
from functions_WL import *
from datetime import datetime
today = datetime.today()
today = today.strftime('%B%d')
import gc
    
Nsamples = 10000

# Whether you want ZPU (photometric redshift zero point uncertainty) or RU (redshift sample uncertainty) inside analysis
ZPU = True
RU = True
#Bin conditionalization uncertainty is negelected in Y6 analysis 
#people have found in Y3 this is very small, about 0.003 for all bins while the combined uncertainty is always dominated by other factors, like photometric calibration uncertainty on 0th bin, and SN+SV for all 4 bins
#Y6 uncertainty should be similar as it share simliar redshift sample and deep sample. Thus, this uncertainty can be safely neglected
#RUBC = False

ZPU_label = "_ZPU" if ZPU else ""
RU_label = "_RU" if RU else ""


### dir
data_dir = '/global/cfs/cdirs/des/boyan/sompz_output/y6_data_10000Tile_final_unblind/noshear/weighted_pile3_oldbinning/'
outpath = '/global/cfs/cdirs/des/boyan/sompz_output/y6_data_10000Tile_final_unblind/nz_realizations/sub_files_oldbinning/'
outpath_ZPU = outpath + 'ZPU_assignment/'

bands = ['U','G','R','I','Z','J','H','K']

#cfg
cfgfile = '/global/cfs/cdirs/des/boyan/sompz_y6/y6_sompz_10000Tile_v6_final_noshear_unblind.cfg'

with open(cfgfile, 'r') as fp:
    cfg = yaml.safe_load(fp)
    
deep_balrog_file = cfg['deep_balrog_file']
redshift_deep_balrog_file = cfg['redshift_deep_balrog_file']
deep_cells_assignment_balrog_file = cfg['deep_cells_assignment_balrog_file']
wide_cells_assignment_balrog_file = cfg['wide_cells_assignment_balrog_file']

#Balrog  
balrog_data_noshift = build_balrog_df(deep_balrog_file, 
                      deep_cells_assignment_balrog_file, 
                      wide_cells_assignment_balrog_file)

balrog_data_noshift['overlap_weight'] = balrog_data_noshift['response'] * balrog_data_noshift['weights'] / balrog_data_noshift['injection_counts']


#Redshift
cosmos_noshift = build_spec_df(redshift_deep_balrog_file, balrog_data_noshift)


# Note the Balrog and redshift file must contain all detected Balrog Indices of NaNs:
# need to have 'weight_response_shear' R*w inside

### Load dictionary containing which wide cells belong to which tomographic bin
tomo_bins_wide_modal_even = None
with open( '/global/cfs/cdirs/des/boyan/sompz_output/y6_data_10000Tile_final/noshear/weighted_pile3/tomo_bins_wide_dict.pkl', 'rb') as file:
    tomo_bins_wide_modal_even = pickle.load(file)

print('finish data prep')  
######################################### Finished all data prep #############################################
############################################################################################################## 

ncells_deep = 64*64
ncells_wide = 32*32

## Tomographic bin selection
zpdfcols = ["Z{:.2f}".format(s).replace(".","_") for s in np.arange(0,4.00,0.01)]

LHC_samples = np.load("./LHC_samples.npy")
print("Starting loop of LHC shifts")

# prepare data for ZPU and RU
if ZPU:
    # Load deep galaxy ID, this is the galaxy ID for the flux-perturbed galaxy cell assignments
    deep_data_ID = np.load(outpath_ZPU + 'deep_data_ID.npy')
    
if RU:
    #SC is the median redshift bias of COSMOS galaxies wrt spectroscopic galaxies
    #SC[:,0] is mag_i, SC[:,1] is the bias(z)/(1+z) for that mag_i
    #The function below interpolates this bias for all mag_i values
    #SP is the median redshift bias of PAUS galaxies wrt spectroscopic galaxies
    median_bias_SC_file = np.load('/global/cfs/cdirs/des/boyan/sompz_data_y6/Redshift_Catalog/redshift_bias/median_bias_SC.npz')
    median_bias_SC = median_bias_SC_file['bias']
    std_bias_SC = median_bias_SC_file['std']
    ibin_edges_SC = median_bias_SC_file['ibin_edges']
    zbin_edges_SC = median_bias_SC_file['zbin_edges']
    
    
    median_bias_SP_file = np.load('/global/cfs/cdirs/des/boyan/sompz_data_y6/Redshift_Catalog/redshift_bias/median_bias_SP.npz')
    median_bias_SP = median_bias_SP_file['bias']
    std_bias_SP = median_bias_SP_file['std']
    ibin_edges_SP = median_bias_SP_file['ibin_edges']
    zbin_edges_SP = median_bias_SP_file['zbin_edges']
    
   


#Start running
for LHC_id, LHC_sample in enumerate(LHC_samples):
    
    LHC_id += 0
    
    print('id', LHC_id)
    filename = 'nz_samples_y6%s%s_LHC%d_1e6_stdRUmethod_unblind_oldbinning_Nov5.h5'%(RU_label, ZPU_label, LHC_id)
    if os.path.exists(outpath + filename):
        print("%d exists, continuing..."%LHC_id)
        continue
    
    T0 = time.time()
    
    # zero point uncertainty
    if ZPU:
        # If ZPU, load new cell deep assignment of galaxies based on purtubed deep fluxes
        
        # Load deep cell assignment with zpu
        cells_deep = np.load("%s/som_deep_64x64_assign_LHC%d.npz"%(outpath_ZPU, LHC_id))['cells']
    
        cells_deep_df = pd.DataFrame({'ID': deep_data_ID, 'cell_deep': cells_deep})
        
        # use this cell_deep column for balrog_data and cosmos
        balrog_data = balrog_data_noshift.copy().drop('cell_deep', axis=1)
        cosmos = cosmos_noshift.copy().drop('cell_deep', axis=1)

        balrog_data = balrog_data.merge(cells_deep_df, on='ID', how='left')
        cosmos = cosmos.merge(cells_deep_df, on='ID', how='left')
       
    else:
        
        balrog_data = balrog_data_noshift.copy()
        cosmos = cosmos_noshift.copy()

        


    ## Redshift sample uncertainty shift
    if RU:
        print('running RU')
        #New way magi-z bins of estimating RU
        
        #only perturb unique galaxy and match later
        unique_zpdf = cosmos[['ID','SOURCE','BDF_MAG_DERED_CALIB_I'] + zpdfcols].drop_duplicates()
        zpdf = unique_zpdf[zpdfcols].values
        origins = unique_zpdf['SOURCE'].values
        mags = unique_zpdf['BDF_MAG_DERED_CALIB_I'].values
        z_bins = np.arange(0, 4, 0.01)
        zmeans = np.dot(zpdf, z_bins) / zpdf.sum(axis=1)
        print('zmeans shape', np.shape(zmeans))
        
        #add 0s for zpdf 
        zero_columns_beginning = np.zeros((zpdf.shape[0], 100))
        zero_columns_end = np.zeros((zpdf.shape[0], 100))
        zpdf = np.hstack((zero_columns_beginning, zpdf, zero_columns_end))
        zbinsc_laigle_stacked = np.arange(-1.0,5.0,0.01)
       
        
        #Get the gaussian draws from pre-generated LHC
        sigma_C, sigma_P, sigma_bincond = LHC_sample[-4:-1]
        #print(sigma_C, sigma_P)
        
        new_zpdf = np.zeros_like(zpdf)

        #add a 0 column before and after zpdf, and do pileup on 0 and 3.99 after interpolation
        #This ensures in final pdf, the tail information is not mitigated
        for i in range(len(zpdf)):
            if origins[i] == 'COSMOS2020_FARMER_LEPHARE':

                mag_index =  np.digitize(mags[i], ibin_edges_SC) -1
                z_index = np.digitize(zmeans[i], zbin_edges_SC) -1
                mag_index = np.clip(mag_index, 0, len(ibin_edges_SC) - 2)
                z_index = np.clip(z_index, 0, len(zbin_edges_SC) - 2)
                
                # bias is cosmo - spec
                bias = median_bias_SC[mag_index, z_index]
                std = std_bias_SC[mag_index, z_index]
                shift =  bias + sigma_C * std
                
                f = interp1d(zbinsc_laigle_stacked, zpdf[i], bounds_error=False, fill_value=(0,0))
                #shift f(x) to the right is the same as shifting x axis to the left
                _npz = f(zbinsc_laigle_stacked+shift)
                _npz /= _npz.sum()
                new_zpdf[i] = _npz
                del f

    
                
                
            elif origins[i] == 'PAUSCOSMOS':

                mag_index =  np.digitize(mags[i], ibin_edges_SP) -1
                z_index = np.digitize(zmeans[i], zbin_edges_SP) -1
                mag_index = np.clip(mag_index, 0, len(ibin_edges_SP) - 2)
                z_index = np.clip(z_index, 0, len(zbin_edges_SP) - 2)
                
                bias = median_bias_SP[mag_index, z_index]
                std = std_bias_SP[mag_index, z_index]
                shift =  bias + sigma_P * std
                
                f = interp1d(zbinsc_laigle_stacked, zpdf[i], bounds_error=False, fill_value=(0,0))
                #shift f(x) to the right is the same as shifting x axis to the left
                _npz = f(zbinsc_laigle_stacked+shift)
                _npz /= _npz.sum()
                new_zpdf[i] = _npz
                del f
              
                
            else:
                new_zpdf[i] = zpdf[i]
               
            
            

        #pileup added columns
        new_zpdf[:, 100] += np.sum(new_zpdf[:, :100], axis=1) 
        new_zpdf[:, -101] += np.sum(new_zpdf[:, -100:], axis=1) 
        new_zpdf = new_zpdf[:, 100:-100]

        all_sums_to_1 = np.allclose(new_zpdf.sum(axis=1), 1)
        print(new_zpdf.sum(axis=1))
        print('Does all galaxy sum to 1: ',all_sums_to_1)
        
        unique_zpdf.loc[:,zpdfcols] = new_zpdf
        unique_zpdf.drop(columns = ['SOURCE','BDF_MAG_DERED_CALIB_I'])
        
        cosmos = cosmos.drop(columns=zpdfcols).merge(unique_zpdf, on='ID', how='left')
                
        print('finish RU')
        
        


        
    #cosmos is the redshift galaxies matched to deep and detected by Balrog. (not unique)


    ###########################################################################################
    ### All code below is for calaulating Sample Variance and Shot Noise
    ###########################################################################################

    ### Load shear-response weightd p(chat) and p(c|chat)
    # pcchat needs to be calculated in everey cell assignment
    pchat = np.load(data_dir+'pchat.npy')
    pcchat = np.load(data_dir+'pcchat.npy') # (4096,1024)
    pcchat = np.zeros_like(pcchat)
    np.add.at(pcchat, 
          (balrog_data.cell_deep.values,balrog_data.cell_wide_unsheared.values),
          balrog_data.overlap_weight.values)
    
    ###########################################################################################
    ### Define new redshift bins
    ###########################################################################################

    # Define the redshift binning. This is currently set by the sample variance.
    # This binning (dz = 0.05) ensures minimal correlation between bins, which is crucial since the Dir sampling is assuming each redshfit bin to be independent
    min_z   = 0.01
    max_z   = 4
    delta_z = 0.05
    zbins   = np.arange(min_z,max_z+delta_z,delta_z)
    zbinsc  = zbins[:-1]+(zbins[1]-zbins[0])/2.
    
    
    

    #####################################################################################
    ### Calculating bincond and shear-response weight factors (with interpolation)
    #####################################################################################
    

    #For all matrices below including redshift (Nzc_0 etc.), the function rebins cosmos (0,4.00,0.01) to the upper zbinsc using interpolation
    #Note for linear interpolation, it would be the same to 1) rebin for each galaxy and then combine and 2)combine to Nzc etc, and then rebin
    #I have tried to 1) but it consume too much memory and time, so I will use method 2) for Nzc and Rzc
    #Proof that 1) and 2) is the same:
    #f1(x) = f1(x0) + (f1(x1) - f1(x0))/(x1-x0) * (x-x0)
    #f2(x) = f2(x0) + (f2(x1) - f2(x0))/(x1-x0) * (x-x0)
    #f1(x) + f2(x) = (f1(x0) + f2(x0))  + ((f1(x1) + f2(x1)) - (f1(x0) + f2(x0))) / (x1-x0) * (x-x0)
    #The rebinning first makes zbinsc = [0,0.01,0.02,0.03,...,3.99] to [0.015,0.025,0.035,...,4.005] 
    #The combine 0.015 - 0.055 to 1 bin
    #Also note the way of normalization doesn't matter. Since only fraction matters
    
    ### Counts in the redshift sample (weighted by balrog, but not weighted by responseXlensing weights.)
    ### Including condition on tomographic bin.
    #Nzc, Nzc_i shape (80,4096)
    Nzc = return_Nzc(cosmos)  
    Nzc_0 = return_Nzc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
    Nzc_1 = return_Nzc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
    Nzc_2 = return_Nzc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
    Nzc_3 = return_Nzc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])

    ### Counts in the deep sample (weighted by balrog, but not weighted by responseXlensing weights.)
    ### Including condition on tomographic bin.
    Nc = return_Nc(balrog_data)
    Nc_0 = return_Nc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
    Nc_1 = return_Nc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
    Nc_2 = return_Nc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
    Nc_3 = return_Nc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])

    ### If after the bin condition there are no redshift counts in a deep cell, don't apply the bin condition in that deep cell.
    sel_0 = ((np.sum(Nzc, axis=0)>0) & (np.sum(Nzc_0, axis=0)==0))
    sel_1 = ((np.sum(Nzc, axis=0)>0) & (np.sum(Nzc_1, axis=0)==0))
    sel_2 = ((np.sum(Nzc, axis=0)>0) & (np.sum(Nzc_2, axis=0)==0))
    sel_3 = ((np.sum(Nzc, axis=0)>0) & (np.sum(Nzc_3, axis=0)==0))
    Nzc_0[:,sel_0] = Nzc[:,sel_0].copy()
    Nzc_1[:,sel_1] = Nzc[:,sel_1].copy()
    Nzc_2[:,sel_2] = Nzc[:,sel_2].copy()
    Nzc_3[:,sel_3] = Nzc[:,sel_3].copy()
    
    ### Average responseXlensing in each deep cell and redshift bin. The responseXlensing of each galaxy is weighted by its balrog probability.
    ### Including condition on tomographic bin.
    #Rzc, Rzc_i shape (80,4096)
    
    Rzc = return_Rzc(cosmos)
    Rzc_0 = return_Rzc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
    Rzc_1 = return_Rzc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
    Rzc_2 = return_Rzc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
    Rzc_3 = return_Rzc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])


    ### Average responseXlensing in each deep cell in the REDSHIFT sample. The responseXlensing of each galaxy is weighted by its balrog probability.
    ### Including condition on tomographic bin.
    Rc_redshift = return_Rc(cosmos)
    Rc_0_redshift = return_Rc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
    Rc_1_redshift = return_Rc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
    Rc_2_redshift = return_Rc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
    Rc_3_redshift = return_Rc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])

    ### Average responseXlensing in each deep cell in the DEEP sample. The responseXlensing of each galaxy is weighted by its balrog probability.
    ### Including condition on tomographic bin.
    Rc_deep = return_Rc(balrog_data)
    Rc_0_deep = return_Rc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
    Rc_1_deep = return_Rc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
    Rc_2_deep = return_Rc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
    Rc_3_deep = return_Rc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])

    ### We do not need the balrog and redshift samples. We can delete them.
    del cosmos
    del balrog_data
    
    

    #return [gzt_0, gzt_1, gzt_2, gzt_3], gzt_0 = pz_c_1/pz_c_0, bincond fraction
    fraction_Nzt = return_bincondition_fraction_Nzt_redshiftsample(np.array([Nzc, Nzc_0, Nzc_1, Nzc_2, Nzc_3]))
    #return [gt_0, gt_1, gt_2, gt_3], gt_0 = deep_sample_Nt[1]/deep_sample_Nt[0], bincond frac for deep
    fraction_Nt_D = return_bincondition_fraction_Nt_deepsample(np.array([Nc, Nc_0, Nc_1, Nc_2, Nc_3]))
    #The operation essentially performs elementwise multiplication, where the second array (fraction_Nt_D) is reshaped to add a middle dimension and then broadcasted so that each of the 80 elements in the middle dimension uses the same value.
    #(4,80,4096), each element is gzc(R,Bin)/gc(R,Bin) * gc(D,Bin) - total bin-cond weight for Redshift-Deep part
    bincond_combined = fraction_Nzt*fraction_Nt_D[:,None,:]
    



    
    
    redshift_sample_Rzt = np.array([Rzc, Rzc_0, Rzc_1, Rzc_2, Rzc_3])
    redshift_sample_Rt = np.array([Rc_redshift, Rc_0_redshift, Rc_1_redshift, Rc_2_redshift, Rc_3_redshift])
    deep_sample_Rt = np.array([Rc_deep, Rc_0_deep, Rc_1_deep, Rc_2_deep, Rc_3_deep])
    # return [Rzt_0_final, Rzt_1_final, Rzt_2_final, Rzt_3_final]
    # final Rzt = <Rzt>r * <Rt>D / <Rt>r
    Rt_combined = return_bincondition_weight_Rzt_combined(redshift_sample_Rzt, redshift_sample_Rt, deep_sample_Rt)


   
    #############################################################
    ### Compute superphenotypes
    #############################################################

    # We need superphenotypes because this method require each sample variance to be disjoint. phenotype t (or deep cell c) are highly correlated. So we are going to use fzc = f(c|zT) * f(z|T) * f(T) etc. to aviod correlation in the algorithm. also redshift is rebinnind to dz = 0.05 for same reason.
    
    
    # The small t(phenotype) here represnts a deep cell c
    nts = Nc.copy()
    nzt = Nzc.copy()  

    # Removing pheno-type t that don't have galaxies
    # Note all Dir-related properties are masked - we don't want a deep cell with 0 galaxy inside to be sampled with >0 galaxies
    maskt = (np.sum(nzt,axis=0)>0.)
    nts = nts[maskt]
    nzt = nzt[:,maskt]
    
    # Computing the mean redshift per phenotype - prepare combining similar redshift phenotype to super-phenotype T
    # here zmeant is the mean redshift per phenotype t * 100
    # for example a mean redshift = 1.017, here zmeant = 102
    # zmeant - (4096) with maskt
    zmeant = np.zeros(nzt.shape[1])
    for i in range(nzt.shape[1]):
        zmeant[i] = np.average(np.arange(len(zbinsc)),weights=nzt.T[i])
    zmeant = np.rint(zmeant)


    
    # Decide which phenotypes go to which superphenotype
    # Choose number of superphenotypes nT
    nT = 6
    #bins stores the zmeant list for each T
    bins = {str(b):[] for b in range(nT)}
    j = 0 
    sumbin = 0
    #nTs stores the number of galaxies in t wrt cell meanz 
    nTs = np.zeros(len(zbinsc))
    for i in range(len(zbinsc)):
        #increase mean redshift by 0.05(i increase 1 here) 
        #add all galaxies in t with this mean redshift into sumbin
        sumbin += np.sum(nzt[:,((zmeant==i))],axis=1).sum()
        nTs[i] = np.sum(nzt[:,((zmeant==i))],axis=1).sum()
        #if not enough galaxy in this T still add, otherwide T num j+=1. append i to T dict
        if (sumbin <= np.sum(nzt)/(nT-1))|(j==nT-1):
            bins[str(j)].append(i)
        else:
            j += 1
            bins[str(j)].append(i)
            sumbin = np.sum(nzt[:,((zmeant==i))],axis=1).sum()

    # tot number of redshift galaxies
    print('total galaxy num', np.sum(nzt), np.sum(nzt)/6) 
    # number of redshift galaxies in each phenotype
    print('nTs', np.sum(nTs[np.array(bins['0'])]), np.sum(nTs[np.array(bins['1'])]),np.sum(nTs[np.array(bins['2'])]),np.sum(nTs[np.array(bins['3'])]),np.sum(nTs[np.array(bins['4'])]),np.sum(nTs[np.array(bins['5'])]))
    # what is the t zmeans for each T
    print('bins', bins)

    # Make a map that converts z to T
    z2Tmap = np.zeros((len(zmeant))).astype(int)
    for i in range(nT):
        z2Tmap[np.isin(zmeant.astype(int),bins[str(i)])] = i
        

    # Compute p(T), p(z,T) for the superphenotypes
    nzTi = np.zeros((len(zbinsc),nT))
    nTi = np.zeros((nT))
    for i in range(nT):
        nzTi[:,i] = np.sum(make_nzT(nzt,1,False)[:,bins[str(i)]],axis=1)
        nTi[i] = np.sum(make_nT(nzt,nts,1)[bins[str(i)]])

    # Calculate correlation metirc to see the overlap of nz between superphenotypes
    # Should be ~1 because we made assumption that T are independent, so det(corr metric) ->1
    print ('Correlation metric = %.3f'%corr_metric(nzTi))

    
    #############################################################
    ### Load Sample Variance from theory. 
    #############################################################

    ### Load the sample variance theory ingredient. This estimates the ratio between Shot noise and sample variance.
    
    #load redshift sample variance
    sv_th = np.load('/pscratch/sd/a/alexalar/desy3data/cosmos_sample_variance.npy')[0]
    sv_th = np.diagonal(sv_th)[:]
    #make sure the maximum zbinsc is the same
    sv_th = sv_th[:len(zbinsc)]
    assert sv_th.shape[0]==len(zbinsc)
    
    #load deep sample variance
    sv_th_new = np.load('/pscratch/sd/a/alexalar/desy3data/marco_sv_v2/sample_variance.npy')
    #combine the four deep field sample variance and cut at zbinsc
    #taking inverse matrix of 4 deep fields, sum them up, and inverse again
    #This will result a smaller variance than any field alone
    sv_th_new_final = np.linalg.pinv(np.sum(np.array([np.linalg.pinv(x) for x in sv_th_new]),axis=0))
    sv_th_new_final_diag = np.diagonal(sv_th_new_final)
    sv_th_new_final_diag = sv_th_new_final_diag[:len(zbinsc)]
    
    # var = 1 + N* sv_th
    varn_th = 1 + np.sum(nzt,axis=1)*sv_th
    varn_th_deep_v2 = 1 + np.sum(nzt/np.sum(nzt,axis=0) * nts,axis=1)*sv_th_new_final_diag
    


    
    #######################################################################################
    ### Compute matrices and lambdas in prep for Dirichlet sampling
    ### Redshift: N_Tcz_Rsampe, N_Tz_Rsample (lambda_T), N_T_Rsample (lambda_mean_R)
    ### Deep: N_c_Dsample , N_T_Dsample(lambda_mean)
    #######################################################################################
    
    nt = sum(maskt)
    nz=len(zbinsc)
    N_Tcz_Rsample = np.zeros((nT,nt,nz))  #(nT,maskt,80) #maskt masks t that don't have galaxies
    for i in range(nT):
        sel = z2Tmap==i
        N_Tcz_Rsample[i, sel] = nzt.T[sel]

    N_Tc_Dsample = np.zeros((nT,nt))    #(nT,maskt) #maskt masks t that don't have galaxies
    for i in range(nT):
        sel = z2Tmap==i
        N_Tc_Dsample[i, sel] = nts[sel]

    
    
    N_T_Rsample = np.sum(N_Tcz_Rsample, axis=(1,2))
    N_z_Rsample = np.sum(N_Tcz_Rsample, axis=(0,1))
    N_Tz_Rsample = np.sum(N_Tcz_Rsample, axis=(1))
    N_cz_Rsample = np.sum(N_Tcz_Rsample, axis=(0))

    N_T_Dsample = np.sum(N_Tc_Dsample, axis=(1))
    N_c_Dsample = np.sum(N_Tc_Dsample, axis=(0))

    lambda_z_step1 = varn_th_deep_v2.copy()
    lambda_z_step2 = varn_th.copy()
    lambda_mean = np.sum(lambda_z_step1*N_z_Rsample/N_z_Rsample.sum())
    lambda_mean_R = np.sum(lambda_z_step2*N_z_Rsample/N_z_Rsample.sum())
    lambda_T = np.array([np.sum(lambda_z_step2 * x/x.sum()) for x in N_Tz_Rsample])

    # onecell -  cells c that have only one redshift galaxy incide 
    # N_cz_Rsample_onecell - normalized N_cz_Rsample for onecell
    # why are we doing this?
    onecell = np.sum(N_cz_Rsample>0,axis=1) == 1
    N_cz_Rsample_onecell = (N_cz_Rsample/np.sum(N_cz_Rsample,axis=1)[:,None])[onecell]
    
    ###########################################################################################
    ### Compute f(c, chat|bin) / (f(c|bin)*f(chat|bin)) - Balrog portion for nz realization 
    ### Output fcchat_i is in the order fchatc
    ###########################################################################################
    

    #f(cchat|bin) / (f(c|bin)*f(chat|bin))
    #This is the Balrog portion of nz realization
    
    # pcchat.sum() returns scalar value, should sum up to 1
    fcchat = pcchat.T/pcchat.sum()      #(1024,4096)

    fcchat_0 = fcchat[tomo_bins_wide_modal_even[0]]
    fcchat_1 = fcchat[tomo_bins_wide_modal_even[1]]
    fcchat_2 = fcchat[tomo_bins_wide_modal_even[2]]
    fcchat_3 = fcchat[tomo_bins_wide_modal_even[3]]
    # normfactor is fc and fchat of balrog
    normfactor_0 = np.multiply.outer(np.sum(fcchat_0,axis=1), np.sum(fcchat_0,axis=0))
    normfactor_1 = np.multiply.outer(np.sum(fcchat_1,axis=1), np.sum(fcchat_1,axis=0))
    normfactor_2 = np.multiply.outer(np.sum(fcchat_2,axis=1), np.sum(fcchat_2,axis=0))
    normfactor_3 = np.multiply.outer(np.sum(fcchat_3,axis=1), np.sum(fcchat_3,axis=0))
    
    fcchat_0 = np.divide(fcchat_0 , normfactor_0, np.zeros(np.shape(fcchat_0)), where = normfactor_0!=0)
    fcchat_1 = np.divide(fcchat_1 , normfactor_1, np.zeros(np.shape(fcchat_1)), where = normfactor_1!=0)
    fcchat_2 = np.divide(fcchat_2 , normfactor_2, np.zeros(np.shape(fcchat_2)), where = normfactor_2!=0)
    fcchat_3 = np.divide(fcchat_3 , normfactor_3, np.zeros(np.shape(fcchat_3)), where = normfactor_3!=0)

    fcchat_0[~np.isfinite(fcchat_0)] = 0
    fcchat_1[~np.isfinite(fcchat_1)] = 0
    fcchat_2[~np.isfinite(fcchat_2)] = 0
    fcchat_3[~np.isfinite(fcchat_3)] = 0

    ###########################################################################################
    ### Compute f(chat|bin) - Wide portion for nz realization 
    ###########################################################################################

    fchat_0 = pchat[tomo_bins_wide_modal_even[0]]
    fchat_1 = pchat[tomo_bins_wide_modal_even[1]]
    fchat_2 = pchat[tomo_bins_wide_modal_even[2]]
    fchat_3 = pchat[tomo_bins_wide_modal_even[3]]
    
    ###########################################################################################
    ### Compute Fcchat_i - Balrog * Wide portion for nz realization
    ### Note Fcchat_i is in shape Fchatc_i
    ### This is shear-response weighted and bin conditionalized
    ###########################################################################################

    Fcchat_0 = fcchat_0*fchat_0[:,None]
    Fcchat_1 = fcchat_1*fchat_1[:,None]
    Fcchat_2 = fcchat_2*fchat_2[:,None]
    Fcchat_3 = fcchat_3*fchat_3[:,None]
    

    # Save info to h5 file
    try:
        print(save_h5)
        store = pd.HDFStore(save_h5)
        store['nzt'] = pd.DataFrame(nzt)
        store['nzTi'] = pd.DataFrame(nzTi)
        store['nTi'] = pd.Series(nTi)
        store['nts'] = pd.Series(nts)
        store['bincond_combined_0'] = pd.DataFrame(bincond_combined[:,:,maskt][0])
        store['bincond_combined_1'] = pd.DataFrame(bincond_combined[:,:,maskt][1])
        store['bincond_combined_2'] = pd.DataFrame(bincond_combined[:,:,maskt][2])
        store['bincond_combined_3'] = pd.DataFrame(bincond_combined[:,:,maskt][3])
        store['R_combined_0'] = pd.DataFrame(Rt_combined[:,:,maskt][0])
        store['R_combined_1'] = pd.DataFrame(Rt_combined[:,:,maskt][1])
        store['R_combined_2'] = pd.DataFrame(Rt_combined[:,:,maskt][2])
        store['R_combined_3'] = pd.DataFrame(Rt_combined[:,:,maskt][3])
        store['sv_th'] = pd.Series(sv_th)
        store['sv_th_deep'] = pd.Series(sv_th_new_final_diag)
        store['varn_th'] = pd.Series(varn_th)
        store['varn_th_deep'] = pd.Series(varn_th_deep_v2)
        store['fcchat_0'] = pd.DataFrame(fcchat_0[:,maskt])
        store['fcchat_1'] = pd.DataFrame(fcchat_1[:,maskt])
        store['fcchat_2'] = pd.DataFrame(fcchat_2[:,maskt])
        store['fcchat_3'] = pd.DataFrame(fcchat_3[:,maskt])
        store['fchat_0'] = pd.Series(fchat_0)
        store['fchat_1'] = pd.Series(fchat_1)
        store['fchat_2'] = pd.Series(fchat_2)
        store['fchat_3'] = pd.Series(fchat_3)
        store['z2Tmap'] = pd.Series(z2Tmap)
        store['maskt'] = pd.Series(maskt)
        store.close()
    except:
        pass


   
    ################################################################################################
    ### Until now, we have finished preparing all data that don't need sampling
    ### 1) The Balrog and Wide probabilities: Fcchat_0, Fcchat_1, Fcchat_2, Fcchat_3 
    ### 2) NzcT etc. matrices for Dirchlet sampling
    ### 3) bin-cond fraction: bincond_combined
    ### 4) shear response weight fracton: Rt_combined
    ### We now go inside the loop and use Dirichlet sampling to sample Nsample nzs
    ################################################################################################

    # epsilon of dirichlet equation, to ensure that dirchlet don't give error on Dir(0)
    alpha = 1e-300
    
    # The functions for producing Nsamples nz. These functions use previous (global) data 
    # and can't be in another file
    ########################################################################################
    def draw_3sdir_onlyR():
    
        #3sDir Redshift sample fzc(Redshift) = sum_T: fc|zT * fz|T * fT
    
        ### step1
        f_T = np.random.dirichlet(N_T_Rsample/lambda_mean_R+alpha)

        ### step2
        f_z_T = np.array([np.random.dirichlet(x/lambda_T[i]+alpha) for i,x in enumerate(N_Tz_Rsample)])

        ### step3
        f_cz_Rsample = np.random.dirichlet(N_cz_Rsample.reshape(np.prod(N_cz_Rsample.shape))+alpha).reshape(N_cz_Rsample.shape)
        f_cz = np.zeros((nt,nz))
        for k in range(N_Tcz_Rsample.shape[0]):  # For each T
            #z2Tmap - (4096) with maskt. Each value is what T c belongs to 
            #sel is the c indices that belong to kth T
            sel = z2Tmap==k                   
            dummy = f_cz_Rsample[sel] 
            dummy = np.divide(dummy, np.sum(dummy,axis=0), np.zeros(np.shape(dummy)), where = np.sum(dummy,axis=0)!=0)
            dummy[np.isnan(dummy)] = 0
            f_cz[sel] += np.einsum('cz,z->cz', dummy, f_z_T[k])* f_T[k]

        return f_cz


    def draw_3sdir_newmethod():
        
        # All Ns (both Redshift and deep) are masked from phenotype t that don't have redshift galaxies
        
        #3s Dir deep sample fc(Deep) = sum_T: fc|T * fT
        f_T = np.random.dirichlet(N_T_Dsample/lambda_mean+alpha)

        f_cT = np.zeros(nt)
        for k in range(nT):
            #z2Tmap - (4096) with maskt. Each value is what T c belongs to 
            #sel is the c indices that belong to kth T
            sel = z2Tmap==k
            #f_cT - (4096) with maskt, is actually fc(Deep)
            f_cT[sel] = np.random.dirichlet(N_Tc_Dsample[k,sel]+alpha) * f_T[k]

        #3sDir Redshift sample fzc(Redshift) = sum_T: fc|zT * fz|T * fT
        f_cz = draw_3sdir_onlyR()

        # fzc = fzc(Redshift)/sum_z:fzc(Redshift) * fc(Deep)
        sums = np.sum(f_cz, axis=1)[:, None]  
        f_z_c = np.divide(f_cz , sums, np.zeros(np.shape(f_cz)), where = sums!=0)
        
        
        # if onecell (only one redshift galaxy in cell c), then put pre-calculated values inside without Dirichlet sampling and phenotype T
        # This look like a 'hard code' to put inside correct value. Maybe there is something going one in this case: like superphenotypy T won't be a good representation for a one galaxy cell phenotype
        f_z_c[onecell] = N_cz_Rsample_onecell   

        ### compute f_{zc}
        f_cz = f_z_c * f_cT[:,None]
        return f_cz



    def aux_fun(i):
        np.random.seed()
        #want to generate Nsamples nz realization
        nz_samples_newmethod = np.zeros((Nsamples,4, len(zbinsc)))

        for i_sample in range(Nsamples):
            # f_zt = fzc(redshift)/fc(Redshfit) * fc(Deep)
            # fc(Redshfit) = sum_z (fzc(redshift))
            f_zt = draw_3sdir_newmethod()
            # generate nz: multiply by bincond fraction, weight, and Balrog + Wide portions
            nz_samples_newmethod[i_sample] = return_nzsamples_fromfzt(f_zt, i_sample)
            if i_sample%(1e3) == 0 :
                print(i_sample)

        return nz_samples_newmethod  
    

    def return_nzsamples_fromfzt(fzt_dummy, i):
        fzt = np.zeros((4096,len(zbinsc))).T   #(80,4096)
        #fill in phenotype with redshfit galaxy inside with fzt input, otherwise 0
        fzt[:,maskt] = fzt_dummy.T

        # fzt = fzc(Redshift)/fc(Redshift) * fc(Deep) - (4,80,4096)
        # Add the bin condition and the average response and shear weights.
        # bincond_combined - (4,80,4096), each element is gzc(R,Bin)/gc(R,Bin) * gc(D,Bin) - total bin-cond weight for Redshift-Deep part
        # Rt_combined - (4,80,4096), Rzt = <Rzt>r * <Rt>D / <Rt>r
        fzt_0 = fzt * bincond_combined[0] * Rt_combined[0]
        fzt_1 = fzt * bincond_combined[1] * Rt_combined[1]
        fzt_2 = fzt * bincond_combined[2] * Rt_combined[2]
        fzt_3 = fzt * bincond_combined[3] * Rt_combined[3]

        fzt_0 /= np.sum(fzt_0)
        fzt_1 /= np.sum(fzt_1)
        fzt_2 /= np.sum(fzt_2)
        fzt_3 /= np.sum(fzt_3)

        fzt_0[~np.isfinite(fzt_0)] = 0
        fzt_1[~np.isfinite(fzt_1)] = 0
        fzt_2[~np.isfinite(fzt_2)] = 0
        fzt_3[~np.isfinite(fzt_3)] = 0

        # add Balrog and Wide fraction and do summation over c and chat
        # Note Fcchat_i is in shape Fchatc_i (1024,4096)
        # For simplicity of calculation, all balrog and wide fraction are transposed to have 4096 in the last row
        nz_0 = np.einsum('zt,dt->z', fzt_0, Fcchat_0)
        nz_1 = np.einsum('zt,dt->z', fzt_1, Fcchat_1)
        nz_2 = np.einsum('zt,dt->z', fzt_2, Fcchat_2)
        nz_3 = np.einsum('zt,dt->z', fzt_3, Fcchat_3)

        nz_0 /= nz_0.sum()
        nz_1 /= nz_1.sum()
        nz_2 /= nz_2.sum()
        nz_3 /= nz_3.sum()
        
        if nz_0.sum() ==0:
            print('nz_0',nz_0)
            print(i)
            print(bincond_combined[0])
            print(Rt_combined[0])
            print(fzt)

        nz_samples = np.array([nz_0, nz_1, nz_2, nz_3])
        return nz_samples

    ########################################################################################
    # for Nsamples = 10000, using pool uses ~ same time as direct run
    '''
    indices = np.array_split(np.arange(Nsamples), 64)
    
    p = mp.Pool(64)
    nz_samples_newmethod = np.concatenate(p.map(aux_fun, indices), axis=0)
    p.terminate()
    '''
    p = mp.Pool(100)
    nz_samples_newmethod = np.concatenate(p.map(aux_fun, range(100)), axis=0)
    p.terminate()
    


    ##########################################################################################################
    ### Input: all the pre-prepared data and probabilities
    ### 1) Draw 3sDir sampling for Redshift and Deep using precomputed NzcT etc. matrics
    ### 2) Multiply pre-computed bincond and shear-response weight to Redshift and Deep probabilities
    ### 3) combine with pre-computed Balrog and Wide probabilities -> nz
    ### Output: nz (Nsamples, 4, 80)
    ##########################################################################################################
    #nz_samples_newmethod = aux_fun()

    # save the nz realiztions to h5 file
    print("Saving " + filename)
    with h5py.File(outpath + filename, 'w') as hf:
        hf.create_dataset("zbins",  data=zbins)
        hf.create_dataset("zbinsc",  data=zbinsc)
        hf.create_dataset("bin0",  data=nz_samples_newmethod[:,0])
        hf.create_dataset("bin1",  data=nz_samples_newmethod[:,1])
        hf.create_dataset("bin2",  data=nz_samples_newmethod[:,2])
        hf.create_dataset("bin3",  data=nz_samples_newmethod[:,3])


    T1 = time.time()
    print("Total time for %d: %.2f"%(LHC_id, T1-T0))
    
    # only SV + SN 
    if ((RU==False) & (ZPU==False)):
        break
        
