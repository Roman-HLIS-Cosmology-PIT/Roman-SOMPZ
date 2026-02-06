# Diogo Souza - Jul 14 2025
import sys, os

cocoa_path = '/gpfs/scratch/pit-roman-hlis/Diogo/cocoa/Cocoa' # Where did you install CoCoA? Change to your path!
if os.getcwd().strip().endswith("Cocoa") and os.getcwd()!=cocoa_path:
    raise ValueError(f"RULE OF THUMB: You're at {os.getcwd()}. Run this code from ./Cocoa") #TODO: change the logic

##################################################################
##################################################################
##################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
import h5py
from sklearn.decomposition import PCA
from getdist import MCSamples, plots
from scipy.stats import norm
from numpy import linalg as la
from cobaya.yaml import yaml_load
from yaml import safe_load, dump, safe_dump
from cobaya.model import get_model
import camb
from camb import model
import argparse
import shutil
sys.path.insert(0, os.environ['ROOTDIR']+'/external_modules/code/CAMB/build/lib.linux-x86_64-'+os.environ['PYTHON_VERSION'])

plt.rcParams['figure.figsize'] = (3.5, 2.5)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.size'] = 15

colors1 = [
    "#1b5f6f",  # dark teal
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#882255",  # wine
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999",  # grey
    "#117733",  # dark green
]

##################################################################
##################################################################
##################################################################

camb_path = os.path.dirname(camb.__file__)
print('Using CAMB %s installed at %s'%(camb.__version__,camb_path))
possible_scenarios = ['sc1bd4','sc1bd5','sc1bd6','sc1bd7',
                      'sc2bd4','sc2bd5','sc2bd6','sc2bd7',
                      'sc3bd4','sc3bd5','sc3bd7'
                      ]
rename_scenarios = {'sc1bd4':'DRM-D1','sc1bd5':'DRM-D2','sc1bd6':'DRM-D3','sc1bd7':'DRM-D4',
                    'sc2bd4':'W-D1'  ,'sc2bd5':'W-D2'  ,'sc2bd6':'W-D3'  ,'sc2bd7':'W-D4'  ,
                    'sc3bd4':'M-D1'  ,'sc3bd5':'M-D2'  ,'sc3bd7':'M-D4'
                    }

parser = argparse.ArgumentParser()
parser.add_argument('--scenario_model'   , type=str, required=False, default='sc1bd4', help='Roman scenario')
parser.add_argument('--scenario_fiducial', type=str, required=False, default='sc1bd4', help='Roman scenario as fiducial for projection onto PCs')
parser.add_argument('--realization'      , type=int, required=False, default=None    , help='Realization tag: any number in [0,...,999999]. If None, use the mean n(z)')
parser.add_argument('--npcs_used'        , type=int, required=False, default=414     , help='Number of PCs')
parser.add_argument('--want_weights'     , type=int, required=False, default=1       , help='Set weights to PCA: If 1 use Fisher matrix (default). If 0 use Identity matrix')
parser.add_argument('--want_projection'  , type=str, required=False, default=1       , help='No projection use 0 (default). Projection of n(z)_mean from scenario `scenario` onto PCs use 1. Projection of n(z)_realization onto PCs (use 1 and the `sim` tag MUST be != None)')
parser.add_argument('--interactive_mode' , type=str, required=False, default=0       , help='If 0 (False) will change the yaml at memory lvel only. If 1 (True) will dump to the yaml file.')

args = parser.parse_args()

sc_mod           = args.scenario_model
sc_fid           = args.scenario_fiducial
sim              = args.realization
npcs_used        = args.npcs_used
want_weights     = args.want_weights
want_projection  = args.want_projection
interactive_mode = args.interactive_mode # TODO: FEATURE TO BE IMPLEMENTED

if sc_mod not in possible_scenarios:
    raise ValueError(f"Invalid Roman scenario model: '{sc_mod}'! Should be one of: {possible_scenarios}")
elif sc_mod in possible_scenarios:
    print(f"\033[32mUSING ROMAN SCENARIO MODEL   : {sc_mod}\033[0m")

if sc_fid not in possible_scenarios:
    raise ValueError(f"Invalid Roman scenario fiducial: '{sc_fid}'! Should be one of: {possible_scenarios}")
elif sc_fid in possible_scenarios:
    print(f"\033[32mUSING ROMAN SCENARIO FIDUCIAL: {sc_fid}\033[0m")

if sim==None:
    print(f"\033[32mUSING THE MEAN n(z) OF THE FIDUCIAL {sc_fid}\033[0m")
elif sim!=None:
    print(f"\033[32mUSING THE REALIZATION n(z) INDEX = {sim} OF THE FIDUCIAL {sc_fid}\033[0m")
print(f"\033[32mUSING {npcs_used} PCS\033[0m")

##################################################################
##################################################################
##################################################################
import contextlib
@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
##################################################################
##################################################################
##################################################################

class SURVEY:
    def __init__(self,survey,use_r=None):
        self.survey = survey
        self.use_r = use_r

    # Global parameters
    def nz_params(self):
        if self.survey == 'des':
            Nz = 299 #len(z)
            Nt = 4
        elif self.survey == 'roman':   
            Nz = 46 #len(z)
            Nt = 9
        return Nz,Nt

    # Auxiliary functions
    def get_nzs(self):
        Nz,Nt = self.nz_params()
        if self.survey == 'des':
            nz = f'{cocoa_path}/cocoa_photoz/roman_nz_realizations/Fisher_matrix/Tz_realizations_WZ_bq_pile3_0d01.npy' # shared by Boyan Yin
            nz = np.load(nz)             ## shape = (10095, 4, 300) = (Ns, Nt, Nz) = (# of simulations, # of tomo bins, # of redshift)
            nz = nz[:,:,1:]              ## shape = (10095, 4, 299) | (first :) => all 10095 simulations, (second :) => all 4 tomo bins, (third ,1:) => all redshift bins except the first where nz is 0.0.
            Ns = np.shape(nz)[0]         ## Number of simulations: 10095
            nz = nz.reshape(Ns, Nt * Nz) ## shape = (10095, 1196) = (Ns, Nt*Nz)
            min_z   = 0.01
            max_z   = 3
            delta_z = (max_z - min_z) / Nz
            zbins   = np.arange(min_z,max_z+delta_z,delta_z)
            z  = zbins[:-1]+(zbins[1]-zbins[0])/2.
            return z, nz

        elif self.survey == 'roman':
            nz = h5py.File(f'{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_mod}/nz_samples_LHC0_pointZ_1e6_Roman_{sc_mod}.h5','r') # shared by Boyan Yin
            z = np.array(nz['zbinsc'])
            return z, nz

    def roman_nbar(self):    
        nbar = f'{cocoa_path}/cocoa_photoz/n_nbar_ndiff_covndiff/nbar_sc1b_d4.nz'
        return nbar

    def cov_of_ndiff(self):
        Nz,Nt = self.nz_params()
        z,nz = self.get_nzs()

        if self.survey == 'roman':
            # Never use r=10^6 on login node.
            if self.use_r is not None:
                #np.random.seed(42)
                #sims_indexes = np.random.randint(1, 10**6 + 1, size=self.use_r)
                # tot = len(sims_indexes)
                sims_indexes = self.use_r

                rows = []
                print(f"Total # of realizations r = {self.use_r}\nNormalizing nz for r:")
                for jj, j in enumerate(sims_indexes):
                    row = []
                    for i in range(Nt):
                        norm = np.trapz(y=nz[f'bin{i}'][j], x=z)
                        normalized = nz[f'bin{i}'][j] / norm
                        row.append(normalized)
                    #row shape after appending all bins: (Nt,Nz)
                    print(jj, end=' ')
                    rows.append(np.hstack(row))
                #rows shape after appending all realizations: (tot,Nt*Nz)
                nz = np.vstack(rows) # (tot,Nt*Nz)
                nbar = np.mean(nz,axis=0)
                ndiff = nz - nbar
                Cn = np.einsum('ij,ik->jk',ndiff,ndiff) / (ndiff.shape[0]-1) # equivalent to Cn = ndiff.T @ ndiff / sample size - 1
            else:
                # Use Cn from all 10^6  
                path_to_Cn = f'{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_mod}/Cn_{sc_mod}.txt'
                print(f"\033[32mUSING COVARIANCE OF REALIZATIONS Cn @: {path_to_Cn}\033[0m")
                Cn = np.genfromtxt(path_to_Cn) # TODO: INTEGRAR COM OUTPUTS DE PIP_1_NNORM_NBAR_NDIFF_CN.PY
                ndiff = None # Because we can't load (414,1M) data here
        elif self.survey == 'des':
            ### nzs for DES already are normalized
            ### set parameter below to True if you want to check
            check_des_nz_normalization = False
            if check_des_nz_normalization:
                rows = []
                for j in range(10095):
                    row = []
                    for i in range(Nt):
                        start, end = Nz*i, Nz*(i+1)
                        norm = np.trapz(y=nz[j,start:end], x=z)
                        normalized = nz[j,start:end] / norm
                        row.append(normalized)
                    rows.append(np.hstack(row))
                nz = np.vstack(rows)

            nbar = np.mean(nz,axis=0)
            ndiff = nz - nbar
            Cn = np.einsum('ij,ik->jk',ndiff,ndiff) / (ndiff.shape[0]-1) # equivalent to Cn = ndiff.T @ ndiff / sample size - 1
        self.Cn = Cn
        self.ndiff = ndiff    
        return self.Cn, self.ndiff
    
    # Unweighted PCA    
    def pcs(self,method='eig'):
        Nz,Nt = self.nz_params()
        Nd = Nz*Nt
        if method == 'eig':
            eig_vals, eig_vecs = np.linalg.eig(self.Cn)
            eigvals, eigvecs = np.linalg.eig(self.Cn)
        elif method == 'eigh':
            eigvals, eigvecs = np.linalg.eigh(self.Cn)
        elif method == 'svd':    
            U, S, VT = np.linalg.svd(self.Cn)
            eigvecs, eigvals = U,S
        elif method == 'sklearn':
            pca = PCA()
            pca.fit(self.Cn)
            eigvecs, eigvals = pca.components_, pca.singular_values_
            explained_variance = pca.explained_variance_
            explained_variance_ratio = pca.explained_variance_ratio_
        ######
        if method != 'sklearn':
            eigvals = np.real(eigvals)
            eigvecs = np.real(eigvecs)
            idx     = np.argsort(eigvals)[::-1] # 1. Get indices to sort eigenvalues in descending order
            eigvals = eigvals[idx]              # 2. Sort eigenvalues (largest to smallest)
            eigvals = np.maximum(0,eigvals)     # 3. Clamp any small negative eigenvalues to 0 (often due to numerical errors)
            eigvecs = eigvecs[:, idx]           # 4. Reorder columns of eigenvectors to match sorted eigenvalues
            return eigvecs, eigvals
        elif method == 'sklearn':
            return eigvecs.T, eigvals, explained_variance,explained_variance_ratio
    
    def forward_pca(self, params_values,nbar_path,pcs_path,npcs_nz):
        nbar = np.genfromtxt(nbar_path)
        U = np.genfromtxt(pcs_path)[:,:npcs_nz]
        s = nbar[:,1:].shape
        z = nbar[:,0]
        # Model: n(z) = <n>(z) + α_1*PC_1(z) + α_2*PC_2(z) + ... + α_n*PC_n(z)
        if npcs_nz > 0:
            alphas = np.array([params_values.get("roman_alpha_"+str(i+1)) for i in range(npcs_nz)])
            correction = (alphas * U).sum(axis=1)
    
            nz_model = nbar[:,1:].T.flatten() + correction
            nz_model = nz_model.reshape(s[::-1]).T
            nz_model = np.column_stack((z,nz_model))
            return nz_model
        else:
            return nbar

    # Credit: Gary M. Bernstein
    def nearestPD(self,A):
        """Find the nearest positive-definite matrix to input
        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].
        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        if self.isPD(A3):
            return A3
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self.isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
        return A3

    def isPD(self,B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False              
    
    def getModes(self,Fisher, Cn, chisq_threshold=0):
        '''Calculate the compression matrix from n to u and the
        modes U that multiply each U.  
        Parameters:
        `D`:      The matrix J^T C_c^{-1} J, where C_c is the covariance
                matrix of the observables c, and J is the Jacobian dc/dn,
                shape=(N,N)
        `Cn`:     Covariance matrix of the n(z) parameter vector n, shape=(N,N)
        `chisq_threshold`:  Your desired upper bound for the mean error in chisq
                resulting from the compression.
        Returns:
        `X`:      The compression matrix, u = X @ n, shape=(nModes,N).  The
                covariance matrix of the u's will be the identity.
        `U`:      The decoding matrix, n' = U @ u, shape=(N,nModes)
        `dchisq`: The mean chisq shift generated by each mode, shape=(nModes)
        `resid`:  Mean amount of chisq shift caused by compression.

        You can safely use fewer than `nModes` of the returned X and U if you 
        think the discarded `dchisq` values are small enough.'''

        thr = 1e-25     # Dynamic range threshold for eigenvalues

        # Symmetrize and sqrt D
        D=Fisher
        D = 0.5*(D + D.T)
        Dval, Dvec = np.linalg.eigh(D)
        # Any negatives are numerical problems
        Dval = np.maximum(0., Dval)

        # And Cn
        Cnval, Cnvec = np.linalg.eigh(Cn)
        Cnval = np.maximum(0., Cnval)

        # Build the keystone matrix and its SVD
        M = np.einsum('i,ji,jk,k->ik',np.sqrt(Cnval), Cnvec, Dvec, np.sqrt(Dval))
        Um,s,Vmt = np.linalg.svd(M)

        # Sort SV's and throw away unwanted ones
        order = np.argsort(s*s)  # increasing order
        kill = np.count_nonzero(np.cumsum(s[order]**2) < chisq_threshold)
                                            
        resid = np.sum((s[order[:kill]]**2))

        keep = np.flip(order[kill:])  # Decreasing influence order
        Um = Um[:,keep]
        Vm = Vmt.T[:,keep]
        s = s[keep]

        # Build the encoder
        tmp = np.where(Cnval>thr*np.max(Cnval), 1/np.sqrt(Cnval), 0.)
        X = np.einsum('ij,i,ki->jk',Um, tmp, Cnvec)  # X is (M,N)
        # And the decoder
        tmp = np.where(Dval>thr*np.max(Dval), 1/np.sqrt(Dval), 0.)
        U = np.einsum('ji,i,ik,k->jk',Dvec, tmp, Vm,s) # Vm s
        
        return X, U, s*s, resid, Dvec, Dval, Cnvec, Cnval, Um, Vm, M
    
    ###############################
    ############ CoCoA ############
    ###############################
    
##################################################################
##################################################################
##################################################################
roman_obj = SURVEY('roman',use_r=None)
Nz,Nt     = roman_obj.nz_params()
# Cn, ndiff = roman_obj.cov_of_ndiff() # must be called before pcs

# z,nz      = roman_obj.get_nzs()
# nbar_file = roman_obj.roman_nbar()
# eig_vecs, eig_vals = roman_obj.pcs('eig')
# eih_vecs, eih_vals = roman_obj.pcs('eigh')
# svd_vecs, svd_vals = roman_obj.pcs('svd')
# skl_vecs, skl_vals,\
# explained_variance, explained_variance_ratio = roman_obj.pcs('sklearn')

def fisher(want_weights=1,Cn_shape=0):
    if want_weights:
        step_size   = '1.000000e-03'
        fisher_path = f"{cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/fisher_{step_size}.txt"
        print(f"\033[32mUSING WEIGHTS. FISHER MATRIX        @: {fisher_path}\033[0m")
        # inv_cov       = np.genfromtxt(f'{stencil_5pt}/inv_cov.txt')
        # deriv_ξpm     = np.genfromtxt(f'{cocoa_path}/jacobian_outs/{relative_path}/deriv_ξpm_{step_size}.txt')
        # fisher_ξpm    = deriv_ξpm@inv_cov[:deriv_ξpm.shape[1],:deriv_ξpm.shape[1]]@deriv_ξpm.T
        try:
            fisher_ξpm = np.genfromtxt(fisher_path)
        except FileNotFoundError:
            print(f"\033[31mTHE FISHER MATRIX @: {fisher_path} WAS NOT FOUND\033[0m")

    if not want_weights:
        print("NOT USING WEIGHTS")
        fisher_ξpm = np.identity(Cn_shape)   

    if not roman_obj.isPD(fisher_ξpm):
        print('fixing cov')
        F=roman_obj.nearestPD(fisher_ξpm)
    else:
        F=fisher_ξpm 
    return F

##################################################################
##################################################################
##################################################################
def create_yaml(alphas):
    ### CREATE THE YAML FILE FILE
    info     = None
    template = f'{cocoa_path}/cocoa_photoz/yamls/roman_pca/modelvector_generator_template.yaml'
    
    with open(template, 'r') as t:
        template_lines = t.readlines()

    new_comb = f'Model_{sc_mod}_Fiducial_{sc_fid}r{sim}' # new combination
    new_yaml = f'{cocoa_path}/cocoa_photoz/yamls/roman_pca/modelvector_generator_{new_comb}.yaml'
    print(f"\033[32mNEW YAML CREATED AND SAVED @: {new_yaml}\033[0m")
    with open(new_yaml,'w') as f:
        for line in template_lines:
            if line.strip().startswith("output:"):
                new_line = f"output: ./cocoa_photoz/results/chains/roman_pca/modelvector_evaluation_Model_{sc_mod}_Fiducial_{sc_fid}r{sim}\n"
                f.write(new_line)
            elif line.strip() == "path: template":
                new_line = f"    path: ./external_modules/data/roman_real/{sc_mod}\n"
                f.write(new_line)
            elif line.strip() == "data_file: ''":
                new_line = f"    data_file: {new_comb}.dataset\n"
                f.write(new_line)
            elif line.strip().startswith("pcs_file:"):
                new_line = f"    pcs_file: D_{sc_mod}_weights{want_weights}.txt\n"
                f.write(new_line)
            elif line.strip() == "### ALPHAS START":
                f.write("  ### ALPHAS START\n")
                for alpha_name, alpha_value in zip(list(alphas.keys()),list(alphas.values())):
                    new_line=f'  {alpha_name}: {alpha_value}\n'
                    f.write(new_line)
            else:
                f.write(line)
    return None            

def create_datafile(eval):    
    ### CREATE THE .DATASET CONFIGURATION FILE
    new_comb = f'Model_{sc_mod}_Fiducial_{sc_fid}r{sim}' # new combination
    new_datafile      =  f'{cocoa_path}/projects/roman_real/data/{sc_mod}/{new_comb}.dataset'
    datafile_template = {'data_file'       : 'sc1bd4_r131103_fidRR.modelvector',
                         'cov_file'        : 'cov_sc1bd4',   # CHECK IMPORTANCE -DONE- The covariance change the chi2 value. Once you choose the covariance value, just be consistent and use the SAME covariance matrix trhough all the work!! I WILL USE THIS
                        #'cov_file'        : 'cov_sc1bd4r0', # CHECK IMPORTANCE -DONE- The covariance change the chi2 value. Once you choose the covariance value, just be consistent and use the SAME covariance matrix trhough all the work!! I WILL NOT USE THIS
                         'mask_file'       : 'mask_sc1bd4r131103.mask',
                         'nz_lens_file'    : 'foo.nz',
                         'nz_source_file'  : 'foo.nz',
                         'lens_ntomo'      : 9,
                         'source_ntomo'    : 9,
                         'n_theta'         : 15,
                         'theta_min_arcmin': 2.5,
                         'theta_max_arcmin': 250.
                         }
    match eval:
        case 0:
            # INITIZALIZATION PHASE
            # ARBITRARY DV 
            # CORRECT NZ
            with open(new_datafile,"w") as g:
                for df_key,df_val in zip(list(datafile_template.keys()),list(datafile_template.values())):
                    if df_key == 'nz_lens_file':
                        g.write(f'{df_key} = nz_fid_{sc_fid}r{sim}.nz\n')
                    elif df_key == 'nz_source_file':    
                        g.write(f'{df_key} = nz_fid_{sc_fid}r{sim}.nz\n')
                    else:
                        g.write(f'{df_key} = {df_val}\n')
        case 1:
            # CROSS-CHECK PHASE
            # CORRECT DV 
            # CORRECT NZ
            # if sim == None and sc_fid!=sc_mod and sc_fid in possible_scenarios:
                # shutil.(nz_fid_path)
            with open(new_datafile,"w") as g:
                for df_key,df_val in zip(list(datafile_template.keys()),list(datafile_template.values())):
                    if df_key == 'data_file':
                        g.write(f'{df_key} = Model_{sc_mod}_Fiducial_{sc_fid}r{sim}.modelvector\n')
                    elif df_key == 'nz_lens_file':
                        g.write(f'{df_key} = nz_fid_{sc_fid}r{sim}.nz\n')
                    elif df_key == 'nz_source_file':    
                        g.write(f'{df_key} = nz_fid_{sc_fid}r{sim}.nz\n')
                    else:
                        g.write(f'{df_key} = {df_val}\n')
        case 2:
            # COMPUTATIONAL PHASE
            # CORRECT DV 
            # MEAN NZ
            # FULL PCS OR FULL RECONSTRUCTION
            with open(new_datafile,"w") as g:
                for df_key,df_val in zip(list(datafile_template.keys()),list(datafile_template.values())):
                    if df_key == 'data_file':
                        g.write(f'{df_key} = Model_{sc_mod}_Fiducial_{sc_fid}r{sim}.modelvector\n')
                    elif df_key == 'nz_lens_file':
                        g.write(f'{df_key} = nbar_{sc_mod}.nz\n')
                    elif df_key == 'nz_source_file':    
                        g.write(f'{df_key} = nbar_{sc_mod}.nz\n')
                    else:
                        g.write(f'{df_key} = {df_val}\n')
        case 3:
            # COMPUTATIONAL PHASE
            # CORRECT DV 
            # MEAN NZ
            pass
    return None    

##################################################################
##################################################################
##################################################################
def projection(encoding,decoding,plot=False):
    nbar_path  = f'{cocoa_path}/projects/roman_real/data/{sc_mod}/nbar_{sc_mod}.nz'
    nbar       = np.genfromtxt(nbar_path)
    z          = nbar[:,0]

    if sim == None and sc_mod==sc_fid:
        # Trivial case: all amplitudes are zero.
        nz_fid = nbar
        nz_fid_path = f'{cocoa_path}/projects/roman_real/data/{sc_mod}/nz_fid_{sc_fid}r{sim}.nz'
        np.savetxt(nz_fid_path,nz_fid)
        print(f"\033[32mCASE 1:\nnz_fid_path  => {nz_fid_path}\nn_model_path => {nbar_path}\033[0m")
    elif sim == None and sc_fid!=sc_mod and sc_fid in possible_scenarios:
        # The mean n(z) used in the model n(z)_mean + ∑α*PC is from scenario `sc_mod`.
        # The fiducial DV is generated from the mean n(z) from scenario `sc_fid`.
        nz_fid_path = f'{cocoa_path}/projects/roman_real/data/{sc_fid}/nbar_{sc_fid}.nz'
        nz_fid = np.genfromtxt(nz_fid_path)
        norms = np.trapz(y=nz_fid[:,1:], x=z, axis=0)
        nz_fid[:,1:] /= norms
        print(f"\033[32mCASE 2:\nnz_fid_path  => {nz_fid_path}\nn_model_path => {nbar_path}\033[0m")
    elif sim != None and 0<=sim<=999999 and sc_fid in possible_scenarios:
        # The mean n(z) used in the model n(z)_mean + ∑α*PC is from scenario `sc_mod`.
        # The fiducial DV is generated from the realization n(z) with tag `sim` from the scenario `sc_fid`.
        nz_fid_path_h5py = f'{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_fid}/nz_samples_LHC0_pointZ_1e6_Roman_{sc_fid}.h5'
        nz_fid = h5py.File(nz_fid_path_h5py,'r') 
        nz_fid = np.array([nz_fid[f'bin{i}'][sim] for i in range(Nt)]).T
        norms = np.trapz(y=nz_fid, x=z, axis=0)
        nz_fid /= norms
        nz_fid = np.column_stack((z, nz_fid))
        nz_fid_path = f'{cocoa_path}/projects/roman_real/data/{sc_mod}/nz_fid_{sc_fid}r{sim}.nz'
        np.savetxt(nz_fid_path,nz_fid)
        print(f"\033[32mCASE 3:\nnz_fid_path  => {nz_fid_path}\nn_model_path => {nbar_path}\033[0m")

    ndiff = nz_fid[:,1:] - nbar[:,1:] # (46,9)
    decoding = decoding[:,:npcs_used]
    
    params_values={}
    u = encoding @ ndiff.T.flatten().T # Eq 21 of 2506.00758 (compression)

    print()
    print("ALPHAS WRITTEN TO YAML FILE (START)")
    print(ndiff.T.flatten().shape)
    print(ndiff.T.flatten().T.shape)
    for i in range(npcs_used):
        print(f'  roman_alpha_{i+1}: {u[i]}')
        params_values.update({f"roman_alpha_{i+1}": u[i]})
    create_yaml(alphas=params_values)
    print("ALPHAS WRITTEN TO YAML FILE (END)")

    np.savetxt(f'{cocoa_path}/projects/roman_real/data/{sc_mod}/nbar_{sc_mod}.nz',nbar) # nz_model = nbar

    alphas = np.array([params_values.get("roman_alpha_"+str(i+1)) for i in range(npcs_used)])
    correction = (alphas * decoding).sum(axis=1)   # Eq 22 of 2506.00758 (decompression) [projecting]

    s = nbar[:,1:].shape
    z = nbar[:,0]
    nz_model = nbar[:,1:].T.flatten() + correction # Eq 22 of 2506.00758 (decompression) [correcting]
    nz_model = nz_model.reshape(s[::-1]).T
    nz_model = np.column_stack((z,nz_model))

    nz_model[nz_model < 0] = 0
    norms = np.trapz(y=nz_model[:,1:],x=z,axis=0)
    nz_model[:,1:] /= norms    

    if plot:
        for i in range(1,10):
            plt.plot(z,nz_fid[:,i],c="red",label=f"fid {sc_fid}" if i==1 else None,lw="3",alpha=.4)
            plt.plot(z,nbar[:,i],c="gray",ls="--",label=f"mean {sc_mod}" if i==1 else None)
            plt.plot(z,nz_model[:,i],c="darkgreen",ls=":",label="recon" if i==1 else None)

        plt.legend(loc="lower center",ncols=3,fontsize=10,bbox_to_anchor=(0.5, 1.0),frameon=False)
        plt.xlabel(r"$z$")
        plt.ylabel(r"$n(z)$")
        plt.savefig(f"nz_Model_{sc_mod}_Fiducial_{sc_fid}r{sim}_npcs{npcs_used}.pdf")

    return params_values

def execute_end_to_end_pipeline(extra_info=True):
    if want_weights == 0:
        print("\033[31m!!-------ATTENTION--------!!\033[0m")
        print("\033[31m!!-------ATTENTION--------!!\033[0m")
        print(f"\033[31mNOT USING FISHER: want_weights={want_weights==1}\033[0m")
        print("\033[31m!!-------ATTENTION--------!!\033[0m")
        print("\033[31m!!-------ATTENTION--------!!\033[0m")
    Cn, _ndiff_ = roman_obj.cov_of_ndiff() # must be called before pcs
    F = fisher(want_weights=want_weights,
               Cn_shape=Cn.shape[0])
    E, D, ss, resid, Dvec, Dval, Cnvec, Cnval, Um, Vm, M = roman_obj.getModes(Fisher=F, Cn=Cn, chisq_threshold=0)
    if want_projection:
        alphas = projection(encoding=E,decoding=D,plot=True)

    if extra_info:
        idx_desc = np.argsort(Cnval)[::-1]
        Cnval = Cnval[idx_desc]
        Cnvec = Cnvec[:, idx_desc]

        X = D@E.T
        ss_tot = np.sum(ss)
        ss_rev = ss_tot - np.cumsum(ss)
        np.savetxt(f"{cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/E_{sc_mod}_weights{want_weights}.txt",E)
        np.savetxt(f"{cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/D_{sc_mod}_weights{want_weights}.txt",D)
        np.savetxt(f"{cocoa_path}/projects/roman_real/data/{sc_mod}/D_{sc_mod}_weights{want_weights}.txt" ,D) # Yes, we want D here too! 
        np.savetxt(f"{cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/X_{sc_mod}_weights{want_weights}.txt",X)
        np.savetxt(f"{cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/Cnvec_{sc_mod}_weights{want_weights}.txt",Cnvec)
        np.savetxt(f"{cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/ssrev_{sc_mod}_weights{want_weights}.txt",ss_rev) # ss "reversed", i.e., discounted from the total variance
        print()
        print(f"\033[32mDECODING MATRIX (D) SAVED      @: {cocoa_path}/projects/roman_real/data/{sc_mod}/D_{sc_mod}_weights{want_weights}.txt      (PCS FILE USED BY COCOA)\033[0m")
        print(f"\033[32mDECODING MATRIX (D) SAVED      @: {cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/D_{sc_mod}_weights{want_weights}.txt     (BACKUP ONLY)\033[0m")
        print(f"\033[32mENCODING MATRIX (E) SAVED      @: {cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/E_{sc_mod}_weights{want_weights}.txt     (BACKUP ONLY)\033[0m")
        print(f"\033[32mPROJECTION MATRIX (X=DE) SAVED @: {cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/X_{sc_mod}_weights{want_weights}.txt     (BACKUP ONLY)\033[0m")
        print(f"\033[32mEINGENVECTORS OF Cn SAVED      @: {cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/Cnvec_{sc_mod}_weights{want_weights}.txt (BACKUP ONLY)\033[0m")
        print(f"\033[32m χ² FROM DROPPED MODES SAVED   @: {cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/ssrev_{sc_mod}_weights{want_weights}.txt (BACKUP ONLY)\033[0m")
    return alphas

alphas = execute_end_to_end_pipeline(extra_info=False)

##################################################################
##################################################################
##################################################################

def execute_cocoa_with_yaml(yaml_info):
    # with suppress_stdout_stderr():
    model = get_model(yaml_info)   
    varied_points = {} # The yaml do not contain any varied parameter! 
    loglikes_i = model.loglikes(varied_points,as_dict=False,return_derived=False)[0]
    return loglikes_i

##################################################################
##################################################################
##################################################################

def chi2_vs_npcs(eval):
    ### INITIALIZE THE YAML FILE - START

    info = None
    new_comb = f'Model_{sc_mod}_Fiducial_{sc_fid}r{sim}' # new combination
    new_yaml = f'{cocoa_path}/cocoa_photoz/yamls/roman_pca/modelvector_generator_{new_comb}.yaml'
    info_txt = new_yaml

    # Load the yaml
    with open(info_txt) as f: 
        info = yaml_load(f).copy()

    # Adjust the paths for theory (camb) and likelihood (roman) codes
    del info["theory"]["camb"]["path"]
    info["packages_path"] = camb_path
    info["likelihood"]["roman_real.roman_real_cosmic_shear"]["path"] = f'{cocoa_path}/external_modules/data/roman_real/{sc_mod}'   
    ### INITIALIZE THE YAML FILE - END

    output_file = f"{cocoa_path}/chi2_vs_npcs_weighted/chi2_vs_npcs_Model_{sc_mod}_Fiducial_{sc_fid}r{sim}_weights{want_weights}_first_50PCs.txt"
    
    # if os.path.exists(output_file):
    #     print(f"❌ File already exists: {output_file}")
    #     sys.exit(1)
    # else:
    open(output_file,"w").close()

    print()
    print("---------------------------------//-----------------------------------")
    print("---------------------------------//-----------------------------------")
    print(f"\033[32m[INPUT] COCOA WILL USE YAML @: {info_txt}\033[0m")
    match eval:
        case 3:
            print(f"\033[32m[OUTPUT] SAVED AT @: {output_file}\033[0m")
    print("---------------------------------//-----------------------------------")
    print("---------------------------------//-----------------------------------\n")

    match eval:
        case 0:
            # INITIZALIZATION PHASE: DATAFILE IS MODIFIED ONCE, YAML IS MODIFIED ONCE
            # ARBITRARY DV 
            # CORRECT NZ
            print("\033[32mSTEP 1: INITIZALIZATION\033[0m")
            print("\033[31m(ARBITRARY DV)\033[0m")            
            print(f"\033[31m(CORRECT NZ: {sc_fid}. REALIZATION r={sim})\033[0m") 
            print(f"\033[31m(GENERATING THE CORRECT DV)\033[0m") 
            create_datafile(eval=eval)
            info["likelihood"]["roman_real.roman_real_cosmic_shear"]["print_datavector"] = True
            info["likelihood"]["roman_real.roman_real_cosmic_shear"]["print_datavector_file"] =\
                  f'{cocoa_path}/projects/roman_real/data/{sc_mod}/Model_{sc_mod}_Fiducial_{sc_fid}r{sim}.modelvector'
            loglikes_eval0 = execute_cocoa_with_yaml(info)
            chi2_eval0 = -2*loglikes_eval0
            print(f"\033[32mARBITRARY χ²: {chi2_eval0}\033[0m")
        case 1:
            # CROSS-CHECK PHASE: DATAFILE IS MODIFIED ONCE, YAML IS MODIFIED ONCE
            # CORRECT DV 
            # CORRECT NZ
            external_modeling = info["likelihood"]["roman_real.roman_real_cosmic_shear"]["external_nz_modeling"]
            will_print_dv     = info["likelihood"]["roman_real.roman_real_cosmic_shear"]["print_datavector"]
            print("\033[32mSTEP 2: CROSS-CHECK A (SANITY CHECK)\033[0m")
            print("\033[31m(CORRECT DV)\033[0m")            
            print(f"\033[31m(CORRECT NZ: {sc_fid}. REALIZATION r={sim})\033[0m") 
            print(f"\033[31m(EXTERNAL NZ MODELING = {external_modeling}) - MUST BE 0!\033[0m") 
            print("\033[31m(# OF PCS = 0)\033[0m") 
            print(f"\033[31m(WILL PRINT DV?: {will_print_dv}) - MUST BE FALSE!\033[0m") 
            create_datafile(eval=eval)
            info["likelihood"]["roman_real.roman_real_cosmic_shear"]["print_datavector_file"] = ''
            loglikes_eval1 = execute_cocoa_with_yaml(info)
            chi2_eval1 = -2*loglikes_eval1
            print(f"\033[32mCOMPUTED χ²: {chi2_eval1} - MUST BE ~ 0!\033[0m")
        case 2:
            # CROSS-CHECK PHASE: DATAFILE IS MODIFIED ONCE, YAML IS MODIFIED ONCE
            # CORRECT DV 
            # MEAN NZ
            # FULL PCS OR FULL RECONSTRUCTION
            info["likelihood"]["roman_real.roman_real_cosmic_shear"]["print_datavector"] = False
            info["likelihood"]["roman_real.roman_real_cosmic_shear"]["print_datavector_file"] = ''
            info["likelihood"]["roman_real.roman_real_cosmic_shear"]["external_nz_modeling"] = int(1)
            external_modeling = info["likelihood"]["roman_real.roman_real_cosmic_shear"]["external_nz_modeling"]
            info["likelihood"]["roman_real.roman_real_cosmic_shear"]["npcs_nz"] = int(414)
            using_npcs = info["likelihood"]["roman_real.roman_real_cosmic_shear"]["npcs_nz"]
            yaml_params = info["params"].keys()
            alpha_keys = [k for k in yaml_params if k.startswith("roman_alpha_")]
            print("\033[32mSTEP 3: CRITICAL CROSS-CHECK B (SANITY CHECK)\033[0m")
            print("\033[31m(CORRECT DV)\033[0m")            
            print(f"\033[31m(MEAN NZ: {sc_mod})\033[0m") 
            print(f"\033[31m(EXTERNAL MODELING = {external_modeling}) - MUST BE 1!\033[0m") 
            print(f"\033[31m(# OF PCS = {using_npcs}). HOW MANY ALPHAS? {len(alpha_keys)} - MUST BE {npcs_used}!\033[0m") 
            create_datafile(eval=eval)
            loglikes_eval2 = execute_cocoa_with_yaml(info)
            chi2_eval2 = -2*loglikes_eval2
            print(f"Model: {rename_scenarios[sc_mod]}, Fiducial: {rename_scenarios[sc_fid]}")
            print(f"\033[32mCOMPUTED χ²: {chi2_eval2} - MUST BE ~ 0!\033[0m")
            print(f"roman_alpha_1: {alphas['roman_alpha_1']}")
            print(f"roman_alpha_414: {alphas['roman_alpha_414']}")
        case 3:
            # COMPUTATIONAL PHASE: DATAFILE MODIFIED ONCE, YAML IS MODIFIED npcs_used TIMES
            # CORRECT DV 
            # MEAN NZ
            print("=======================================")
            print("=======================================")
            print("\033[32mSTEP 4: COMPUTATION\033[0m")
            print("=======================================")
            print("=======================================")
            print("\033[31m(CORRECT DV)\033[0m")            
            print(f"\033[31m(MEAN NZ: {sc_mod})\033[0m") 
            print("\033[31m(INCREMENTAL # OF PCS)\033[0m") 
            create_datafile(eval=eval)
            info["likelihood"]["roman_real.roman_real_cosmic_shear"]["print_datavector"] = False
            info["likelihood"]["roman_real.roman_real_cosmic_shear"]["print_datavector_file"] = ''
            info["likelihood"]["roman_real.roman_real_cosmic_shear"]["external_nz_modeling"] = int(1)
            with open(output_file,"a",buffering=1) as f:
                f.write(f"# Using yaml {info_txt}\n")
                f.write("# PCs, χ²\n")
                print("# of PCs , χ²")
                npcs_initial = 0
                npcs_final   = npcs_used
                for n in range(npcs_initial,npcs_final):
                    info["likelihood"]["roman_real.roman_real_cosmic_shear"]["npcs_nz"] = n
                    loglikes_i = execute_cocoa_with_yaml(info)
                    chi2_i = -2*loglikes_i
                    f.write(f"{n}, {chi2_i}\n")
                    # print(f"# of PCs: {info['likelihood']['roman_real.roman_real_cosmic_shear']['npcs_nz']}, χ²: {chi2_i}\n",end='') # For terminal
                    print(f"{info['likelihood']['roman_real.roman_real_cosmic_shear']['npcs_nz']}, {chi2_i}") # for Slurm
                    if n==50:
                        print(f"Computed {n} PCs. ENOUGH FOR NOW (CHANGE IF YOU WANT MORE)")
                        sys.exit(1)

    return None        

def execute_all_steps():
    for eval in [0,1,2,3]:
        # 0: INITIALIZATION
        # 1: CROSS-CHECK 1
        # 2: CROSS-CHECK 2
        # 3: COMPUTATION
        chi2_vs_npcs(eval=eval)
execute_all_steps()


def compute_alpha_distribution(compute_simple=False,compute_specific=False,plot_simple=False,plot_specific=False):
    if compute_simple==True and compute_specific==False:
        # Scenario A x Scenario A
        E = np.genfromtxt(f"{cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/E_{sc_mod}_weights{want_weights}.txt")
        ndiff = np.load(f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_mod}/ndiff_{sc_mod}.npy")
        print(E.shape)
        print(ndiff.T.shape)
        alphas_distr = E @ ndiff.T
        np.save(f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_mod}/alphas_distri_E_{sc_mod}@ndiff.T_{sc_mod}.npy", alphas_distr)
        del alphas_distr,E,ndiff
    elif compute_simple==False and compute_specific==True:
        # (1) Scenario Model x Scenario Model    
        print('STEP 1 START')
        E_m_file    =f"{cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/E_{sc_mod}_weights{want_weights}.txt"
        ndiff_m_file=f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_mod}/ndiff_{sc_mod}.npy"
        outs_mm_file=f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_mod}/alphas_distri_E_{sc_mod}@ndiff.T_{sc_mod}.npy" 

        E_m = np.genfromtxt(E_m_file)
        ndiff_m = np.load(ndiff_m_file)
        print(E_m.shape)
        print(ndiff_m.T.shape)
        alphas_distr_mm = E_m @ ndiff_m.T
        np.save(outs_mm_file, alphas_distr_mm)
        del alphas_distr_mm, E_m_file, ndiff_m_file, outs_mm_file
        print('STEP 1 ENDS')

        # (2) Scenario Fiducial x Scenario Fiducial  
        print('STEP 2 START')  
        E_f_file    =f"{cocoa_path}/jacobian_outs/stencil_5pt_{sc_fid}/E_{sc_fid}_weights{want_weights}.txt"
        ndiff_f_file=f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_fid}/ndiff_{sc_fid}.npy"
        outs_ff_file=f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_fid}/alphas_distri_E_{sc_fid}@ndiff.T_{sc_fid}.npy" 

        E_f     = np.genfromtxt(E_f_file)
        ndiff_f = np.load(ndiff_f_file)
        print(E_f.shape)
        print(ndiff_f.T.shape)
        alphas_distr_ff = E_f @ ndiff_f.T
        np.save(outs_ff_file, alphas_distr_ff)
        del alphas_distr_ff,E_f_file,ndiff_f_file,outs_ff_file 
        print('STEP 2 ENDS')
        
        # (3) Scenario Model x Scenario Fiducial    
        print('STEP 3 START')
        E_m_file    =f"{cocoa_path}/jacobian_outs/stencil_5pt_{sc_mod}/E_{sc_mod}_weights{want_weights}.txt"
        ndiff_f_file=f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_fid}/ndiff_{sc_fid}.npy"
        outs_mf_file=f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_mod}/alphas_distri_E_{sc_mod}@ndiff.T_{sc_fid}.npy" 

        E_m = np.genfromtxt(E_m_file)
        ndiff_f = np.load(ndiff_f_file)
        print(E_m.shape)
        print(ndiff_f.T.shape)
        alphas_distr_mf = E_m @ ndiff_f.T
        np.save(outs_mf_file, alphas_distr_mf)
        del alphas_distr_mf,E_m_file,ndiff_f_file,outs_mf_file
        print('STEP 3 ENDS')
        
        # (4) Scenario Fiducial x Scenario Model  
        print('STEP 4 START')  
        E_f_file    =f"{cocoa_path}/jacobian_outs/stencil_5pt_{sc_fid}/E_{sc_fid}_weights{want_weights}.txt"
        ndiff_m_file=f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_mod}/ndiff_{sc_mod}.npy"
        outs_fm_file=f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_fid}/alphas_distri_E_{sc_fid}@ndiff.T_{sc_mod}.npy" 

        E_f = np.genfromtxt(E_f_file)
        ndiff_m = np.load(ndiff_m_file)
        print(E_f.shape)
        print(ndiff_m.T.shape)
        alphas_distr_fm = E_f @ ndiff_m.T
        np.save(outs_fm_file, alphas_distr_fm)
        del alphas_distr_fm,E_f_file,ndiff_m_file,outs_fm_file
        print('STEP 4 ENDS')

    if plot_simple==True and plot_specific==False:
        test = np.load(f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_mod}/alphas_distri_E_{sc_mod}@ndiff.T_{sc_mod}.npy")
        test_partial = test[:,:100000].T

        nEig=414
        names = ["projected_alpha%s" %i for i in range(nEig)]
        labels = [rf"u_{{{i+1}}}" for i in range(nEig)]

        chains = MCSamples(samples=test_partial,names=names,labels=labels)

        ndim = 5
        nsamp = 10000
        random_state = np.random.default_rng(10)
        A = random_state.random((ndim, ndim))
        cov = np.identity(5)

        samps = random_state.multivariate_normal([0] * ndim, cov, size=nsamp)
        names = ["projected_alpha%s" % i for i in range(ndim)]
        labels = [rf"u_{{{i}}}" for i in range(ndim)]
        lims = {f"projected_alpha{i}": (-4,4) for i in range(5)}
        samples_gauss = MCSamples(samples=samps, names=names, labels=labels)

        g = plots.get_subplot_plotter(subplot_size=1)
        g.settings.title_limit_fontsize = 13
        g.settings.legend_fontsize = 20
        g.settings.axes_labelsize = 20
        g.settings.axes_fontsize = 15

        g.triangle_plot([chains,samples_gauss],
                        ["projected_alpha%s" %i for i in range(5)],
                        legend_labels=["M1-D1","Gauss(0,1)"],
                        contour_colors=["#1b5f6f","black"],
                        contour_ls=['-',':'],
                        contour_lws=[1.5,1.3],
                        param_limits=lims,
                        filled=[True,False],
                        line_args=[{"ls": "-", "color": "#1b5f6f",'lw':1.5},
                                {"ls": ":", "color": "black",'lw':1.3}],
                        )

        plt.savefig("alphas_distributions_simple.pdf")
    elif plot_simple==False and plot_specific==True:    
        print('PLOT SPECIFIC START')
        alphas_mm = np.load(f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_mod}/alphas_distri_E_{sc_mod}@ndiff.T_{sc_mod}.npy")
        alphas_ff = np.load(f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_fid}/alphas_distri_E_{sc_fid}@ndiff.T_{sc_fid}.npy")
        alphas_mf = np.load(f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_mod}/alphas_distri_E_{sc_mod}@ndiff.T_{sc_fid}.npy")
        alphas_fm = np.load(f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_fid}/alphas_distri_E_{sc_fid}@ndiff.T_{sc_mod}.npy")
        alphas_partial_mm = alphas_mm[:,:10000].T
        alphas_partial_ff = alphas_ff[:,:10000].T
        alphas_partial_mf = alphas_mf[:,:10000].T
        alphas_partial_fm = alphas_fm[:,:10000].T

        nEig=414
        names = ["projected_alpha%s" %i for i in range(nEig)]
        labels = [rf"u_{{{i+1}}}" for i in range(nEig)]

        chains_mm = MCSamples(samples=alphas_partial_mm,names=names,labels=labels)
        chains_ff = MCSamples(samples=alphas_partial_ff,names=names,labels=labels)
        chains_mf = MCSamples(samples=alphas_partial_mf,names=names,labels=labels)
        chains_fm = MCSamples(samples=alphas_partial_fm,names=names,labels=labels)

        ndim = 5
        nsamp = 10000
        random_state = np.random.default_rng(10)
        A = random_state.random((ndim, ndim))
        cov = np.identity(5)

        samps = random_state.multivariate_normal([0] * ndim, cov, size=nsamp)
        names = ["projected_alpha%s" % i for i in range(ndim)]
        labels = [rf"u_{{{i}}}" for i in range(ndim)]
        lims = {f"projected_alpha{i}": (-4,4) for i in range(5)}
        samples = MCSamples(samples=samps, names=names, labels=labels)

        g = plots.get_subplot_plotter(subplot_size=1)
        g.settings.title_limit_fontsize = 13
        g.settings.legend_fontsize = 20
        g.settings.axes_labelsize = 20
        g.settings.axes_fontsize = 15

        g.triangle_plot([chains_mm,
                         chains_ff,
                         chains_mf,
                         chains_fm],
                        ["projected_alpha%s" %i for i in range(5)],
                        legend_labels=[r"E: 1, $\Delta$: 1",
                                       r"E: 2, $\Delta$: 2",
                                       r"E: 1, $\Delta$: 2",
                                       r"E: 2, $\Delta$: 1",
                                        ],
                        legend_loc="upper right",
                        contour_colors=["#1b5f6f","#E69F00", "#882255", "#56B4E9"],
                        contour_ls=['-','--','-.',':'],
                        contour_lws=[1.5,1.5,1.5,1.5],
                        param_limits=lims,
                        filled=[True,False,False,False],
                        line_args=[
                                {"ls": "-" , "color": "#1b5f6f",'lw':1.5},
                                {"ls": "--", "color": "#E69F00",'lw':1.5},
                                {"ls": "-.", "color": "#882255",'lw':1.5},
                                {"ls": ":" , "color": "#56B4E9",'lw':1.5}
                                ],
                        )     
        plt.savefig("alphas_distributions_mix.pdf")
        print("alphas_distributions_mix.pdf")

    elif plot_simple==True and plot_specific==True:
        print('PLOT DOUBLE START')
        alphas_mm = np.load(f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_mod}/alphas_distri_E_{sc_mod}@ndiff.T_{sc_mod}.npy")
        alphas_ff = np.load(f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_fid}/alphas_distri_E_{sc_fid}@ndiff.T_{sc_fid}.npy")
        alphas_mf = np.load(f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_mod}/alphas_distri_E_{sc_mod}@ndiff.T_{sc_fid}.npy")
        alphas_fm = np.load(f"{cocoa_path}/cocoa_photoz/roman_nz_realizations/{sc_fid}/alphas_distri_E_{sc_fid}@ndiff.T_{sc_mod}.npy")
        alphas_partial_mm = alphas_mm[:,:10000].T
        alphas_partial_ff = alphas_ff[:,:10000].T
        alphas_partial_mf = alphas_mf[:,:10000].T
        alphas_partial_fm = alphas_fm[:,:10000].T

        nEig=414
        names = ["projected_alpha%s" %i for i in range(nEig)]
        labels = [rf"u_{{{i+1}}}" for i in range(nEig)]

        chains_mm = MCSamples(samples=alphas_partial_mm,names=names,labels=labels)
        chains_ff = MCSamples(samples=alphas_partial_ff,names=names,labels=labels)
        chains_mf = MCSamples(samples=alphas_partial_mf,names=names,labels=labels)
        chains_fm = MCSamples(samples=alphas_partial_fm,names=names,labels=labels)

        ndim = 5
        nsamp = 10000
        random_state = np.random.default_rng(10)
        A = random_state.random((ndim, ndim))
        cov = np.identity(5)

        samps = random_state.multivariate_normal([0] * ndim, cov, size=nsamp)
        names = ["projected_alpha%s" % i for i in range(ndim)]
        labels = [rf"u_{{{i}}}" for i in range(ndim)]
        lims = {f"projected_alpha{i}": (-4.5,4.5) for i in range(5)}
        samples_gauss = MCSamples(samples=samps, names=names, labels=labels)

        g = plots.get_subplot_plotter(subplot_size=1)
        g.settings.title_limit_fontsize = 13
        g.settings.legend_fontsize = 13.5
        g.settings.axes_labelsize = 20
        g.settings.axes_fontsize = 15

        g.triangle_plot(roots=[
                         chains_mf,
                         chains_fm],
                         upper_roots=[chains_mm,
                                      samples_gauss],
                        params=["projected_alpha%s" %i for i in range(5)],
                        legend_loc="upper right",
                        contour_colors=["#E69F00", "#882255"],
                        upper_kwargs={"contour_colors": ["#1b5f6f","black"],
                                     "filled":[True,False],
                                     "contour_ls":["-",":"],
                                     "line_args":[{"ls": "-", "color": "#1b5f6f",'lw':1.5},
                                                  {"ls": ":", "color": "black",'lw':1.3}],
                                     },
                        legend_labels=[
                                       r"E: 1, $\Delta$: 2",
                                       r"E: 2, $\Delta$: 1",
                                       r"E: 1, $\Delta$: 1",
                                       r"Gauss(0,1)",
                                        ],
                        contour_ls=['--','-.','-',':'],
                        contour_lws=[1.5,1.5,1.5,1.5],
                        param_limits=lims,
                        filled=[True,False,False,False],
                        line_args=[
                                {"ls": "--", "color": "#E69F00",'lw':1.5},
                                {"ls": "-.", "color": "#882255",'lw':1.5},
                                ],
                        legend_ncol=4,
                        )     
        plt.savefig("alphas_distributions_double.pdf")
        # For alphas distributions with smaller smoothing scale use settings={"mult_bias_correction_order": 0, "smooth_scale_2D": 0.1, "smooth_scale_1D": 0.1}. See https://getdist.readthedocs.io/en/latest/plot_gallery.html for more details
        # alphas_distributions_double_smooth_scale_0.1.pdf
        print("alphas_distributions_double.pdf")

    return None
# compute_alpha_distribution(compute_simple  =False,
#                            compute_specific=False,
#                            plot_simple     =True,
#                            plot_specific   =True)