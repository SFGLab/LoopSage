#Basic Libraries
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import scipy.stats as stats
from tqdm import tqdm

# scipy
from scipy.stats import norm
from scipy.stats import poisson

# My own libraries
from LoopSage_preproc import *
from LoopSage_plots import *
from LoopSage_md import *
from LoopSage_em import *

def Kappa(mi,ni,mj,nj):
    k=0
    if mi<mj and mj<ni and ni<nj: k+=1 # np.abs(ni-mj)+1
    if mj<mi and mi<nj and nj<ni: k+=1 # np.abs(nj-mi)+1
    if mj==ni or mi==nj or ni==nj or mi==mj: k+=1
    return k

class LoopSage:
    def __init__(self,region,chrom,bedpe_file,N_beads=None,N_lef=None,f=None,b=None,kappa=None,r=None,track_file=None,bw_files=None,label=None):
        '''
        Definition of simulation parameters and input files.
        
        region (list): [start,end].
        chrom (str): indicator of chromosome.
        bedpe_file (str): path where is the bedpe file with CTCF loops.
        track_file (str): bigwig file with cohesin coverage.
        bw_files (list of str): bigwig files with (or other protein of interest) coverage.
        N_beads (int): number of monomers in the polymer chain.
        N_lef (int): number of cohesins in the system.
        kappa (float): LEF crossing coefficient of Hamiltonian.
        f (float): folding coeffient of Hamiltonian.
        b (float): binding coefficient of Hamiltonian.
        r (list): strength of each ChIP-Seq experinment.
        '''

        self.N_beads = N_beads
        self.N_bws = len(bw_files) if np.all(bw_files!=None) else 0
        print('Number of beads:',self.N_beads)
        self.chrom, self.region = chrom, region
        self.bedpe_file, self.bw_files, self.track_file = bedpe_file, bw_files, track_file
        self.preprocessing()
        self.kappa = 1e7 if kappa==None else kappa
        self.b = -self.N_beads if b==None else b
        self.r = np.full(self.N_bws,-self.N_beads/5) if (np.any(r==None) and self.N_bws>0) else r
        self.states = np.full(self.N_beads,False)
        self.avg_loop, self.max_loop = int(np.average(self.dists))+1, int(np.max(self.dists))+1
        self.log_avg_loop = np.average(np.log(self.dists+1))
        self.params = stats.maxwell.fit(self.dists)
        self.loop_pdist = stats.maxwell.pdf(np.arange(self.N_beads), *self.params)
        self.N_lef = self.N_CTCF
        print('Number of LEFs:',self.N_lef)
        self.f = -1000*np.sqrt(self.log_avg_loop/3.5)*self.N_CTCF/self.N_lef if f==None else f
        self.path = make_folder(self.N_beads,self.N_lef,self.region,self.chrom,label=label)
    
    def E_bind(self,ms,ns):
        '''
        Calculation of the CTCF binding energy. Needs cohesins positions as input.
        '''
        binding = np.sum(self.L[ms]+self.R[ns]) 
        E_b = self.b*binding/(np.sum(self.L)+np.sum(self.R)) 
        return E_b
    
    def E_bw(self,ms,ns):
        '''
        Calculation of the RNApII binding energy. Needs cohesins positions as input.
        '''
        E_bw = 0
        for i in range(self.N_bws):
            E_bw += self.r[i]*np.sum(self.BWs[i,ms]+self.BWs[i,ns])/np.sum(self.BWs[i])
        return E_bw

    def E_cross(self,ms,ns):
        '''
        Calculation of the cohesin crossing energy. Needs cohesins positions as input.
        '''
        crossing = 0
        for i in range(self.N_lef):
            for j in range(i+1,self.N_lef):
                crossing+=Kappa(ms[i],ns[i],ms[j],ns[j])
        return self.kappa*crossing/self.N_lef
    
    def E_fold(self,ms,ns):
        '''
        Calculation of the folding energy (or entropic cost) for the formation of loops. Needs cohesins positions as input.
        '''
        folding=np.sum(np.log(ns-ms))
        return self.f*folding/(self.N_lef*self.log_avg_loop)
    
    def get_E(self,ms,ns):
        '''
        Calculation of the total energy as sum of the specific energies of the system.
        Needs cohesins positions as input.
        '''
        if np.any(self.BWs==None):
            energy=self.E_bind(ms,ns)+self.E_cross(ms,ns)+self.E_fold(ms,ns)
        else:
            energy=self.E_bind(ms,ns)+self.E_cross(ms,ns)+self.E_fold(ms,ns)+self.E_bw(ms,ns)
        return energy

    def get_dE_bind(self,ms,ns,m_new,n_new,idx):
        return self.b*(self.L[m_new]+self.R[n_new]-self.L[ms[idx]]-self.R[ns[idx]])/(np.sum(self.L)+np.sum(self.R))

    def get_dE_fold(self,ms,ns,m_new,n_new,idx):
        return self.f*(np.log(n_new-m_new)-np.log(ns[idx]-ms[idx]))/(self.N_lef*self.log_avg_loop)

    def get_dE_bw(self,ms,ns,m_new,n_new,idx):
        dE_bw = 0
        for i in range(self.N_bws):
            dE_bw += self.r[i]*(self.BWs[i,m_new]+self.BWs[i,n_new]-self.BWs[i,ms[idx]]-self.BWs[i,ns[idx]])/np.sum(self.BWs[i])
        return dE_bw
    
    def get_dE_cross(self,ms,ns,m_new,n_new,idx):
        K1, K2 = 0, 0
        for i in range(self.N_lef):
            if i!=idx: K1+=Kappa(ms[idx],ns[idx],ms[i],ns[i])
        for i in range(self.N_lef):
            if i!=idx: K2+=Kappa(m_new,n_new,ms[i],ns[i])
        return self.kappa*(K2-K1)/self.N_lef

    def get_dE(self,ms,ns,m_new,n_new,idx):
        '''
        Calculation of the energy difference.

        ms, ns (np arrays): cohesin positions.
        m_new, n_new (ints): the two new cohesin positions of the cohesin of interest.
        idx (int): the index that represent the cohesin of interest.
        '''
        dE_bind = self.get_dE_bind(ms,ns,m_new,n_new,idx)
        dE_fold = self.get_dE_fold(ms,ns,m_new,n_new,idx)
        if np.all(self.BWs!=None): dE_bw = self.get_dE_bw(ms,ns,m_new,n_new,idx)
        dE_cross = self.get_dE_cross(ms,ns,m_new,n_new,idx)
        dE = dE_bind+dE_fold+dE_cross+dE_bw if np.all(self.BWs!=None) else dE_bind+dE_fold+dE_cross
        return dE

    def unfolding_metric(self,ms,ns):
        '''
        This is a metric for the number of gaps (regions unfolded that are not within a loop).
        Cohesin positions are needed as input.
        '''
        fiber = np.zeros(self.N_beads)
        for i in range(self.N_lef):
            fiber[ms[i]:ns[i]]=1
        unfold = 2*(self.N_beads-np.count_nonzero(fiber))/self.N_beads
        return unfold
    
    def unbind_bind(self,poisson_choice=True):
        '''
        Implements one of the Monte Carlo moves.
        A cohesin unbinds from a specific position and loads randomly in different part of polymer.
        In case that there is cohesin track, there is preferential loading of cohesin.
        The left cohesin position is chosen randomly one from the available empty monomers.
        The right cohesin position is chosen from poisson distribution with average <average loop>/8.
        _____________________________________________________________________________________________
        poisson choice (bool): True if it is needed to choose initial cohesin positions from poisson 
                               distribution.
        '''
        # bind left part of cohesin to a random available place
        if np.all(self.track==None):
            m_new = rd.randint(0,self.N_beads-2)
        else:
            m_new = rd.choices(np.arange(self.N_beads-2), weights=self.track[:-2], k=1)[0]
        
        # bind right part of cohesin somewhere close to the left part
        n_new = m_new+5 if not poisson_choice else m_new+1+poisson.rvs(self.avg_loop//8)
        if n_new>=self.N_beads: n_new = rd.randint(m_new+1,self.N_beads-1)
        return int(m_new), int(n_new)

    def slide_right(self,m_old,n_old):
        '''
        Monte Carlo move where a chosen cohesin does one step right.
        '''
        if n_old+1<self.N_beads:
            m_new, n_new = m_old, n_old+1
        else:
            m_new, n_new = m_old, n_old
        return m_new, n_new

    def slide_left(self,m_old,n_old,mode='d'):
        '''
        Monte Carlo move where a chosen cohesin does one step left.
        '''
        if m_old-1>0:
            m_new, n_new = m_old-1,n_old
        else:
            m_new, n_new = m_old,n_old
        return m_new, n_new

    def initialize(self):
        '''
        Random initialization of polymer DNA fiber with some cohesin positions.
        '''
        ms, ns = np.zeros(self.N_lef).astype(int), np.zeros(self.N_lef).astype(int)
        for i in range(self.N_lef):
            ms[i], ns[i] = self.unbind_bind()
        return ms, ns
    
    def run_energy_minimization(self,N_steps,MC_step,burnin,T=1,T_min=0,poisson_choice=True,mode='Metropolis',viz=False,save=False, m_init=None, n_init=None):
        '''
        Implementation of the stochastic Monte Carlo simulation.

        Input parameters:
        N_steps (int): number of Monte Carlo steps.
        MC_step (int): sampling frequency.
        burnin (int): definition of the burnin period.
        T (float): simulation (initial) temperature.
        mode (str): it can be either 'Metropolis' or 'Annealing'.
        viz (bool): True in case that user wants to see plots.
        vid (bool): it creates a funky video with loops how they extrude in 1D.
        m_init (numpy array): to import initial state of preference for left positions of LEFs.
        n_init (numpy array): to import initial state of preference for right positions of LEFs.
        '''
        self.Ti = T
        Ts = list()
        self.burnin, self.MC_step = burnin, MC_step 
        bi = burnin//MC_step
        if np.any(m_init==None) or np.any(n_init==None):
            ms, ns = self.initialize()
        else:
            ms, ns = m_init, n_init
        E = self.get_E(ms,ns)
        Es,Ks,Fs,Bs,ufs = list(),list(),list(),list(),list()
        self.Ms, self.Ns = np.zeros((self.N_lef,N_steps)).astype(int), np.zeros((self.N_lef,N_steps)).astype(int)

        if viz: print('Running simulation...')
        for i in tqdm(range(N_steps),disable=(not viz)):
            self.Ti = T-(T-T_min)*(i+1)/N_steps if mode=='Annealing' else T
            Ts.append(self.Ti)
            for j in range(self.N_lef):
                # Randomly choose a move (sliding or rebinding)
                r = rd.choice([0,1,2])
                if r==0:
                    m_new, n_new = self.unbind_bind(poisson_choice)
                elif r==1:
                    m_new, n_new = self.slide_right(ms[j],ns[j])
                else:
                    m_new, n_new = self.slide_left(ms[j],ns[j])

                # Compute energy difference
                dE = self.get_dE(ms,ns,m_new,n_new,j)
                if dE <= 0 or np.exp(-dE/self.Ti) > np.random.rand():
                    ms[j], ns[j] = m_new, n_new
                    E += dE
                
                # Save trajectories
                self.Ms[j,i], self.Ns[j,i] = ms[j], ns[j]
            
            # Compute Metrics
            if i%MC_step==0:
                ufs.append(self.unfolding_metric(ms,ns))
                Es.append(E)
                Ks.append(self.E_cross(ms,ns))
                Fs.append(self.E_fold(ms,ns))
                Bs.append(self.E_bind(ms,ns))
                
        if viz: print('Done! ;D')

        # Save simulation info
        if save:
            f = open(self.path+'/other/info.txt', "w")
            f.write(f'Number of beads {self.N_beads}.\n')
            f.write(f'Number of cohesins {self.N_lef}. Number of CTCFs {self.N_CTCF}.\n')
            f.write(f'Bedpe file for CTCF binding is {self.bedpe_file}.\n')
            f.write(f'LEF track file for LEF preferential relocation is {self.track_file}.\n')
            if self.bw_files!=None:
                for bw_f in self.bw_files:
                    f.write(f'BWs track file for BWs binding potential is {bw_f}.\n')
            f.write(f'Initial temperature {T}. Minimum temperature {T_min}.\n')
            f.write(f'Monte Carlo optimization method: {mode}.\n')
            f.write(f'Monte Carlo steps {N_steps}. Sampling frequency {MC_step}. Burnin period {burnin}.\n')
            f.write(f'Crossing energy in equilibrium is {np.average(Ks[bi:]):.2f}. Crossing coefficient kappa={self.kappa}.\n')
            f.write(f'Folding energy in equilibrium is {np.average(Fs[bi:]):.2f}. Folding coefficient f={self.f}.\n')
            f.write(f'Binding energy in equilibrium is {np.average(Bs[bi:]):.2f}. Binding coefficient b={self.b}.\n')
            f.write(f'Energy at equillibrium: {np.average(Es[bi:]):.2f}.\n')
            f.close()

            np.save(self.path+'/other/Ms.npy',self.Ms)
            np.save(self.path+'/other/Ns.npy',self.Ns)
            np.save(self.path+'/other/Ts.npy',Ts)
            np.save(self.path+'/other/ufs.npy',ufs)
            np.save(self.path+'/other/Es.npy',Es)
            np.save(self.path+'/other/Bs.npy',Bs)
            np.save(self.path+'/other/Fs.npy',Fs)
            np.save(self.path+'/other/Ks.npy',Ks)
        
        # Some vizualizations
        if viz: coh_traj_plot(self.Ms,self.Ns,self.N_beads, self.path)
        if viz: make_timeplots(Es, Bs, Ks, Fs, bi, mode, self.path)
        if viz: coh_probdist_plot(self.Ms,self.Ns,self.N_beads,self.path)
        if viz and self.N_beads<=2000: stochastic_heatmap(self.Ms,self.Ns,MC_step,self.N_beads,self.path)
        if viz and self.N_beads<=2000: make_loop_hist(self.Ms,self.Ns,self.path)
        
        return Es, self.Ms, self.Ns, Bs, Ks, Fs, ufs

    def preprocessing(self):
        self.L, self.R, self.dists = binding_vectors_from_bedpe(self.bedpe_file,self.N_beads,self.region,self.chrom,False,False)
        self.BWs = np.zeros((self.N_bws,self.N_beads))
        if np.all(self.bw_files!=None):
            for i, f in enumerate(self.bw_files):
                self.BWs[i,:] = load_track(file=f,region=self.region,chrom=self.chrom,N_beads=self.N_beads,viz=False) if np.all(self.bw_files!=None) else None
        self.track = load_track(self.track_file,self.region,self.chrom,self.N_beads,False,True) if np.all(self.track_file!=None) else None
        self.N_CTCF = np.max([np.count_nonzero(self.L),np.count_nonzero(self.R)])
        print('Number of CTCF:',self.N_CTCF)

    def run_EM(self,platform='CUDA'):
        em = EM_LE(self.Ms,self.Ns,self.N_beads,self.burnin,self.MC_step,self.path,platform)
        sim_heat = em.run_pipeline(write_files=True,plots=True)
        corr_exp_heat(sim_heat,self.bedpe_file,self.region,self.chrom,self.N_beads,self.path)

    def run_MD(self,platform='CUDA'):
        md = MD_LE(self.Ms,self.Ns,self.N_beads,self.burnin,self.MC_step,self.path,platform)
        sim_heat = md.run_pipeline(write_files=True,plots=True)
        corr_exp_heat(sim_heat,self.bedpe_file,self.region,self.chrom,self.N_beads,self.path)

def main():
    # Definition of Monte Carlo parameters
    N_steps, MC_step, burnin, T, T_min = int(2e4), int(1e2), 1000, 5,1
    
    # Definition of region
    region, chrom = [178421513, 179491193], 'chr1'

    # Definition of data
    label=f'method_paper_Annealing'
    bedpe_file = '/home/skorsak/Documents/data/method_paper_data/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe'
    # coh_track_file = '/mnt/raid/data/Petros_project/bw/RAD21_ChIPseq/mm_BMDM_WT_Rad21_heme_60min.bw'
    # bw_file1 = '/home/skorsak/Documents/data/Petros_project/bw/BACH1_ChIPseq/mm_Bach1_1h_rep1_heme_merged.bw'
    # bw_file2 = '/home/skorsak/Documents/data/Petros_project/bw/RNAPol_ChIPseq/WT-RNAPOLIIS2-1h-heme100-rep1_S5.bw'
    # bw_files = [bw_file1,bw_file2]
    
    sim = LoopSage(region,chrom,bedpe_file,label=label,N_beads=1000)
    Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps,MC_step,burnin,T,T_min,poisson_choice=True,mode='Annealing',viz=True,save=True)
    sim.run_EM('CUDA') # for molecular mechanics or run_MD for molecular dynamics with OpenMM

if __name__=='__main__':
    main()
