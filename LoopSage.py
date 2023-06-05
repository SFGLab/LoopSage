#Basic Libraries
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import scipy.stats as stats
from tqdm import tqdm

# scipy
from scipy.stats import norm
from scipy.stats import poisson

# my own libraries
from LoopSage_preproc import *
from LoopSage_plots import *
from LoopSage_md import *

def delta(x1,x2):
    d = 1 if x1==x2 else 0
    return d

def Kappa(mi,ni,mj,nj):
    k=0
    if mi<mj and mj<ni and ni<nj: k+=1 # np.abs(ni-mj)+1
    if mj<mi and mi<nj and nj<ni: k+=1 # np.abs(nj-mi)+1
    if mj==ni or mi==nj or ni==nj or mi==mj: k+=1
    return k

class LoopSage:
    def __init__(self,N_beads,N_coh,kappa,f,b,L,R,dists,path,track=None):
        self.N_beads, self.N_coh, self.N_CTCF = N_beads, N_coh, np.max([np.count_nonzero(L),np.count_nonzero(R)])
        self.kappa, self.f, self.b = kappa, f, b
        self.L, self.R = L, R
        self.states = np.full(self.N_beads,False)
        self.dists = np.array(dists)
        self.track = track
        self.avg_loop, self.max_loop = int(np.average(self.dists))+1, int(np.max(self.dists))+1
        self.log_avg_loop = np.average(np.log(self.dists+1))
        self.path=path
        print('Average logarithmic loop size',self.log_avg_loop)
        self.params = stats.maxwell.fit(self.dists)
        self.loop_pdist = stats.maxwell.pdf(np.arange(self.N_beads), *self.params)
    
    def E_bind(self,ms,ns):
        binding = 0
        for i in range(self.N_coh):
            binding += self.L[ms[i]]+self.R[ns[i]] #if self.b_mode=='vector' else self.M[ms[i],ns[i]]
        E_b = self.b*binding/(np.sum(self.L)+np.sum(self.R)) #if self.b_mode=='vector' else self.b*binding/np.sum(self.M)
        return E_b

    def E_knot(self,ms,ns):
        knotting = 0
        for i in range(self.N_coh):
            for j in range(i+1,self.N_coh):
                knotting+=Kappa(ms[i],ns[i],ms[j],ns[j])
        return self.kappa*knotting/self.N_coh
    
    def E_fold(self,ms,ns):
        folding=0
        for i in range(self.N_coh):
            folding+=np.log(ns[i]-ms[i])
        return self.f*folding/(self.N_coh*self.log_avg_loop)
    
    def get_E(self,ms,ns):
        energy=self.E_bind(ms,ns)+self.E_knot(ms,ns)+self.E_fold(ms,ns)
        return energy

    def get_dE(self,ms,ns,m_new,n_new,idx):
        dE_bind = self.b*(self.L[m_new]+self.R[n_new]-self.L[ms[idx]]-self.R[ns[idx]])/(np.sum(self.L)+np.sum(self.R)) #if self.b_mode=='vector' else self.b*(self.M[m_new,n_new]-self.M[ms[idx],ns[idx]])/np.sum(self.M)
        dE_fold = self.f*(np.log(n_new-m_new)-np.log(ns[idx]-ms[idx]))/(self.N_coh*np.log(self.avg_loop))
        # dE_dfold = self.Ti*self.f*(self.loop_pdist[n_new-m_new]-self.loop_pdist[ns[idx]-ms[idx]])/(self.N_coh*self.loop_pdist[self.avg_loop])

        K1, K2 = 0, 0
        for i in range(self.N_coh):
            if i!=idx: K1+=Kappa(ms[idx],ns[idx],ms[i],ns[i])
        for i in range(self.N_coh):
            if i!=idx: K2+=Kappa(m_new,n_new,ms[i],ns[i])
        dE_knot = self.kappa*(K2-K1)/self.N_coh

        dE = dE_bind+dE_fold+dE_knot
        return dE

    def unfolding_metric(self,ms,ns):
        fiber = np.zeros(self.N_beads)
        for i in range(self.N_coh):
            fiber[ms[i]:ns[i]]=1
        unfold = self.log_avg_loop*(self.N_beads-np.count_nonzero(fiber))/self.N_beads
        return unfold
    
    def unbind_bind(self):
        # bind left part of cohesin to a random available place
        if np.all(self.track==None):
            m_new = rd.randint(0,self.N_beads-2)
        else:
            m_new = rd.choices(np.arange(self.N_beads-2), weights=self.track[:-2], k=1)[0]
        
        # bind right part of cohesin somewhere close to the left part
        n_new = m_new+1+poisson.rvs(self.avg_loop//8)
        if n_new>=self.N_beads: n_new = rd.randint(m_new+1,self.N_beads-1)
        return int(m_new), int(n_new)

    def slide_right(self,m_old,n_old):
        if n_old+1<self.N_beads:
            m_new, n_new = m_old, n_old+1
        else:
            m_new, n_new = m_old, n_old
        return m_new, n_new

    def slide_left(self,m_old,n_old):
        if m_old-1>0:
            m_new, n_new = m_old-1,n_old
        else:
            m_new, n_new = m_old,n_old
        return m_new, n_new
    
    def initialize(self):
        ms, ns = np.zeros(self.N_coh).astype(int), np.zeros(self.N_coh).astype(int)
        for i in range(self.N_coh):
            ms[i], ns[i] = self.unbind_bind()
        return ms, ns
    
    def run_energy_minimization(self,N_steps,MC_step,burnin,T=1,mode='Metropolis',viz=False,vid=False):
        self.Ti=T
        ms, ns = self.initialize()
        E = self.get_E(ms,ns)
        Es,Ks,Fs,Bs,ufs, slides, unbinds = list(),list(),list(),list(),list(), list(), list()
        Ms, Ns = np.zeros((self.N_coh,N_steps)).astype(int), np.zeros((self.N_coh,N_steps)).astype(int)
        N_slide, N_bind = 0, 0

        # print('Running simulation...')
        for i in tqdm(range(N_steps)):
            for j in range(self.N_coh):
                # Randomly choose a move (sliding or rebinding)
                r = rd.choice([0,1,2])
                if r==0:
                    m_new, n_new = self.unbind_bind()
                    N_bind+=1
                elif r==1:
                    m_new, n_new = self.slide_right(ms[j],ns[j])
                    N_slide+=1
                else:
                    m_new, n_new = self.slide_left(ms[j],ns[j])
                    N_slide+=1

                # Compute energy difference
                self.Ti = (T-(i+1)/N_steps) if mode=='Annealing' else T
                dE = self.get_dE(ms,ns,m_new,n_new,j)

                if dE <= 0 or np.exp(-dE/self.Ti) > np.random.rand():
                    ms[j], ns[j] = m_new, n_new
                    E += dE

                # Save trajectories
                Ms[j,i], Ns[j,i] = ms[j], ns[j]
                if i%MC_step==0:
                    if vid: draw_arcplot(Ms[:,i],Ns[:,i],self.N_beads,i//MC_step,self.path)
            
            # Compute Metrics
            if i%MC_step==0:
                ufs.append(self.unfolding_metric(ms,ns))
                Es.append(E)
                Ks.append(self.E_knot(ms,ns))
                Fs.append(self.E_fold(ms,ns))
                Bs.append(self.E_bind(ms,ns))
                slides.append(N_slide)
                unbinds.append(N_bind)

                N_slide, N_bind = 0, 0
        print('Done! ;D')
        
        # Some vizualizations
        save_info(self.N_beads,self.N_coh,self.N_CTCF,self.kappa,self.f,self.b,self.avg_loop,self.path,N_steps,MC_step,burnin,mode,ufs,Es,Ks,Fs,Bs)
        if viz: make_timeplots(Es, Bs, Ks, Fs, burnin, self.path)
        if viz: make_moveplots(unbinds, slides, self.path)
        if viz: coh_traj_plot(Ms,Ns,self.N_beads, self.path)
        if viz: stochastic_heatmap(Ms,Ns,MC_step,self.N_beads,self.path)
        if vid: make_gif(N_steps//MC_step, self.path)
        
        return Es, Ms, Ns, Bs, Ks, Fs, ufs

def main():
    N_beads,N_coh,kappa,f,b = 1000,50,20000,-1000,-1000
    N_steps, MC_step, burnin, T = int(1e4), int(1e2), 10, 5
    region, chrom = [225286830, 225996745], 'chr1'
    bedpe_file = "/mnt/raid/data/encode/ChIAPET/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe"
    L, R, dists = binding_vectors_from_bedpe_with_peaks(bedpe_file,N_beads,region,chrom,False)
    track = load_track('/mnt/raid/data/encode/ChIP-Seq/ENCSR000DZP_Smc3/ENCFF775OOS_pvalue.bigWig',region,chrom,N_beads,True)
    # M = binding_matrix_from_bedpe("/mnt/raid/data/Trios/bedpe/interactions_maps/hg00731_CTCF_pooled_2.bedpe",N_beads,[178421513,179491193],'chr1',False)
    print('Number of CTCF:',np.max([np.count_nonzero(L),np.count_nonzero(R)]))
    path = make_folder(N_beads,N_coh,region,chrom,label='ChIA_PET_ENCSR184YZV_CTCF')
    sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R,dists,path,track)
    Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps,MC_step,burnin,T,mode='Annealing',viz=True,vid=False)
    np.save(path+'/other/Ms.npy',Ms)
    np.save(path+'/other/Ns.npy',Ns)
    np.save(path+'/other/Fs.npy',Fs)
    np.save(path+'/other/Bs.npy',Bs)
    # md = MD_LE(Ms,Ns,N_beads,burnin,MC_step,path)
    # sim_heat = md.run_pipeline(write_files=True,plots=True)
    # corr_exp_heat(sim_heat,bedpe_file,region,chrom,N_beads,path)

if __name__=='__main__':
    main()
