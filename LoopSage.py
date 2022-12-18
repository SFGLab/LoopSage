#Basic Libraries
import matplotlib.pyplot as plt
import numpy as np
import random as rd
from tqdm import tqdm

# scipy
from scipy.stats import norm
from scipy.stats import poisson

# my own libraries
from LoopSage_preproc import *
from LoopSage_plots import *

def delta(x1,x2):
    d = 1 if x1==x2 else 0
    return d

def Kappa(mi,ni,mj,nj):
    k=0
    if mi<mj and mj<ni and ni<nj: k+=1 # or np.abs(ni-mj)+1
    if mj<mi and mi<nj and nj<ni: k+=1 # or np.abs(nj-mi)+1
    if mj==ni or mi==nj or ni==nj or mi==mj: k+=1
    return k

class LoopSage:
    def __init__(self,N_beads,N_coh,kappa,f,b,L,R):
        self.N_beads, self.N_coh = N_beads, N_coh
        self.kappa, self.f, self.b = kappa, f, b
        self.L, self.R = L, R
        self.states = np.full(self.N_beads,False)
        
        anchors = np.nonzero(self.L)[0]
        self.avg_loop = int(np.average(np.abs(anchors[1:]-anchors[:-1])))+1
        print('Average loop size:',self.avg_loop)
    
    def E_bind(self,ms,ns):
        binding = 0
        for i in range(self.N_coh):
            binding += self.L[ms[i]]+self.R[ns[i]]
        return self.b*binding

    def E_knot(self,ms,ns):
        knotting = 0
        for i in range(self.N_coh):
            for j in range(i+1,self.N_coh):
                knotting+=Kappa(ms[i],ns[i],ms[j],ns[j])
        return self.kappa*knotting
    
    def E_fold(self,ms,ns,mode='sqr'):
        folding=0
        for i in range(self.N_coh):
            if mode=='abs':
                folding+=np.abs(ns[i]-ms[i])
            elif mode=='sqr':
                folding+=np.abs(ns[i]-ms[i])**2
            elif mode=='sqrr':
                folding+=np.sqrt(np.abs(ns[i]-ms[i]))
            
        return self.f*folding
    
    def get_E(self,ms,ns):
        energy=self.E_bind(ms,ns)+self.E_knot(ms,ns)+self.E_fold(ms,ns)
        return energy

    def get_dE(self,ms,ns,m_new,n_new,idx):
        dE_bind = self.b*(self.L[m_new]+self.R[m_new]-self.L[ms[idx]]-self.R[ns[idx]])
        dE_fold = self.f*((m_new-n_new)**2-(ms[idx]-ns[idx])**2)

        K1, K2 = 0, 0
        for i in range(self.N_coh):
            if i!=idx: K1+=Kappa(ms[idx],ns[idx],ms[i],ns[i])
        for i in range(self.N_coh):
            if i!=idx: K2+=Kappa(m_new,n_new,ms[i],ns[i])
        dE_knot = self.kappa*(K2-K1)

        dE = dE_bind+dE_fold+dE_knot
        return dE
    
    def unbind_bind(self):
        # bind left part of cohesin to a random available place
        m_new = rd.randint(0,self.N_beads-2)
        
        # bind right part of cohesin somewhere close to the left part
        n_new = m_new+1+poisson.rvs(self.avg_loop//4)
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

    def run_energy_minimization(self,N_steps,MC_step,T=1,mode='Metropolis'):
        ms, ns = self.initialize()
        E = self.get_E(ms,ns)
        Es,Ks,Fs,Bs = list(), list(),list(),list()
        Ms, Ns = np.zeros((self.N_coh,N_steps//MC_step)).astype(int), np.zeros((self.N_coh,N_steps//MC_step)).astype(int)

        print('Running simulation...')
        for i in tqdm(range(N_steps)):
            for j in range(self.N_coh):
                # Randomly choose a move (sliding or rebinding)
                r = rd.choice([0,1])
                if r==0:
                    m_new, n_new = self.unbind_bind()
                else:
                    r = rd.choice([0,1])
                    if r==0:
                        m_new, n_new = self.slide_right(ms[j],ns[j])
                    else:
                        m_new, n_new = self.slide_left(ms[j],ns[j])

                # Compute energy difference
                dE = self.get_dE(ms,ns,m_new,n_new,j)

                if mode=='Metropolis':
                    if dE <= 0 or np.exp(-dE/T) > np.random.rand():
                        ms[j], ns[j] = m_new, n_new
                        E = E+dE
                elif mode=='Annealing':
                    T = 1-(i+1)/N_steps
                    if np.exp(-dE/T) > np.random.rand():
                        ms[j], ns[j] = m_new, n_new
                        E = E+dE
                else:
                    raise(Exception(f'Mode `{self.mode}` does not exist. You can choose only between `Metropolis` and `Annealing`.'))
                
                # Save trajectories
                if i%MC_step==0:
                    Ms[j,i//MC_step], Ns[j,i//MC_step] = ms[j], ns[j]

            # Compute Metrics
            if i%MC_step==0:
                Es.append(np.abs(E))
                Ks.append(np.abs(self.E_knot(ms,ns)))
                Fs.append(np.abs(self.E_fold(ms,ns)))
                Bs.append(np.abs(self.E_bind(ms,ns)))
        print('Done! ;D')

        return Es, Ms, Ns, Bs, Ks, Fs

def main():
    L, R = binding_from_bedpe_with_peaks('/mnt/raid/data/Abhishek_data/CTCF_hg38_PeakSupport_probabilistic_motifs.bedpe',1000,[48100000,48700000],'chr3',False)
    N_beads,N_coh,kappa,f,b = 1000,100,10,-1,-1
    sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R)
    Es, Ms, Ns, Bs, Ks, Fs = sim.run_energy_minimization(N_steps=10000,MC_step=100,T=1,mode='Metropolis')
    draw_arcplot(Ms[:,-1],Ns[:,-1],N_beads)
    make_timeplots(Es, Bs, Ks, Fs)

# if __name__=='__main__':
#     main()