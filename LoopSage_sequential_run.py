from LoopSage import *

N_steps, MC_step, burnin, T, T_min = int(1e4), int(2e2), 1000, 3,1
region, chrom = [45770053,46036890], 'chr1'

## Run 1
label=f'Petros_wt0h_sequential'
bedpe_file = '/mnt/raid/data/Petros_project/loops/wt0h_pooled_2.bedpe'
bw_file1 = '/mnt/raid/data/Petros_project/bw/BACH1_ChIPseq/mm_Bach1_0h_rep1_heme_merged.bw'
bw_file2 = '/mnt/raid/data/Petros_project/bw/RNAPol_ChIPseq/WT-RNAPOLIIS2-1h-untreated-rep1_S1.bw'
bw_files = [bw_file1,bw_file2]
sim = LoopSage(region,chrom,bedpe_file,label=label,N_lef=50,N_beads=1000,bw_files=bw_files)
Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps,MC_step,burnin,T,T_min,poisson_choice=True,mode='Metropolis',viz=True,save=True)
sim.run_MD()

## Run 2
label=f'Petros_wt1h_sequential'
bedpe_file = '/mnt/raid/data/Petros_project/loops/wt1h_pooled_2.bedpe'
bw_file1 = '/mnt/raid/data/Petros_project/bw/BACH1_ChIPseq/mm_Bach1_1h_rep1_heme_merged.bw'
bw_file2 = '/mnt/raid/data/Petros_project/bw/RNAPol_ChIPseq/WT-RNAPOLIIS2-1h-heme100-rep1_S5.bw'
bw_files = [bw_file1,bw_file2]
sim = LoopSage(region,chrom,bedpe_file,label=label,N_lef=50,N_beads=1000,bw_files=bw_files)
Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps,MC_step,burnin,T,T_min,poisson_choice=True,mode='Metropolis',viz=True,save=True, m_init=Ms[:,-1], n_init=Ns[:,-1])
sim.run_MD()