import imageio
import os
import shutil
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from matplotlib.pyplot import figure
from matplotlib.pyplot import cm
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats
from LoopSage import *
from tqdm import tqdm
from scipy import stats

def make_loop_hist(Ms,Ns,path=None):
    Ls = np.abs(Ns-Ms).flatten()
    Ls_df = pd.DataFrame(Ls)
    figure(figsize=(10, 7), dpi=600)
    sns.histplot(data=Ls_df, bins=30,  kde=True,stat='density')
    plt.grid()
    plt.legend()
    plt.ylabel('Probability',fontsize=16)
    plt.xlabel('Loop Length',fontsize=16)
    if path!=None:
        save_path = path+'/plots/loop_length.png'
        plt.savefig(save_path,format='png',dpi=200)
        save_path = path+'/plots/loop_length.svg'
        plt.savefig(save_path,format='svg',dpi=600)
        save_path = path+'/plots/loop_length.pdf'
        plt.savefig(save_path,format='pdf',dpi=600)
    plt.close()

    Is, Js = Ms.flatten(), Ns.flatten()
    IJ_df = pd.DataFrame()
    IJ_df['mi'] = Is
    IJ_df['nj'] = Js
    figure(figsize=(8, 8), dpi=600)
    sns.jointplot(IJ_df, x="mi", y="nj",kind='hex',color='Red')
    if path!=None:
        save_path = path+'/plots/ij_prob.png'
        plt.savefig(save_path,format='png',dpi=200)
        save_path = path+'/plots/ij_prob.svg'
        plt.savefig(save_path,format='svg',dpi=600)
        save_path = path+'/plots/ij_prob.pdf'
        plt.savefig(save_path,format='pdf',dpi=600)
    plt.close()

    m_idx, m_counts = np.unique(Ms, return_counts=True)
    n_idx, n_counts = np.unique(Ns, return_counts=True)
    Nm, Nn = np.sum(m_counts), np.sum(n_counts)
    N_beads = np.max([np.max(m_idx),np.max(n_idx)])+1
    m_probs, n_probs = np.zeros(N_beads), np.zeros(N_beads)
    m_probs[m_idx], n_probs[n_idx] = m_counts/Nm, n_counts/Nn
    prob_mat = np.outer(m_probs,n_probs)
    m_non, n_non = np.nonzero(prob_mat)
    nm_diff, nm_count_probs = np.zeros(len(m_non)), np.zeros(len(m_non))
    for i in range(len(m_non)):
        nm_count_probs[i] = prob_mat[m_non[i], n_non[i]]
        nm_diff[i] = np.abs(m_non[i] - n_non[i])
    count_df = pd.DataFrame()
    count_df['Length'] = nm_diff
    count_df['Count'] = nm_count_probs*Nm
    sns.displot(data=count_df,x='Length',y='Count',color="blue",cbar=True)
    plt.xscale('log')
    plt.yscale('log')
    if path!=None:
        save_path = path+'/plots/count_length.png'
        plt.savefig(save_path,format='png',dpi=200)
    plt.close()

def make_gif(N,path=None):
    with imageio.get_writer('plots/arc_video.gif', mode='I') as writer:
        for i in range(N):
            image = imageio.imread(f"plots/arcplots/arcplot_{i}.png")
            writer.append_data(image)
    save_path = path+"/plots/arcplots/" if path!=None else "/plots/arcplots/"
    shutil.rmtree(save_path)

def make_timeplots(Es, Bs, Ks, Fs, burnin, path=None):
    try:
        os.mkdir('plots')
    except OSError as error: pass
    figure(figsize=(10, 8), dpi=600)
    plt.plot(Es, 'k')
    plt.plot(Bs, 'cyan')
    plt.plot(Ks, 'green')
    plt.plot(Fs, 'red')
    plt.axvline(x=burnin, color='blue')
    plt.ylabel('Metrics', fontsize=16)
    plt.ylim((np.min(Es)-10,-np.min(Es)))
    plt.xlabel('Monte Carlo Step', fontsize=16)
    # plt.yscale('symlog')
    plt.legend(['Total Energy', 'Binding', 'Knotting', 'Folding'], fontsize=16)
    plt.grid()

    if path!=None:
        save_path = path+'/plots/energies.png'
        plt.savefig(save_path,format='png',dpi=200)
        save_path = path+'/plots/energies.svg'
        plt.savefig(save_path,format='svg',dpi=200)
        save_path = path+'/plots/energies.pdf'
        plt.savefig(save_path,format='pdf',dpi=600)
    plt.close()

    # Autocorrelation plot
    plot_acf(Fs, title=None, lags=len(Fs)//2)
    plt.ylabel("Autocorrelations", fontsize=16)
    plt.xlabel("Lags", fontsize=16)
    plt.grid()
    if path!=None: 
        save_path = path+'/plots/autoc.png'
        plt.savefig(save_path,dpi=200)
        save_path = path+'/plots/autoc.svg'
        plt.savefig(save_path,format='svg',dpi=200)
        save_path = path+'/plots/autoc.pdf'
        plt.savefig(save_path,format='pdf',dpi=200)
    plt.close()

def make_moveplots(unbinds, slides, path=None):
    try:
        os.mkdir('plots')
    except OSError as error: pass
    figure(figsize=(10, 8), dpi=600)
    plt.plot(unbinds, 'blue')
    plt.plot(slides, 'red')
    plt.ylabel('Number of moves', fontsize=16)
    plt.xlabel('Monte Carlo Step', fontsize=16)
    # plt.yscale('symlog')
    plt.legend(['Rebinding', 'Sliding'], fontsize=16)
    plt.grid()
    if path!=None:
        save_path = path+'/plots/moveplot.png'
        plt.savefig(save_path,dpi=600)
        save_path = path+'/plots/moveplot.pdf'
        plt.savefig(save_path,dpi=600)
    plt.close()

def temperature_biff_diagram(T_range, f=-500, b=-200,N_beads=500,N_coh=50, kappa=10000, file='CTCF_hg38_PeakSupport_2.bedpe'):
    Bins, Knots, Folds, UFs = np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range))
    for i, T in enumerate(tqdm(T_range)):
        L, R = binding_from_bedpe_with_peaks(file,N_beads,[48100000,48700000],'chr3',False)
        sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R)
        Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps=2000,MC_step=20,T=T,mode='Metropolis',viz=False)
        Bins[i], Knots[i], Folds[i], UFs[i] = np.average(Bs[10:]), np.average(Ks[10:]), np.average(Fs[10:]), np.average(ufs[10:])
    
    figure(figsize=(10, 8), dpi=600)
    plt.plot(T_range,np.abs(Bins),'ro-')
    # plt.plot(T_range,Knots,'go-')
    plt.plot(T_range,np.abs(Folds),'bo-')
    plt.plot(T_range,np.abs(UFs),'go-')
    plt.ylabel('Metrics', fontsize=18)
    plt.xlabel('Temperature', fontsize=18)
    # plt.yscale('symlog')
    plt.legend(['Binding','Folding', 'Unfolding'], fontsize=16)
    plt.grid()
    plt.savefig('temp_bif_plot.png',dpi=600)
    plt.savefig('temp_bif_plot.pdf',dpi=600)
    plt.close()

    return Knots, Bins, Folds

def temperature_T_Ncoh_diagram(T_range, Ncoh_range=np.array([10,25,50,100]), f=-1000, b=-1000, kappa=100000, N_beads=1000, file='/mnt/raid/data/encode/ChIAPET/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe'):
    colors = ['red','green','magenta','blue']
    figure(figsize=(10, 6), dpi=600)
    for j, N_coh in tqdm(enumerate(Ncoh_range)):
        save_path = make_folder(N_beads,N_coh,[178421513, 179491193],'chr1',label='biff_diags')
        Bins, Knots, Folds, UFs = np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range))
        errFolds, errUFs = np.zeros(len(T_range)), np.zeros(len(T_range))
        for i, T in enumerate(T_range):
            L, R, dists = binding_vectors_from_bedpe_with_peaks(file,N_beads,[178421513, 179491193],'chr1',False)
            sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R,dists,save_path)
            Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps=10000,MC_step=100,T=T,burnin=10,mode='Metropolis',viz=False)
            Bins[i], Knots[i], Folds[i], UFs[i] = np.average(Bs[10:]), np.average(Ks[10:]), np.average(Fs[10:])/f, np.average(ufs[10:])
            errFolds[i], errUFs[i] = np.abs(np.std(Fs[10:])/f), np.std(ufs[10:])
        c = colors[j]
        N_CTCF = (np.count_nonzero(L)+np.count_nonzero(R))/2
        plt.errorbar(T_range,np.abs(Folds),yerr=errFolds,fmt='o-',label=f'Ncoh={N_coh}',color=c)
        plt.errorbar(T_range,np.abs(UFs),yerr=errUFs,fmt='o--',color=c)
    print('Number of CTCF:',N_CTCF)
    plt.ylabel('Metrics', fontsize=16)
    plt.xlabel('Temperature', fontsize=16)
    # plt.yscale('symlog')
    plt.legend(fontsize=13)
    # plt.grid()
    plt.savefig(f'Ncoh_temp_bif_plot_f{int(np.abs(f))}_b{int(np.abs(b))}.pdf',format='pdf',dpi=600)
    plt.savefig(f'Ncoh_temp_bif_plot_f{int(np.abs(f))}_b{int(np.abs(b))}.png',format='png',dpi=600)
    plt.savefig(f'Ncoh_temp_bif_plot_f{int(np.abs(f))}_b{int(np.abs(b))}.svg',format='svg',dpi=600)
    plt.close()
    
    return Knots, Bins, Folds, UFs

def temperature_loop_distplot(T_range, N_coh=50, f=-1000, b=-1000, kappa=100000, N_beads=1000, file='/mnt/raid/data/encode/ChIAPET/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe'):
    colors = ['red','green','magenta','blue']
    df = pd.DataFrame()

    for i, T in enumerate(T_range):
        L, R, dists = binding_vectors_from_bedpe_with_peaks(file,N_beads,[178421513, 179491193],'chr1',False)
        sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R,dists,None)
        Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps=10000,MC_step=100,T=T,burnin=10,mode='Metropolis',viz=False)
        df[f'T={T}'] = np.abs(Ns-Ms).flatten()

    figure(figsize=(8, 6), dpi=600)
    sns.histplot(data=df,bins=10,kde=True, element="step")
    plt.ylabel('Probability',fontsize=13)
    plt.xlabel('Loop Length',fontsize=13)
    plt.savefig('temp_loop_length.png',format='png',dpi=600)
    plt.savefig('temp_loop_length.svg',format='svg',dpi=600)
    plt.savefig('temp_loop_length.pdf',format='pdf',dpi=600)
    plt.close()

def fb_loop_distplot(fbs, N_coh=50, T=5, kappa=100000, N_beads=1000, file='/mnt/raid/data/encode/ChIAPET/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe'):
    colors = ['red','green','magenta','blue']
    df = pd.DataFrame()

    for fb in fbs:
        L, R, dists = binding_vectors_from_bedpe_with_peaks(file,N_beads,[178421513, 179491193],'chr1',False)
        sim = LoopSage(N_beads,N_coh,kappa,fb,fb,L,R,dists,None)
        Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps=10000,MC_step=100,T=T,burnin=10,mode='Metropolis',viz=False)
        df[f'f=b={fb}'] = np.abs(Ns-Ms).flatten()

    figure(figsize=(8, 6), dpi=600)
    sns.histplot(data=df,bins=10,kde=True, element="step")
    plt.ylabel('Probability',fontsize=13)
    plt.xlabel('Loop Length',fontsize=13)
    plt.savefig('fb_loop_length.png',format='png',dpi=600)
    plt.savefig('fb_loop_length.svg',format='svg',dpi=600)
    plt.savefig('fb_loop_length.pdf',format='pdf',dpi=600)
    plt.close()

def knot_T_diagram(T_range, kappas, N_coh=10, f=-1000, b=-1000, N_beads=1000, file='/mnt/raid/data/encode/ChIAPET/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe'):
    colors = ['red','green','magenta','blue']
    figure(figsize=(10, 6), dpi=600)
    for j, kappa in tqdm(enumerate(kappas)):
        save_path = make_folder(N_beads,N_coh,[178421513, 179491193],'chr1',label='biff_diags')
        Bins, Knots, Folds, UFs, Kappas = np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range))
        errFolds, errUFs, errKs = np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range))
        for i, T in enumerate(T_range):
            L, R, dists = binding_vectors_from_bedpe_with_peaks(file,N_beads,[178421513, 179491193],'chr1',False)
            sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R,dists,save_path)
            Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps=10000,MC_step=100,T=T,burnin=10,mode='Metropolis',viz=False)
            Bins[i], Knots[i], Folds[i], UFs[i], Kappas[i] = np.average(Bs[10:]), np.average(Ks[10:]), np.average(Fs[10:])/f, np.average(ufs[10:]), np.average(Ks[10:])/kappa
            errFolds[i], errUFs[i], errKs[i] = np.abs(np.std(Fs[10:])/f), np.std(ufs[10:]), np.std(Ks[10:])/kappa
        c = colors[j]
        
        plt.errorbar(T_range,np.abs(Folds),yerr=errFolds,marker='o',ls='solid',label=rf'$\kappa$={kappa}',color=c)
        plt.errorbar(T_range,np.abs(UFs),marker='>',ls='dashed',yerr=errUFs,color=c)
        plt.errorbar(T_range,np.abs(Kappas),marker='x',ls='dotted',yerr=errKs,color=c)
    N_CTCF = (np.count_nonzero(L)+np.count_nonzero(R))/2
    print('Number of CTCF:',N_CTCF)
    plt.ylabel('Metrics', fontsize=16)
    plt.xlabel('Temperature', fontsize=16)
    plt.legend(fontsize=13)
    plt.savefig(f'kappa_bif_plot_f{int(np.abs(f))}_b{int(np.abs(b))}_Ncoh_{N_coh}.pdf',format='pdf',dpi=600)
    plt.savefig(f'kappa_bif_plot_f{int(np.abs(f))}_b{int(np.abs(b))}_Ncoh_{N_coh}.png',format='png',dpi=600)
    plt.savefig(f'kappa_bif_plot_f{int(np.abs(f))}_b{int(np.abs(b))}_Ncoh_{N_coh}.svg',format='svg',dpi=600)
    plt.close()

def Nbeads_diagram(Nbs, N_coh=50, T=5, f=-1000, b=-1000, kappa=100000, file='/mnt/raid/data/encode/ChIAPET/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe'):
    Folds, UFs = np.zeros(len(Nbs)), np.zeros(len(Nbs))
    errFolds, errUFs = np.zeros(len(Nbs)), np.zeros(len(Nbs))
    figure(figsize=(10, 6), dpi=600)
    for i, N in enumerate(Nbs):
        save_path = make_folder(N,N_coh,[178421513, 179491193],'chr1',label='biff_diags')
        L, R, dists = binding_vectors_from_bedpe_with_peaks(file,int(N),[178421513, 179491193],'chr1',False)
        sim = LoopSage(N,N_coh,kappa,f,b,L,R,dists,save_path)
        Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps=10000,MC_step=100,T=T,burnin=10,mode='Metropolis',viz=False)
        Folds[i], UFs[i] = np.average(Fs[10:])/f, np.average(ufs[10:])
        errFolds[i], errUFs[i] = np.abs(np.std(Fs[10:])/f), np.std(ufs[10:])

    plt.errorbar(Nbs,np.abs(Folds),yerr=errFolds,fmt='o-',color='black')
    plt.errorbar(Nbs,np.abs(UFs),yerr=errUFs,fmt='o--',color='black')
    plt.xlabel(r'$N_{beads}$', fontsize=16)
    plt.ylabel('Metrics', fontsize=16)
    plt.legend(['Folding','Proportion of Gaps'],fontsize=16)
    plt.savefig('Nbeads_plot.pdf',format='pdf',dpi=600)
    plt.savefig('Nbeads_plot.png',format='png',dpi=600)
    plt.savefig('Nbeads_plot.svg',format='svg',dpi=600)
    plt.close()

def fb_heatmap(fs,bs,T,N_beads=500,N_coh=20,kappa=200000,file='/mnt/raid/data/Trios/bedpe/hiccups_loops_sqrtVC_norm/hg00731_ctcf_vc_sqrt_merged_loops_edited_2.bedpe'):
    fold_mat  = np.zeros([len(fs),len(bs)])
    ufold_mat = np.zeros([len(fs),len(bs)])
    for i,f in enumerate(fs):
        for j,b in enumerate(bs):
            L, R, dists = binding_vectors_from_bedpe_with_peaks(file,N_beads,[48100000,58700000],'chr3',False)
            sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R,dists)
            Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps=2000,MC_step=10,T=T,burnin=100,mode='Metropolis',viz=True)
            fold_mat[i,j] = np.average(Fs[100:])
            ufold_mat[i,j] = np.average(ufs[100:])

    figure(figsize=(12, 12), dpi=600)
    plt.contourf(fs, bs, fold_mat,cmap='gnuplot',vmax=2)
    plt.xlabel('b',fontsize=16)
    plt.ylabel('f',fontsize=16)
    plt.colorbar()
    plt.savefig(f'fold_heat_T{T}.pdf',format='pdf',dpi=600)
    plt.close()

    figure(figsize=(12, 12), dpi=600)
    plt.contourf(fs, bs, ufold_mat,cmap='gnuplot',vmax=1.5)
    plt.xlabel('b',fontsize=16)
    plt.ylabel('f',fontsize=16)
    plt.colorbar()
    plt.savefig(f'ufold_heat_T{T}.pdf',format='pdf',dpi=600)
    plt.close()

def average_pooling(mat,dim_new):
    im = Image.fromarray(mat)
    size = dim_new,dim_new
    im_resized = np.array(im.resize(size))
    return im_resized

def correlation_plot(given_heatmap,T_range,path):
    pearsons, spearmans, kendals = np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range))
    exp_heat_dim = len(given_heatmap)
    for i, T in enumerate(T_range):
        N_beads,N_coh,kappa,f,b = 500,30,20000,-2000,-2000
        N_steps, MC_step, burnin = int(1e4), int(1e2), 20
        L, R = binding_vectors_from_bedpe_with_peaks("/mnt/raid/data/Zofia_Trios/bedpe/hg00731_CTCF_pulled_2.bedpe",N_beads,[178421513,179491193],'chr1',False)
        sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R)
        Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps,MC_step,burnin,T,mode='Metropolis',viz=True,vid=False)
        md = MD_LE(Ms,Ns,N_beads,burnin,MC_step)
        heat = md.run_pipeline(write_files=False,plots=False)
        if N_beads>exp_heat_dim:
            heat = average_pooling(heat,exp_heat_dim)
            L = exp_heat_dim
        else:
            given_heatmap = average_pooling(given_heatmap,N_beads)
            L = N_beads
        a, b = np.reshape(heat, (L**2, )), np.reshape(given_heatmap, (L**2, ))
        pearsons[i] = scipy.stats.pearsonr(a,b)[0]
        spearmans[i] = scipy.stats.spearmanr(a, b).correlation
        kendals[i] = scipy.stats.kendalltau(a, b).correlation
        print(f'\nTemperature:{T}, Pearson Correlation coefficient:{pearsons[i]}, Spearman:{spearmans[i]}, Kendal:{kendals[i]}\n\n')

    figure(figsize=(10, 8), dpi=600)
    plt.plot(T_range,pearsons,'bo-')
    plt.plot(T_range,spearmans,'ro-')
    plt.plot(T_range,kendals,'go-')
    # plt.plot(T_range,Knots,'go-')
    plt.ylabel('Correlation with Experimental Heatmap', fontsize=16)
    plt.xlabel('Temperature', fontsize=16)
    # plt.yscale('symlog')
    plt.legend(['Pearson','Spearman','Kendall Tau'])
    plt.grid()
    save_path = path+'/plots/pearson_plot.pdf' if path!=None else 'pearson_plot.pdf'
    plt.savefig(save_path,dpi=600)
    plt.close()

def coh_traj_plot(ms,ns,N_beads,path):
    N_coh = len(ms)
    figure(figsize=(18, 12))
    color = ["#"+''.join([rd.choice('0123456789ABCDEF') for j in range(6)]) for i in range(N_coh)]
    size = 0.01 if (N_beads > 500 or N_coh > 20) else 0.1
    
    ls = 'None'
    for nn in range(N_coh):
        tr_m, tr_n = ms[nn], ns[nn]
        plt.fill_between(np.arange(len(tr_m)), tr_m, tr_n, color=color[nn], alpha=0.4, interpolate=False, linewidth=0)
    plt.xlabel('Simulation Step', fontsize=16)
    plt.ylabel('Position of Cohesin', fontsize=16)
    plt.gca().invert_yaxis()
    save_path = path+'/plots/coh_trajectories.png' if path!=None else 'coh_trajectories.png'
    plt.savefig(save_path, format='png', dpi=200)
    save_path = path+'/plots/coh_trajectories.svg' if path!=None else 'coh_trajectories.svg'
    plt.savefig(save_path, format='svg', dpi=600)
    save_path = path+'/plots/coh_trajectories.pdf' if path!=None else 'coh_trajectories.pdf'
    plt.savefig(save_path, format='pdf', dpi=600)
    plt.close()

def stochastic_heatmap(ms,ns,step,L,path,comm_prop=True,fill_square=True):
    N_coh, N_steps = ms.shape
    mats = list()
    for t in range(0,N_steps,step):
        # add a loop where there is a cohesin
        mat = np.zeros((L,L))
        for m, n in zip(ms[:,t],ns[:,t]):
            mat[m,n] = 1
            mat[n,m] = 1
        
        # if a->b and b->c then a->c
        if comm_prop:
            for iter in range(3):
                xs, ys = np.nonzero(mat)
                for i, n in enumerate(ys):
                    if len(np.where(xs==(n+1))[0])>0:
                        j = np.where(xs==(n+1))[0]
                        mat[xs[i],ys[j]] = 2*iter+1
                        mat[ys[j],xs[i]] = 2*iter+1

        # feel the square that it is formed by each loop (m,n)
        if fill_square:
            xs, ys = np.nonzero(mat)
            for x, y in zip(xs,ys):
                if y>x: mat[x:y,x:y] += 0.01*mat[x,y]

        mats.append(mat)
    avg_mat = np.average(mats,axis=0)
    figure(figsize=(10, 10))
    plt.imshow(avg_mat,cmap="Reds",vmax=np.average(avg_mat)+3*np.std(avg_mat))
    save_path = path+f'/plots/stochastic_heatmap.svg' if path!=None else 'stochastic_heatmap.svg'
    plt.savefig(save_path,format='svg',dpi=500)
    save_path = path+f'/plots/stochastic_heatmap.pdf' if path!=None else 'stochastic_heatmap.pdf'
    plt.savefig(save_path,format='pdf',dpi=500)
    # plt.colorbar()
    plt.close()