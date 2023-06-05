import cairo
import imageio
import os
import shutil
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from PIL import Image
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from matplotlib.pyplot import figure
from matplotlib.pyplot import cm
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats
from LoopSage import *
from tqdm import tqdm

try:
    os.mkdir(f'plots')
except OSError as error: pass

def draw_contact(c, y, start, end, lw=0.005, h=0.4):
    c.set_source_rgba(0.0, 1.0, 0.0, 0.5)
    c.set_line_width(lw)
    c.move_to(start, y)
    length = end-start
    h = y-length*4/3
    c.curve_to(start+length/3, h, start+2*length/3, h, end, y)
    c.stroke()
#     c.arc_negative((start+end)/2, 0.9, length/2, 0, math.pi)
#     c.stroke()

def draw_arcplot(ms, ns, N_beads,idx, lw_axis=0.005,axis_h=0.8,path=None):
    save_path = path+"/plots/arcplots/" if path!=None else "/plots/arcplots/"
    try:
        os.mkdir(save_path)
    except OSError as error: pass

    with cairo.SVGSurface(save_path+f"arcplot_{idx}.svg", 800, 400) as surface:
        c = cairo.Context(surface)
        c.scale(800, 400)
        c.rectangle(0, 0, 800, 400)
        c.set_source_rgb(1, 1, 1)
        c.fill()

        for i in range(len(ms)):
            start = ms[i]
            end = ns[i]
            start = 0.1 + 0.8*start/N_beads
            end = 0.1 + 0.8*end/N_beads
            draw_contact(c=c, y=axis_h, start=start, end=end,
                         lw=0.002, h=0.4)

        # axis
        c.set_source_rgb(0.0, 0.0, 0.0)
        c.set_line_width(lw_axis)
        c.move_to(0.1, axis_h)
        c.line_to(0.9, axis_h)
        c.stroke()
        c.set_line_width(lw_axis/2)
        c.move_to(0.1, axis_h+0.01)
        c.line_to(0.1, axis_h-0.01)
        c.stroke()
        c.move_to(0.9, axis_h+0.01)
        c.line_to(0.9, axis_h-0.01)
        c.stroke()

        # Save as a SVG and PNG
        surface.write_to_png(save_path+f'arcplot_{idx}.png')
        os.remove(save_path+f"arcplot_{idx}.svg")

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
    plt.xlabel('Monte Carlo Step', fontsize=16)
    # plt.yscale('symlog')
    plt.legend(['Total Energy', 'Binding', 'Knotting', 'Folding'], fontsize=16)
    plt.grid()
    save_path = path+'/plots/energies.png' if path!=None else 'energies.png'
    plt.savefig(save_path,dpi=200)
    plt.show()

    # Autocorrelation plot
    plot_acf(Fs, title=None, lags=len(Fs)//2)
    plt.ylabel("Autocorrelations", fontsize=16)
    plt.xlabel("Lags", fontsize=16)
    plt.grid()
    save_path = path+'/plots/autoc.png' if path!=None else 'autoc.png'
    plt.savefig(save_path,dpi=200)
    plt.show()

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
    save_path = path+'/plots/moveplot.png' if path!=None else 'moveplot.png'
    plt.savefig(save_path,dpi=600)
    plt.show()

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
    plt.show()

    return Knots, Bins, Folds

def temperature_T_Ncoh_diagram(T_range, Ncoh_range=np.array([10,25,50,70]), f=-100, b=-100, kappa=20000, N_beads=500, file='/mnt/raid/data/Trios/bedpe/hiccups_loops_sqrtVC_norm/hg00731_ctcf_vc_sqrt_merged_loops_edited_2.bedpe'):
    colors = ['red','green','purple','cyan']
    figure(figsize=(12, 8), dpi=600)
    for j, N_coh in enumerate(Ncoh_range):
        Bins, Knots, Folds, UFs = np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range))
        for i, T in enumerate(T_range):
            L, R, dists = binding_vectors_from_bedpe_with_peaks(file,N_beads,[40000000,45000000],'chr3',False)
            N_CTCF = (np.count_nonzero(L)+np.count_nonzero(R))/2
            print('Number of CTCF:',N_CTCF)
            sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R,dists)
            Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps=2000,MC_step=10,T=T,burnin=100,mode='Metropolis',viz=False)
            Bins[i], Knots[i], Folds[i], UFs[i] = np.average(Bs[100:]), np.average(Ks[100:]), np.average(Fs[100:]), np.average(ufs[100:])
        
        c = colors[j]
        N_CTCF = (np.count_nonzero(L)+np.count_nonzero(R))/2
        plt.plot(T_range,np.abs(Folds),'-',label=f'Ncoh={N_coh}',c=c)
        plt.plot(T_range,np.abs(UFs),'--',c=c)
    print('Number of CTCF:',N_CTCF)
    plt.ylabel('Metrics', fontsize=18)
    plt.xlabel('Temperature', fontsize=18)
    # plt.yscale('symlog')
    plt.legend(fontsize=16)
    plt.grid()
    plt.savefig(f'Ncoh_temp_bif_plot_f{int(np.abs(f))}_b{int(np.abs(b))}.png',dpi=600)
    plt.show()
    
    return Knots, Bins, Folds

def Nbeads_diagram(Nbs, N_coh=20, T=1, f=-100, b=-100, kappa=20000, file='/mnt/raid/data/Trios/bedpe/hiccups_loops_sqrtVC_norm/hg00731_ctcf_vc_sqrt_merged_loops_edited_2.bedpe'):
    Folds, UFs = np.zeros(len(Nbs)), np.zeros(len(Nbs))
    figure(figsize=(12, 8), dpi=600)
    for i, N in enumerate(Nbs):
        L, R, dists = binding_vectors_from_bedpe_with_peaks(file,N,[40000000,45000000],'chr3',False)
        sim = LoopSage(N,N_coh,kappa,f,b,L,R,dists)
        Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps=2000,MC_step=10,T=T,burnin=100,mode='Metropolis',viz=False)
        Folds[i], UFs[i] = np.average(Fs[100:]), np.average(ufs[100:])

    plt.plot(Nbs,np.abs(Folds),'k-')
    plt.plot(Nbs,np.abs(UFs),'k--')
    plt.xlabel(r'$N_{beads}$', fontsize=18)
    plt.ylabel('Metrics', fontsize=18)
    plt.legend(['Folding','Proportion of Gaps'],fontsize=16)
    # plt.yscale('symlog')
    plt.grid()
    plt.savefig('Nbeads_plot.png',dpi=600)
    plt.show()

def fb_heatmap(fs,bs,T,N_beads=500,N_coh=20,kappa=200000,file='/mnt/raid/data/Trios/bedpe/hiccups_loops_sqrtVC_norm/hg00731_ctcf_vc_sqrt_merged_loops_edited_2.bedpe'):
    fold_mat  = np.zeros([len(fs),len(bs)])
    ufold_mat = np.zeros([len(fs),len(bs)])
    for i,f in enumerate(fs):
        for j,b in enumerate(bs):
            L, R, dists = binding_vectors_from_bedpe_with_peaks(file,N_beads,[48100000,58700000],'chr3',False)
            sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R,dists)
            Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps=2000,MC_step=10,T=T,burnin=100,mode='Metropolis',viz=False)
            fold_mat[i,j] = np.average(Fs[100:])
            ufold_mat[i,j] = np.average(ufs[100:])

    figure(figsize=(12, 12), dpi=600)
    plt.contourf(fs, bs, fold_mat,cmap='gnuplot',vmax=2)
    plt.xlabel('b',fontsize=16)
    plt.ylabel('f',fontsize=16)
    plt.colorbar()
    plt.savefig(f'fold_heat_T{T}.png',dpi=600)
    plt.show()

    figure(figsize=(12, 12), dpi=600)
    plt.contourf(fs, bs, ufold_mat,cmap='gnuplot',vmax=1.5)
    plt.xlabel('b',fontsize=16)
    plt.ylabel('f',fontsize=16)
    plt.colorbar()
    plt.savefig(f'ufold_heat_T{T}.png',dpi=600)
    plt.show()

def average_pooling(mat,dim_new):
    im = Image.fromarray(mat)
    size = dim_new,dim_new
    im_resized = np.array(im.resize(size))
    return im_resized

def correlation_plot(given_heatmap,T_range):
    pearsons, spearmans, kendals = np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range))
    exp_heat_dim = len(given_heatmap)
    for i, T in enumerate(T_range):
        N_beads,N_coh,kappa,f,b = 500,30,20000,-2000,-2000
        N_steps, MC_step, burnin = int(1e4), int(1e2), 20
        L, R = binding_vectors_from_bedpe_with_peaks("/mnt/raid/data/Zofia_Trios/bedpe/hg00731_CTCF_pulled_2.bedpe",N_beads,[178421513,179491193],'chr1',False)
        sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R)
        Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps,MC_step,burnin,T,mode='Metropolis',viz=False,vid=False)
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
    plt.savefig('pearson_plot.png',dpi=600)
    plt.show()

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
    save_path = path+'/plots/coh_trajectories.pdf' if path!=None else 'coh_trajectories.pdf'
    plt.savefig(save_path, format='pdf', dpi=600)
    plt.show()

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
    plt.colorbar()
    plt.show()