import cairo
import imageio
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from matplotlib.pyplot import figure
from matplotlib.pyplot import cm
import scipy.stats
from LoopSage import *

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

def draw_arcplot(ms, ns, N_beads,idx, lw_axis=0.005,axis_h=0.8):
    try:
        os.mkdir(f'plots/arcplots')
    except OSError as error: pass

    with cairo.SVGSurface(f"plots/arcplots/arcplot_{idx}.svg", 800, 400) as surface:
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
        surface.write_to_png(f'plots/arcplots/arcplot_{idx}.png')
        os.remove(f"plots/arcplots/arcplot_{idx}.svg")

def make_gif(N):
    with imageio.get_writer('plots/arc_video.gif', mode='I') as writer:
        for i in range(N):
            image = imageio.imread(f"plots/arcplots/arcplot_{i}.png")
            writer.append_data(image)

    shutil.rmtree(f"plots/arcplots/")

def make_timeplots(Es, Bs, Ks, Fs, burnin):
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
    plt.savefig('timeplot.png',dpi=600)
    plt.show()

def make_moveplots(unbinds, slides):
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
    plt.savefig('moveplot.png',dpi=600)
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

def temperature_T_Ncoh_diagram(T_range,Ncoh_range, f=-500, b=-200, kappa=10000, N_beads=500, file='CTCF_hg38_PeakSupport_2.bedpe'):
    color = iter(cm.rainbow(np.linspace(0, 1, len(Ncoh_range))))
    figure(figsize=(12, 8), dpi=600)
    for N_coh in Ncoh_range:
        Bins, Knots, Folds, UFs = np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range))
        for i, T in enumerate(T_range):
            L, R = binding_from_bedpe_with_peaks(file,N_beads,[48100000,48700000],'chr3',False)
            sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R)
            Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps=2000,MC_step=100,T=T,mode='Metropolis',viz=False)
            Bins[i], Knots[i], Folds[i], UFs[i] = np.average(Bs[10:]), np.average(Ks[10:]), np.average(Fs[10:]), np.average(ufs[10:])

        c = next(color)
        plt.plot(T_range,np.abs(Folds),'-',label=f'Ncoh={N_coh}',c=c)
        plt.plot(T_range,np.abs(UFs),'--',c=c)
    plt.ylabel('Metrics', fontsize=18)
    plt.xlabel('Temperature', fontsize=18)
    # plt.yscale('symlog')
    plt.legend(fontsize=11)
    plt.grid()
    plt.savefig('Ncoh_temp_bif_plot.png',dpi=600)
    plt.show()
    
    return Knots, Bins, Folds

def average_pooling(mat,dim_new):
    im = Image.fromarray(mat)
    size = dim_new,dim_new
    im_resized = np.array(im.resize(size))
    return im_resized

def correlation_plot(given_heatmap,T_range):
    pearsons, spearmans, kendals = np.zeros(len(T_range)), np.zeros(len(T_range)), np.zeros(len(T_range))
    exp_heat_dim = len(given_heatmap)
    for i, T in enumerate(T_range):
        N_beads,N_coh,kappa,f,b = 500,30,10000,-500,-500
        N_steps, MC_step, burnin = int(1e4), int(1e2), 20
        L, R = binding_from_bedpe_with_peaks("/mnt/raid/data/Zofia_Trios/bedpe/hg00731_CTCF_pulled_2.bedpe",N_beads,[178421513,179491193],'chr1',False)
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