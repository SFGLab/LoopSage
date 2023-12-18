# LoopSage
An energy-based model for loop extrusion.

## The model

Let's assume that each cohesin $i$ can be represented of two coordinates $(m_{i},n_{i})$ we allow three moves in our simulation:

* Slide right ( $n_{i} -> n_{i+1}$ to the right).
* Slide left ( $m_{i} -> m_{i-1}$ to the left).
* Rebind somewhere else.


The main idea of the algorithm is to ensemble loop extrusion from a Boltzmann probability distribution, with Hamiltonian,

$$E = \dfrac{f}{N_{fold}}\sum_{i=1}^{N_{coh}}\log(n_i-m_i)+\dfrac{\kappa}{N_{cross}}\sum_{i,j}K(m_i,n_i;m_j,n_j)+\dfrac{b}{N_{bind}}\sum_{i=1}^{N_{coh}}\left(L(m_i)+R(n_i)\right)$$

The first term corresponds to the folding of chromatin, and the second term is a penalty for the appearance of crosss. Therefore, we have the function,
$K(m_{i},n_{i};m_{j},n_{j})$ which takes the value 1 when $m_{i} < m_{j} < n_{i} < n_{j}$ or $m_{i}=m_{j}$ or $m_{i}=n_{j}$.

These $L(\cdot), R(\cdot)$ functions are two functions that define the binding potential and they are orientation specific - so they are different for left and right position of cohesin (because CTCF motifs are orientation specific), therefore when we have a gap in these functions, it means presence of CTCF. These two functions are derived from data with CTCF binning and by running the script for probabilistic orientation. Moreover, by $N_{(\cdot)}$ we symbolize the normalization constants for each factor,

$$N_{fold}=N_{coh}\cdot \langle n_i-m_i\rangle,\quad N_{cross}=N_{coh},\quad N_{bind}=\sum_{k}\left(L(k)+R(k)\right).$$

Therefore, we define the folding, crossing and binding energy, which are also metrics that help us to understand the dynamics of our system,

$$E_{fold} = \dfrac{f}{N_{fold}}\sum_{i=1}^{N_{coh}}\log(n_i-m_i),$$

and

$$E_{cross} = \dfrac{\kappa}{N_{cross}}\sum_{i,j}K(m_i,n_i;m_j,n_j),$$

and

$$E_{bind} = \dfrac{b}{N_{bind}}\sum_{i=1}^{N_{coh}}\left(L(m_i)+R(n_i)\right).$$

An additional term which allows other protein factors that may act as barriers for the motion of LEFs can me optionally added,

$$E_{bw}=\sum_{i=1}^{N_{\text{bw}}}\frac{r_{i}}{N_{bw,i}}\sum_{j=1}^{N_{lef}}\left(W_i(m_j)+W_i(n_j)\right),\qquad N_{bw,i} = \sum_{k}W_i(k)$$

where $N_{bw}$ is the number of \texttt{.BigWig} ChIP-Seq experiments that are imported for each one of the proteins of interest. The energy term $W_{i}(\cdot)$ it corresponds to the average values of the ChIP-Seq experiment $i$ for each loci of interest. Finally, $r_{i}$ it is a weight that corresponds each one of these experiments and by default it is set as $r_{i}=b/2$.

Thus,

$$\Delta E = \Delta E_{fold}+\Delta E_{cross}+\Delta E_{bind} + \Delta E_{bw}.$$

In this manner we accept a move in two cases:

* If $\Delta E<0$ or,
* if $\Delta E>0$ with probability $e^{-\Delta E/kT}$.

And of course, the result - the distribution of loops in equilibrium -  depends on temperature of Boltzmann distribution $T$.

## Packages to install
The following dependecies must be installed in the specific desktop environment that you would like to run the simulation pipeline,



## How to use?

The implementation of the code is very easy and it can be described in the following lines,

```python
N_steps, MC_step, burnin, T, T_min = int(1e4), int(2e2), 1000, 5,1
region, chrom = [88271457,88851999], 'chr10'
label=f'Petros_wt1h'
bedpe_file = '/mnt/raid/data/Petros_project/loops/wt1h_pooled_2.bedpe'
coh_track_file = '/mnt/raid/data/Petros_project/bw/RAD21_ChIPseq/mm_BMDM_WT_Rad21_heme_60min.bw'
bw_file1 = '/mnt/raid/data/Petros_project/bw/BACH1_ChIPseq/mm_Bach1_1h_rep1_heme_merged.bw'
bw_file2 = '/mnt/raid/data/Petros_project/bw/RNAPol_ChIPseq/WT-RNAPOLIIS2-1h-heme100-rep1_S5.bw'
bw_files = [bw_file1,bw_file2]

sim = LoopSage(region,chrom,bedpe_file,label=label,N_lef=50,N_beads=1000,bw_files=bw_files,track_file=coh_track_file)
Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps,MC_step,burnin,T,T_min,poisson_choice=True,mode='Annealing',viz=True,save=True)
sim.run_MD()
```

Firstly, we need to define the input files from which LoopSage would take the information to construct the potential. We define also the specific region that we would like to model. Therefore, in the code script above we define a `bedpe_file` from which information about the CTCF loops it is imported. In `coh_track_file` you can optionally define the track file with some cohesin coverage of ChIP-Seq to determine the distribution of LEFs and allow preferencial binding in regions with higher signal. Then the user can optionally define a list of BigWig files which are needed in case that user would like to model other protein factors and their coefficients $r_i$.

Then, we define the main parameters of the simulation `N_beads,N_coh,kappa,f,b` or we can choose the default ones (take care because it might be the case that they are not the appropriate ones), the parameters of Monte Carlo `N_steps, MC_step, burnin, T`, and we initialize the class `LoopSage()`. The command `sim.run_energy_minimization()` corresponds to the stochastic Monte Carlo simulation, and it produces a set of cohesin constraints as a result (`Ms, Ns`). Note that the stochastic simulation has two modes: `Annealing` and `Metropolis`. We feed cohesin constraints to the molecular simulation part of and we run `MD_LE()` or `MD_EM()` simulation which produces a trajectory of 3d-structures, and the average heatmap.

### Output Files
In the output files, simulation produces one folder with 4 subfolders. In subfolder plots, you can find plots that are the diagnopstics of the algorithm. One of the most basic results you should see is the trajectories of cohesins (LEFs). this diagram should look like that,

![coh_trajectories](https://github.com/SFGLab/LoopSage/assets/49608786/f73ffd2b-8359-4c6d-958b-9a770d4834ba)

In this diagram, each LEF is represented by a different colour. In case of Simulated Annealing, LEFs should shape shorter loops in the first simulation steps, since they have higher kinetic energy due to the high temperature, and very stable large loops in the final steps if the final temperature $T_f$ is low enough. Horizontal lines represent the presence of CTCF points. In case of Metropolis, the distribution of LEFs should look more stable,

![coh_trajectories](https://github.com/SFGLab/LoopSage/assets/49608786/b48e2383-2509-4c68-a726-91a5e61aabf3)

Good cohesin trajectory diagrams should be like the ones previously shown, which means that we do not want to see many unoccupied (white) regions, but we also do not like to see static loops. If the loops are static then it is better to choose higher temperature, or bigger number of LEFs. If the loops are too small, maybe it is better to choose smaller temperature.

Now, to reassure that our algorithm works well we need to observe some fundamental diagnostics of Monte Carlo algorithms.
