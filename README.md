# LoopSage
An energy-based model for loop extrusion.

## The model

Let's assume that each cohesin $i$ can be represented of two coordinates $(m_{i},n_{i})$ we allow three moves in our simulation:

* Slide right ( $n_{i} -> n_{i+1}$ to the right).
* Slide left ( $m_{i} -> m_{i-1}$ to the left).
* Rebind somewhere else.


The main idea of the algorithm is to ensemble loop extrusion from a Boltzmann probability distribution, with Hamiltonian,

$$E = \dfrac{f}{N_{fold}}\sum_{i=1}^{N_{coh}}\sqrt{n_i-m_i}+\dfrac{\kappa}{N_{knot}}\sum_{i,j}K(m_i,n_i;m_j,n_j)+\dfrac{b}{N_{bind}}\sum_{i=1}^{N_{coh}}\left(L(m_i)+R(n_i)\right)$$

The first term corresponds to the folding of chromatin, and the second term is a penalty for the appearance of knots. Therefore, we have the function,
$K(m_{i},n_{i};m_{j},n_{j})$ which takes the value 1 when $m_{i} < m_{j} < n_{i} < n_{j}$ or $m_{i}=m_{j}$ or $m_{i}=n_{j}$.

These $L(\cdot), R(\cdot)$ functions are two functions that define the binding potential and they are orientation specific - so they are different for left and right position of cohesin (because CTCF motifs are orientation specific), therefore when we have a gap in these functions, it means presence of CTCF. These two functions are derived from data with CTCF binning and by running the script for probabilistic orientation. Moreover, by $N_{(\cdot)}$ we symbolize the normalization constants for each factor,

$$N_{fold}=N_{coh}\cdot \langle n_i-m_i\rangle,\quad N_{knot}=N_{coh},\quad N_{bind}=\sum_{k}\left(L(k)+R(k)\right).$$

Therefore, we define the folding, knotting and binding energy, which are also metrics that help us to understand the dynamics of our system,

$$E_{fold} = \dfrac{f}{N_{fold}}\sum_{i=1}^{N_{coh}}\sqrt{n_i-m_i},$$

and

$$E_{knot} = \dfrac{\kappa}{N_{knot}}\sum_{i,j}K(m_i,n_i;m_j,n_j),$$

and

$$E_{bind} = \dfrac{b}{N_{bind}}\sum_{i=1}^{N_{coh}}\left(L(m_i)+R(n_i)\right)$$

And we can write the energy differences as,

$$\Delta E_{fold} = \dfrac{f}{N_{fold}}\left(\sqrt{n^{\prime}_i-m^{\prime}_i}-\sqrt{n_i-m_i}\right),$$

and

$$\Delta E_{knot} = \dfrac{\kappa}{N_{knot}} \left( \sum_{j}K(m^{\prime}_i,n^{\prime}_i;m_j,n_j)-\sum_{j}K(m_i,n_i;m_j,n_j)\right)$$

and

$$\Delta E_{bind} = \dfrac{b}{N_{bind}}\left( L(m^{\prime}_i)+R(n^{\prime}_i)-L(m_i)-R(n_i)\right)$$

where the prime values, symbolize the new coordinates of cohesin, if the new move is accepted. Thus,

$$\Delta E = \Delta E_{fold}+\Delta E_{knot}+\Delta E_{bind}.$$

In this manner we accept a move in two cases:

* If $\Delta E<0$ or,
* if $\Delta E>0$ with probability $e^{-\Delta E/kT}$.

And of course, the result - the distribution of loops in equilibrium -  depends on temperature of Boltzmann distribution $T$.



## How to use?

The implementation of the code is very easy and it can be described in the following lines,

```python
N_beads,N_coh,kappa,f,b = 1000,50,100000,-500,-500
N_steps, MC_step, burnin, T = int(1e4), int(1e2), 15, 2
L, R = binding_from_bedpe_with_peaks("/mnt/raid/data/Zofia_Trios/bedpe/hg00731_CTCF_pulled_2.bedpe",N_beads,[212520553-50000,213377421+50000],'chr2',False)
print('Number of CTCF:',np.max([np.count_nonzero(L),np.count_nonzero(R)]))
sim = LoopSage(N_beads,N_coh,kappa,f,b,L,R)
Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(N_steps,MC_step,burnin,T,mode='Annealing',viz=True,vid=True)
md = MD_LE(Ms,Ns,N_beads,burnin,MC_step)
md.run_pipeline(write_files=True,plots=True)
```

Therefore, we define the main parameters of the simulation `N_beads,N_coh,kappa,f,b`, the parameters of Monte Carlo `N_steps, MC_step, burnin, T`, and we initialize the class `LoopSage()`. The command `sim.run_energy_minimization()` corresponds to the stochastic Monte Carlo simulation, and it produces a set of cohesin constraints as a result (`Ms, Ns`). Note that the stochastic simulation has two modes: `Annealing` and `Metropolis`. We feed cohesin constraints to the molecular simulation part of and we run `MD_LE()` simulation which produces a trajectory of 3d-structures, and the average heatmap.
