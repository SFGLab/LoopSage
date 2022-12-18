import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def binding_from_bedpe_with_peaks(bedpe_file,N_beads,region,chrom,normalization=False):
    # Read file and select the region of interest
    df = pd.read_csv(bedpe_file,sep='\t',header=None)
    df = df[(df[1]>=region[0])&(df[5]<=region[1])&(df[0]==chrom)].reset_index(drop=True)

    # Convert hic coords into simulation beads
    resolution = (region[1]-region[0])//N_beads
    df[1], df[2], df[4], df[5] = (df[1]-region[0])//resolution, (df[2]-region[0])//resolution, (df[4]-region[0])//resolution, (df[5]-region[0])//resolution

    # Compute the matrix
    L, R = np.zeros(N_beads),np.zeros(N_beads)
    for i in range(len(df)):
        x, y = (df[1][i]+df[2][i])//2, (df[4][i]+df[5][i])//2
        L[x] += df[6][i]*(1-df[7][i])
        L[y] += df[6][i]*(1-df[8][i])
        R[x] += df[6][i]*df[7][i]
        R[y] += df[6][i]*df[8][i]

    # Normalize (if neccesary): it means to convert values to probabilities
    if normalization:
        L, R = L/np.sum(L), R/np.sum(R)

    return L, R