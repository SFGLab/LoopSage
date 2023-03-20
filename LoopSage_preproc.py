import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

def binding_vectors_from_bedpe_with_peaks(bedpe_file,N_beads,region,chrom,normalization=False):
    # Read file and select the region of interest
    df = pd.read_csv(bedpe_file,sep='\t',header=None)
    df = df[(df[1]>=region[0])&(df[5]<=region[1])&(df[0]==chrom)].reset_index(drop=True)

    # Convert hic coords into simulation beads
    resolution = (region[1]-region[0])//N_beads
    df[1], df[2], df[4], df[5] = (df[1]-region[0])//resolution, (df[2]-region[0])//resolution, (df[4]-region[0])//resolution, (df[5]-region[0])//resolution
    
    # Compute the matrix
    distances = list()
    L, R = np.zeros(N_beads),np.zeros(N_beads)
    for i in range(len(df)):
        x, y = (df[1][i]+df[2][i])//2, (df[4][i]+df[5][i])//2
        distances.append(distance_point_line(x,y))
        if df[7][i]>=0: L[x] += df[6][i]*(1-df[7][i])
        if df[8][i]>=0: L[y] += df[6][i]*(1-df[8][i])
        if df[7][i]>=0: R[x] += df[6][i]*df[7][i]
        if df[8][i]>=0: R[y] += df[6][i]*df[8][i]
    
    # Normalize (if neccesary): it means to convert values to probabilities
    if normalization:
        L, R = L/np.sum(L), R/np.sum(R)

    # sns.histplot(distances, kde=True, bins=100)
    # plt.ylabel('Count')
    # plt.xlabel('Loop Size')
    # plt.grid()
    # plt.show()

    print('Average loop size:', np.average(distances))
    print('Median loop size:', np.median(distances))
    print('Maximum loop size:', np.max(distances))

    return L, R, distances

def binding_matrix_from_bedpe(bedpe_file,N_beads,region,chrom,normalization=False):
    # Read file and select the region of interest
    df = pd.read_csv(bedpe_file,sep='\t',header=None)
    df = df[(df[1]>=region[0])&(df[5]<=region[1])&(df[0]==chrom)].reset_index(drop=True)
    
    # Convert hic coords into simulation beads
    resolution = (region[1]-region[0])//N_beads
    df[1], df[2], df[4], df[5] = (df[1]-region[0])//resolution, (df[2]-region[0])//resolution, (df[4]-region[0])//resolution, (df[5]-region[0])//resolution
    
    # Compute the matrix
    distances = list()
    M = np.zeros((N_beads,N_beads))
    for i in range(len(df)):
        x, y = (df[1][i]+df[2][i])//2, (df[4][i]+df[5][i])//2
        distances.append(distance_point_line(x,y))
        M[x,y] += df[6][i]
        M[y,x] += df[6][i]

    # Normalize (if neccesary): it means to convert values to probabilities
    if normalization: M = M/np.sum(M)

    sns.histplot(distances, kde=True)
    plt.ylabel('Count')
    plt.xlabel('Loop Size')
    plt.grid()
    plt.show()

    print('Average loop size:',np.average(distances))
    print('Median loop size:',np.median(distances))
    print('Maximum loop size:', np.max(distances))

    return M, distances

def hiccups_edit(file):
    f = pd.read_csv(file,sep='\t')
    new_f = f[['#chr1','x1','x2','chr2','y1','y2','expectedDonut']]
    for i in range(len(f)):
        new_f['#chr1'][i] = 'chr'+str(new_f['#chr1'][i])
        new_f['chr2'][i] = 'chr'+str(new_f['chr2'][i])
    new_f['x1'], new_f['x2'], new_f['y1'], new_f['y2'],  new_f['expectedDonut'] = new_f['x1'].astype(int), new_f['x2'].astype(int), new_f['y1'].astype(int), new_f['y2'].astype(int), new_f['expectedDonut'].apply(np.int64)
    new_f.to_csv(file.split('.')[0]+'_edited.bedpe',sep='\t',index=False)

def distance_point_line(x0,y0,a=1,b=-1,c=0):
    return np.abs(a*x0+b*y0+c)/np.sqrt(a**2+b**2)