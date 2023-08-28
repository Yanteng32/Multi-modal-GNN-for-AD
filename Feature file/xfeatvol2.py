#-*- coding: utf-8 -*-
import itertools
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool 
from sklearn.manifold import TSNE
from matplotlib import cm
from matplotlib import pyplot as plt 
import pandas
import pandas as pd
import csv
from numpy import array
import math 
import scipy.stats
import seaborn
import seaborn as sns
from scipy.sparse import coo_matrix
from itertools import islice
from scipy.spatial import distance
import sklearn.metrics
import scipy.io as sio

from mrifeatvol import featmrivol
from ncmatrix2 import NCmatrix


rootpath = r'E:\GCNduym\pmcivssmci\data\mri' 

xvl = featmrivol()

def aves():

    with open(rootpath + '/' + 'smcipmci2v0620roi10.csv', encoding="utf-8") as csvfile:    #列是脑区标签，无id标签
        lines = csv.reader(csvfile)
        list = []
        ss = []
     
        for row in xvl:
            scores = row           
            scores = np.array(scores).astype(dtype=str).tolist()
            vols = scores  # list:132
            volss = [float(x) if type(x) is str else None for x in vols]
            variance = np.var(volss)
            variance1 = variance ** 0.5
            ss += [variance1]
            a = sum([float(score) for score in scores])
            b = len(scores)
            #average = sum([float(score) for score in scores]) / len(scores)   #float
            average = a / b  # float
            list += [average]
 
        aves = list
    return aves, ss
 

aves, ss = aves()

#aves = aves()

def xfeaturelistlist():
    #定义一个2维的保存ESD
    with open(rootpath + '/' + 'smcipmci2v0620roi10.csv', encoding="utf-8") as csvfile:    #列是脑区标签，无id标签
        lines = csv.reader(csvfile) 
        xfealistin = []        
        for roww in xvl:
            vols = roww
            vols = np.array(vols).astype(dtype=str).tolist()
            ESD0 = []
            xfealist = []
            dem = len(vols)
            for m in range(0, dem):
                ESDM = []            
                for n in range(0, dem):
                    sij = ((ss[m]*ss[m]+ss[n]*ss[n])/2)**0.5
                    ESDMN = abs((float(vols[m])-float(aves[m]))-(float(vols[n])-float(aves[n])))/sij
                    ESDMN = math.exp(2 * ESDMN)
                    ESDMN = (ESDMN - 1) / (ESDMN + 1)
                    ESDMN = 1 - ESDMN
                    ESDM.append(ESDMN)
                ESD0.append(ESDM)
            ESD0 = np.array(ESD0, dtype="float32")
            NC1 = NCmatrix()
            ESD = ESD0 * NC1
        
            liexx = ESD.shape[0]
            xlist = []
            xxlist = []
        
            for xline in ESD:
                xlist.append(xline)
        
            for k in range(0, liexx):
                a1 = xlist[k]
                aa1 = a1[k: liexx]
                #arr2 = list(aa1)
                arr2 = aa1
                #arr2 = np.array(aa1,dtype=float)
                xxlist.append(arr2)

            xfea = [bb for aaa in xxlist for bb in aaa]
            xfealist.append(xfea)
            #xfealist = np.array(xfealist, dtype="float64")
            xfealistin.append(xfealist)
            
            #plt.clf()
            #fig0 = sns.heatmap(ESD0, annot=False, fmt='.3g', cmap='rainbow', vmin = 0, vmax = 1, square = True, xticklabels = False, yticklabels = False, cbar = False)
            #fig0.get_figure().savefig(subname+'df_corr0.png', dpi=300, bbox_inches='tight')
            #fig = sns.heatmap(ESD, annot=False, fmt='.3g', cmap='Blues',  square = True, xticklabels = False, yticklabels = False, cbar = False) #vmin = 0, vmax = 1.  cbar = False cmap='YlOrRd'
            #fig.get_figure().savefig(subname+'corrbl.png', dpi=300, bbox_inches='tight')
            #fig = sns.heatmap(NC1, annot=False, fmt='.3g', cmap='rainbow', vmin = 0, vmax = 1, square = True, xticklabels = False, yticklabels = False, cbar = False)
            #fig.get_figure().savefig(subname+'NC1.png', dpi=300, bbox_inches='tight')
            
        mm = len(xfea)
        nn = len(xfealistin)
        xfealistin = np.array(xfealistin, dtype="float32").reshape(nn, mm)
    
    return xfealistin


x = xfeaturelistlist()
data = pd.DataFrame(x)
data.to_csv('smcipmci2vlg2new'+'.csv')