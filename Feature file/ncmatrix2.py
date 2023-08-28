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


filencgroup = 'E:/GCNduym/NCvolcsv/'


rootpath = r'E:/MRI&PETexp' 

rootpath1 = r'E:/GCNduym/pmcivssmci/data/mri' 

dfyy = pd.read_csv(rootpath + '/' + 'ncsuvr.csv')    #列是人标签，无id标签

 
filenamesad = os.listdir(filencgroup)
xfeatnc = []

for fi in filenamesad:
    namenii = fi[:-4]
    df = pd.read_csv(filencgroup + fi)
    #df = df.loc[:,['Vgm', 'Vwm', 'Vcsf', 'Vlm']] 
    df = df.loc[:, ['Vgm']]
    data = df.values  # data是数组，直接从文件读出来的数据格式是数组
    mrimix = data.flatten()

    xfeatnc.append(mrimix)
xfeatnc = np.array(xfeatnc)
dfxx = xfeatnc


def NCmatrix():

    #data = dfxx.values  # data是数组，直接从文件读出来的数据格式是数组
    data = dfxx
    #index1 = list(dfxx.keys())  # 获取原有csv文件的标题，并形成列表
    data = list(map(list, zip(*data)))  # map()可以单独列出列表，将数组转换成列表
    #data = pd.DataFrame(data, index=index1)  # 将data的行列转换
    x_datazhuan = pd.DataFrame(data)
    distv = distance.pdist(x_datazhuan, metric='correlation')
    dist = distance.squareform(distv)
    
    sigma = np.mean(dist)
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    NC1 = sparse_graph

    return NC1

