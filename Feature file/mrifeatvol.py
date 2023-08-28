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

#from ncmatrix import NCmatrix


filepathad = 'E:/GCNduym/pmcivssmci/data/mri/smcipmci2csv/'


def featmrivol():
    
    filenamesad = os.listdir(filepathad)
    xfealistin = []
    xfealist = []

    for fi in filenamesad:
        namenii = fi[:-4]
        df = pd.read_csv(filepathad + fi)
        #df = df.loc[:,['Vgm', 'Vwm', 'Vcsf', 'Vlm']] 
        df = df.loc[:, ['Vgm']] 
        data = df.values  # data是数组，直接从文件读出来的数据格式是数组
        #data = df.iloc[:, 1:]
        index1 = list(df.keys())  # 获取原有csv文件的标题，并形成列表
        data = list(map(list, zip(*data)))  # map()可以单独列出列表，将数组转换成列表
        data = pd.DataFrame(data, index=index1)  # 将data的行列转换
        #mrimix = data.values
        mrimix = list(np.array(data).flatten())
        #mrimix = data.values.tolist()
        xfealist.append(mrimix)
    xfealistin = xfealist

    return xfealistin

x = featmrivol()

          


