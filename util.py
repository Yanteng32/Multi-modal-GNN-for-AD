import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as functional
import torch.optim as optim
from itertools import islice
from scipy.spatial import distance
from scipy.sparse import coo_matrix
from torch_sparse import SparseTensor
import parameters
from sklearn.metrics import roc_auc_score, confusion_matrix
import models
import random


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot



def getAgeAndSexMaskFromCSV(path):
    columnsex = []  # 定义列数组
    columnage = []
    columnapoe = []
    columnmmse = []
    columnallone = []
    with open(path, "rt", encoding="utf-8") as csvfile:
        lines = csvfile.readlines()
        rows = []  # 定义行数组

        for line in lines:
            line = line.split(",")
            rows.append(line)
        header = rows[0]
        sexIndex = header.index('Sex')
        ageIndex = header.index('Age')
        apoeIndex = header.index('APOE4')
        mmseIndex = header.index('MSE')
        alloneIndex = header.index('allone')

        for row in islice(rows, 1, None):
            columnsex.append(row[sexIndex])
            columnage.append(row[ageIndex])
            columnapoe.append(row[apoeIndex])
            columnmmse.append(row[mmseIndex])
            columnallone.append(row[alloneIndex])

    dim = len(columnage)
    sexMask = np.zeros((dim, dim))
    ageMask = np.zeros((dim, dim))
    apoeMask = np.zeros((dim, dim))
    mmseMask = np.zeros((dim, dim))
    alloneMask = np.zeros((dim, dim))
    
    for i in range(0, dim):
        for j in range(i, dim):
            if columnsex[i] == columnsex[j]:
                sexMask[i, j] = 1
                sexMask[j, i] = 1
            if abs(float(columnage[i]) - float(columnage[j])) <= parameters.age_gap:
                ageMask[i, j] = 1
                ageMask[j, i] = 1
            if columnapoe[i] == columnapoe[j]:
                apoeMask[i, j] = 1
                apoeMask[j, i] = 1                
            if abs(float(columnmmse[i]) - float(columnmmse[j])) <= 1:
            #if columnmmse[i] == columnmmse[j]:
                mmseMask[i, j] = 1
                mmseMask[j, i] = 1
            if columnallone[i] == columnallone[j]:
                alloneMask[i, j] = 1
                alloneMask[j, i] = 1 
            
    sexageMask = sexMask + ageMask
    mmsapoMask = mmseMask + apoeMask
    sexmmsMask = sexMask + mmseMask
    allMask = sexMask + apoeMask + mmseMask   
    return sexMask, ageMask, sexageMask, apoeMask, mmseMask, mmsapoMask, sexmmsMask, allMask, alloneMask


def prepareCombinedData(path0 : str, path1 : str, dataMask, device) -> [any, any, parameters.LabelsAndMasks]:
    dataFrame0 = pd.read_csv(path0)
    dataFrame1 = pd.read_csv(path1)

    feature0 = dataFrame0.iloc[:, 0:-2].values
    feature1 = dataFrame1.iloc[:, 0:-2].values
    feature_combined = np.hstack((feature0, feature1))
    df_y = dataFrame0.values[:, -1]

    distv = distance.pdist(feature_combined, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    adjacency = np.exp(- dist ** 2 / (2 * sigma ** 2))
    adjacency = dataMask * adjacency

    labels_ad = encode_onehot(df_y)
    labels = torch.LongTensor(np.where(labels_ad)[1])
    num_nodes = feature_combined.shape[0]

    train_mask = np.zeros(num_nodes, dtype=np.bool)
    val_mask = np.zeros(num_nodes, dtype=np.bool)
    test_mask = np.zeros(num_nodes, dtype=np.bool)

    train_mask[parameters.train_index] = True
    val_mask[parameters.val_index] = True
    test_mask[parameters.test_index] = True

    train_mask = torch.from_numpy(train_mask).to(device)
    val_mask = torch.from_numpy(val_mask).to(device)
    test_mask = torch.from_numpy(test_mask).to(device)
    labels = labels.to(device)

    labels_and_masks = parameters.LabelsAndMasks(train_mask, val_mask, test_mask, labels)

    print("Node's feature shape: ", feature_combined.shape)
    print("Node's label shape: ", labels.shape)
    print("Number of training nodes: ", train_mask.sum())
    print("Number of validation nodes: ", val_mask.sum())
    print("Number of test nodes: ", test_mask.sum())

    return adjacency, feature_combined, feature0, feature1, labels_and_masks

def prepareData(path, dataMask, device) -> [any, any, parameters.LabelsAndMasks]:
    dataFrame = pd.read_csv(path)  # 列是人标签，无id标签
    dfx = dataFrame.iloc[:, 0:-2]

    dfx_feature = dfx.values
    df_y = dataFrame.values[:, -1]

    distv = distance.pdist(dfx_feature, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    adjacency = np.exp(- dist ** 2 / (2 * sigma ** 2))

    if not parameters.use_dot_product_adjacency:
        adjacency = dataMask * adjacency

    labels_ad = encode_onehot(df_y)
    labels = torch.LongTensor(np.where(labels_ad)[1])
    num_nodes = dfx_feature.shape[0]

    train_mask = np.zeros(num_nodes, dtype=np.bool)
    val_mask = np.zeros(num_nodes, dtype=np.bool)
    test_mask = np.zeros(num_nodes, dtype=np.bool)

    train_mask[parameters.train_index] = True
    val_mask[parameters.val_index] = True
    test_mask[parameters.test_index] = True

    train_mask = torch.from_numpy(train_mask).to(device)
    val_mask = torch.from_numpy(val_mask).to(device)
    test_mask = torch.from_numpy(test_mask).to(device)
    labels = labels.to(device)

    labels_and_masks = parameters.LabelsAndMasks(train_mask, val_mask, test_mask, labels)

    print("Node's feature shape: ", dfx_feature.shape)
    print("Node's label shape: ", labels.shape)
    print("Number of training nodes: ", train_mask.sum())
    print("Number of validation nodes: ", val_mask.sum())
    print("Number of test nodes: ", test_mask.sum())

    return adjacency, dfx_feature, labels_and_masks


def normalization(adjacency):
    """计算 L = D^-0.5 * (A+I) * D^-0.5 """
    adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


def processData(features, device, adjacency, ) -> parameters.ModelInput:
    features = torch.tensor(features, dtype=torch.float32).to(device)
    num_nodes = features.shape[0]

    adjacency = coo_matrix(adjacency)
    normalize_adjacency = normalization(adjacency)

    edge_index = torch.from_numpy(
        np.asarray([normalize_adjacency.row, normalize_adjacency.col]).astype('int64')).long().to(device)

    adjacency_values = torch.from_numpy(normalize_adjacency.data.astype(np.float32)).to(device)

    tensor_adjacency = SparseTensor.from_torch_sparse_coo_tensor(
        torch.sparse.FloatTensor(edge_index, adjacency_values, (num_nodes, num_nodes)).to(device))

    modelInput = parameters.ModelInput(features, edge_index, adjacency_values, tensor_adjacency)
    return modelInput


#def getModels(model_inputs:list[parameters.ModelInput],device):
def getModels(model_inputs:parameters.ModelInput,device):
    modelArr = []
    optimizers = []
    for model_input in model_inputs:
        if model_input.model_type == parameters.ModelType.GCN:
            modelGCN = models.GCN(dim_nodes=model_input.features.shape[1],
                                  hidden_channels=parameters.hidden_channels_GCN).to(device)
            optimizerGCN = optim.Adam(modelGCN.parameters(), lr=parameters.learning_rate,
                                      weight_decay=parameters.weight_decay)
            modelArr.append(modelGCN)
            optimizers.append(optimizerGCN)
        elif model_input.model_type == parameters.ModelType.Cheb:
            modelCheb = models.Cheb(dim_nodes=model_input.features.shape[1],
                                    hidden_channels=parameters.hidden_channels_Cheb).to(device)
            optimizerCheb = optim.Adam(modelCheb.parameters(), lr=parameters.learning_rate,
                                       weight_decay=parameters.weight_decay)
            modelArr.append(modelCheb)
            optimizers.append(optimizerCheb)
    return modelArr, optimizers


def modelForward(model, model_input: parameters.ModelInput):
    if type(model) is models.Cheb:
        logits = model(model_input.features, model_input.edge_index, model_input.adjacency_values)  # 前向传播
    elif type(model) is models.GCN:
        logits = model(model_input.features, model_input.tensor_adjacency)  # 前向传播
    return logits


# 两个模型的输出概率平均
def votingConference(output_logits, labels):
    weight_0 = output_logits[0][1]
    weight_1 = output_logits[1][1]
    probabilities_0 = functional.softmax(output_logits[0][0], 1)
    probabilities_1 = functional.softmax(output_logits[1][0], 1)
    output = (probabilities_0 * weight_0 + probabilities_1 * weight_1)
    result = torch.max(output, 1)[1]
    accuracy = torch.eq(result, labels).float().mean()
    return accuracy, result


def earlyStop(epoch, best_train_acc, best_val_acc, current_train_acc, current_val_acc):
    if parameters.early_stop:
        if best_train_acc > 0.99 and (best_val_acc <= current_val_acc):
            return True
        if epoch > parameters.epochlimit and (
                best_val_acc - parameters.gap) <= current_val_acc and current_train_acc >= (
                best_train_acc - parameters.gap):
            return True
        return False
    else:
        return False


def confusionMatrix(true_label, data_pre):
    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    print('The result of test data for: ')
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)
    print('\r\n')


def calculateAccuracy(logits, labels, mask):
    test_mask_logits = logits[mask]
    predict_y = test_mask_logits.max(1)[1]
    accuracy = torch.eq(predict_y, labels[mask]).float().mean()
    return accuracy, predict_y


def test(model, test_mask, labels, model_input: parameters.ModelInput,
         show_confusion_matrix=False):
    model.eval()
    true_label = []
    data_pre = []
    with torch.no_grad():
        logits = modelForward(model, model_input)
        accuracy, predict_y = calculateAccuracy(logits, labels, test_mask)
        yy = predict_y
        zz = labels[test_mask]
        data_pre.extend(list(yy.cpu().flatten().numpy()))
        true_label.extend(list(zz.cpu().flatten().numpy()))

    if show_confusion_matrix:
        confusionMatrix(true_label, data_pre)
    return accuracy, logits[test_mask]
