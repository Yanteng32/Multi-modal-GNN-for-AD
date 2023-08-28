import numpy as np
from enum import Enum

age_gap = 1

learning_rate = 0.001
weight_decay = 5e-4
minimum_epochs = 30
epochs = 100
epochlimit = 50
gap = 0.02
Kco = 3


hidden_channels_GCN = 32
hidden_channels_Cheb = 32


train_index = np.arange(140, 461)
val_index = np.arange(70, 140)
test_index = np.arange(0, 70)


early_stop = False


class ModelType(Enum):
    Cheb = 1
    GCN = 2


class TrainType(Enum):
    Single = 1
    Mixed = 2


train_type = TrainType.Mixed
#train_type = TrainType.Single

use_dot_product_adjacency = True
#use_dot_product_adjacency = False

class ModelInput:
    def __init__(self, features, edge_index, adjacency_values, tensor_adjacency):
        self.features = features
        self.edge_index = edge_index
        self.adjacency_values = adjacency_values
        self.tensor_adjacency = tensor_adjacency
        self.model_type = ModelType.Cheb # default model type

    def SetModelType(self, model_type: ModelType):
        self.model_type = model_type

    def SetModelWeight(self, model_weight):
        self.model_weight = model_weight


class LabelsAndMasks:
    def __init__(self, train_mask, val_mask, test_mask, labels):
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.labels = labels

    def getTrainY(self):
        return self.labels[self.train_mask]

    def getValY(self):
        return self.labels[self.val_mask]

    def getTestY(self):
        return self.labels[self.test_mask]
