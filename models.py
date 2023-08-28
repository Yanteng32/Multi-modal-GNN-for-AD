import torch
import parameters
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Linear
from torch_geometric.nn import ChebConv
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, dim_nodes, hidden_channels=128):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dim_nodes, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        #self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, 2)

    def forward(self, x, adjacency):  # edge_index
        # 1. Obtain node embeddings
        x = self.conv1(x, adjacency)
        x = F.relu(x)
        x = self.conv2(x, adjacency)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)

        return x


class Cheb(torch.nn.Module):
    def __init__(self, dim_nodes, hidden_channels):
        super(Cheb, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = ChebConv(dim_nodes, hidden_channels, K=parameters.Kco)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=parameters.Kco)
        #self.conv3 = ChebConv(hidden_channels, hidden_channels, K=parameters.Kco)
        self.lin1 = Linear(hidden_channels, 2)

    def forward(self, x, *adjacency):
        x = self.conv1(x, *adjacency)
        x = F.relu(x)
        x = self.conv2(x, *adjacency)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        return x


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use-bias: bool, optional
                是否使用偏置
        """

        super(GraphConvolution, self).__init__()  # super() 函数是用于调用父类(超类)的一个方法。
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
        Args：
        --------------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """

        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


class GcnNet(nn.Module):
    def __init__(self, input_dim, hidden_channels = 64):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_channels)
        self.gcn2 = GraphConvolution(hidden_channels, 2)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits