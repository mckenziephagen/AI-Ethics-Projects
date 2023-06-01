import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import EdgeConv, GCNConv, GraphConv

class ConnectomeNet(nn.Module):
    """
    Baseline ConnectomeNet
    """
    def __init__(self, input_size, hidden_size):

        super(ConnectomeNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.05),

            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(0.05),

            nn.Linear(hidden_size//2, hidden_size//4),
            nn.LeakyReLU(0.05),

            nn.Linear(hidden_size//4, 1)
        )
        self.__weights_init_normal("kaiming")

    def forward(self, x):
        return self.layers(x)

    def __weights_init_normal(self,  init_type):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if init_type == "kaiming":
                    nn.init.kaiming_normal_(layer.weight, a=0.2, nonlinearity="leaky_relu")
                    nn.init.constant_(layer.bias, 0)


class ConnectomeNet_LNL(ConnectomeNet):
    """
    ConnectomeNet implementation for LearnNotToLearn bias correction

    """
    def __init__(self, input_size, hidden_size):

        super(ConnectomeNet_LNL, self).__init__(input_size, hidden_size)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)

        feat_out = x
        x = self.layers[-1](x)

        return  feat_out, x

class Predictor(nn.Module):
    """
    Classifier for bias predictor
    """
    def __init__(self, input_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.LeakyReLU(0.05),
            nn.Linear(input_size//2, input_size//4),
            nn.LeakyReLU(0.05),
            nn.Linear(input_size//4, 1)
        )
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layers(x)
        probs = self.sigmoid(x).squeeze()
        labels = (probs > 0.5).float()

        return labels, probs


class Net(nn.Module):
    def __init__(self, input_size, hidden_size):

        super(Net, self).__init__()

        self.negative_slope = 0.05
        self.kernel_size = 2
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size//self.kernel_size, hidden_size//(self.kernel_size)))
        self.layers.append(nn.Linear(hidden_size//(self.kernel_size**2), 100))
        self.layers.append(nn.Linear(100, 1))

        self.__weights_init_normal("kaiming")

    def forward(self, x):

        for i in range(2):
            x  = self.layers[i](x)
            x = F.leaky_relu(x, self.negative_slope)
            x = F.max_pool1d(x, self.kernel_size)

        for i in range(2, len(self.layers)):
            x  = self.layers[i](x)
            x = F.leaky_relu(x, self.negative_slope)

        return x

    def __weights_init_normal(self,  init_type):
        """
            Kaiming initialization:

            normal distribution with var = 2/dh) or var = 2/(1+ alpha**2)dh
        """

        for i in range(len(self.layers)):
            layer = self.layers[i]

            if isinstance(self.layers[i], nn.Linear):
                if init_type == "kaiming":
                    nn.init.kaiming_normal_(layer.weight, a = self.negative_slope, nonlinearity = "leaky_relu")
                    nn.init.constant_(layer.bias, 0)

                elif init_type == "normal":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity = "relu")
                    nn.init.constant_(layer.bias, 0)

                else:
                    raise

class GraphNetwork(torch.nn.Module):
    """

    """
    def __init__(
        self,
        hidden_channels,
        num_node_features,
        num_output_classes
    ):
        super().__init__()

        # Initialize MLPs used by EdgeConv layers
        self.mlp1 = Sequential(Linear(2 * num_node_features, hidden_channels), ReLU())
        self.mlp2 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())
        self.mlp3 = Sequential(torch.nn.Linear(2 * hidden_channels, hidden_channels), ReLU())

        # Initialize EdgeConv layers
        self.conv1 = EdgeConv(self.mlp1, aggr='max')
        self.conv2 = EdgeConv(self.mlp2, aggr='max')
        self.conv3 = EdgeConv(self.mlp3, aggr='max')

        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        self.linear = torch.nn.Linear(hidden_channels, num_output_classes)

    def forward(self, data):
        """ Performs a forward pass on our simplified cGCN.

        Parameters:
        data (Data): Graph being passed into network.

        Returns:
        torch.Tensor (N x 2): Probability distribution over class labels.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)

        x = global_mean_pool(x, batch)
        x = self.linear(x)
        # x = F.softmax(x, dim=1)

        return x.float()
