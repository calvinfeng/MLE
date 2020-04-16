import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
        self.fully_connected_1 = nn.Linear(input_dim, hidden_dim)
        self.fully_connected_2 = nn.Linear(hidden_dim, output_dim)
        self.drop_out = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
        
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        out = F.relu(self.fully_connected_1(x))
        out = self.drop_out(out)
        out = self.fully_connected_2(out)
        return self.sigmoid(out)