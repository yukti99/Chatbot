import torch
import torch.nn as nn

# Feed-forward Neural Network with two hidden layers
class NeuralNet(nn.Model):
    def __init__(self,input_size,hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,num_classes)
        # activation function
        self.relu = nn.ReLU()

    # to implement the forward nn
    def forward(self,x):
        # input layer
        out = self.l1(x)
        # applying activation function
        out = self.relu(out)
        # second linear layer
        out = self.l2(out)
        # applying activation function
        out = self.relu(out)
        # third linear layer
        out = self.l3(out)
        # no activation and no softmax(for probabilities) as we will use cross-entropy which applies this anyway
        return out
