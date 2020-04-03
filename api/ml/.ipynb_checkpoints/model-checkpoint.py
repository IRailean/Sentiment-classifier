import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak,self.sub,self.maxv = leak,sub,maxv

    def forward(self, x): 
        x = F.leaky_relu(x,self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None: x.sub_(self.sub)
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x
    
def conv1d(in_dim, out_dim, kernel_size, pooling=False):
    layers = []
    layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=0))
    layers.append(GeneralRelu(leak=.1, sub=.4))
    if pooling: layers.append(nn.MaxPool1d(3))
    return nn.Sequential(*layers)

def linear(in_dim, out_dim, dropout_rate=0.5):
    layers = []
    layers.append(nn.Linear(in_dim, out_dim))
    layers.append(GeneralRelu(leak=.1, sub=.4))
    layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

def flatten(x):
    return x.view(x.size(0), -1)

# Model
class CharacterCNN(nn.Module):
    def __init__(self):
        super(CharacterCNN, self).__init__()
        
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
        self.alphabet_size = len(self.alphabet)
        self.max_length = 300
        self.number_of_classes = 3
        self.layers = []

        self.layers.append(conv1d(self.alphabet_size, 256, 7, pooling=True))
        self.layers.append(conv1d(256, 256, 7, pooling=True))
        self.layers.append(conv1d(256, 256, 3, pooling=False))
        self.layers.append(conv1d(256, 256, 3, pooling=False))
        self.layers.append(conv1d(256, 256, 3, pooling=False))
        self.layers.append(conv1d(256, 256, 3, pooling=True))
        self.layers.append(Lambda(flatten))
        
        input_shape = (128,
                       self.max_length,
                       len(self.alphabet))
        self.output_dimension = self._get_conv_output(input_shape)
        
    
        self.layers.append(linear(1792, 1024))
        self.layers.append(linear(1024, 1024))
        self.layers.append(nn.Linear(1024, self.number_of_classes))

        idx = 0
        for module in self.layers:
            idx += 1
            name = "layer" + str(idx)
            self.add_module(name, module)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, a=0.1)
                module.bias.data.zero_()

    def forward(self, x):
        x = x.transpose(1, 2)
        for i,l in enumerate(self.layers):
            x = l(x)
        return x
    
    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = x.transpose(1, 2)
        for l in self.layers:
            if isinstance(l, Lambda):
                break
            x = l(x)            
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension
    
    def get_model_parameters(self):
        return {
            'alphabet': self.alphabet,
            'number_of_characters': self.alphabet_size,
            'max_length': self.max_length,
            'num_classes': self.number_of_classes
        }