import torch
from torch import nn
from collections import OrderedDict


class ConvBlock(nn.Module):

    '''
    A base convolutional block.

    input params:
        in_channels (int):
            `in_channels` of the conv layers (`nn.Conv2d`).

        out_channels (int)
            `out_channels` of the conv layers (`nn.Conv2d`).

        kernel_size (int, optional, default = 3):
            `kernel_size` of the conv layers.

        stride (int, optional, default = 1):
            `stride` of the convolutional layers (`nn.Conv2d`).

        pool (bool, optional, default = True):
            Use max pooling or not.

        pool_kernel_size (int, optional, default = 2):
            `kernel_size` of the max pooling layer. Only make sense when
            `pool = True`.
    
    NOTE:
        OmniglotCNN: 3 × 3 conv + batch norm + ReLU
        MiniImagenetCNN: 3 × 3 conv + batch norm + ReLU + 2 × 2 max-pooling
    '''

    def __init__(self, in_channels, out_channels, kernel_size = 3, 
                 stride = 1, pool = True, pool_kernel_size = 2):

        super(ConvBlock, self).__init__()

        module_list = [
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size,
                      stride = stride, padding = 1, bias = True),
                       # (batch_size, out_channels, img_size, img_size)
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        
        if pool:
            module_list.append(
                nn.MaxPool2d(kernel_size = pool_kernel_size)
            )# (batch_size, out_channels, img_size / 2, img_size / 2)

        self.core = nn.Sequential(*module_list)
        self.init_weights()
    

    def init_weights(self):
        # conv layer
        nn.init.xavier_uniform_(self.core[0].weight.data, gain = 1.0)
        nn.init.constant_(self.core[0].bias.data, 0.0)
        # batch normalization layer
        nn.init.uniform_(self.core[1].weight)


    def forward(self, x):
        output = self.core(x)
        return output


class ConvGroup(nn.Module):

    '''
    A base convolutional group.

    input params:
        in_channels (int, optional, default = 1):
            `in_channels` of the conv layers (`nn.Conv2d`).

        hidden_size (int, optional, default = 64):
            The dimensionality of the hidden representation.

        kernel_size (int, optional, default = 3):
            `kernel_size` of the conv layer.

        stride (int, optional, default = 1):
            `stride` of the conv layers (`nn.Conv2d`).

        pool (bool, optional, default = True):
            Use max pooling or not.

        pool_kernel_size (int, optional, default = 2):
            `kernel_size` of the max pooling layer. Only make sense when
            `pool = True`.
        
        layers (int, optional, default = 4):
            The number of convolutional layers (`ConvBlock`).

    NOTE:
        Omniglot: hidden_size = 64, in_channels = 1, pool = False
        MiniImagenet: hidden_size = 32, in_channels = 3, pool = True
    '''

    def __init__(self, in_channels = 1, hidden_size = 64, kernel_size = 3, 
                 stride = 1, pool = True, pool_kernel_size = 2, layers = 4):
        
        super(ConvGroup, self).__init__()

        module_list = [
            ConvBlock(in_channels, hidden_size, kernel_size, stride, pool, pool_kernel_size)
        ]

        for _ in range(layers - 1):
            module_list.append(
                ConvBlock(hidden_size, hidden_size, kernel_size, stride, pool, pool_kernel_size)
            )

        self.core = nn.Sequential(*module_list)
    

    def forward(self, x):
        output = self.core(x)
        return output


class LinearBlock(nn.Module):

    '''
    A base linear block.

    input params:
        input_size (int):
            The size of the input sample of the linear layer (`in_features`).

        output_size (int)
            The size of the output sample of the linear layer (`out_features`).
    '''

    def __init__(self, input_size, output_size):
        
        super(LinearBlock, self).__init__()

        self.core = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

        self.init_weights()
    

    def init_weights(self):
        # linear layer
        nn.init.xavier_uniform_(self.core[0].weight)
        nn.init.constant_(self.core[0].bias.data, 0.0)
    

    def foward(self, x):
        output = self.core(x)
        return output