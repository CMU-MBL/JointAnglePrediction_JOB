# -------------------------
#
# Create custom 1-Dimensional Convolution Model
#
# --------------------------

import torch.nn as nn
from torch.nn import *  # noqa


class CustomConv1D(nn.Module):
    def __init__(self, inp_size=[42], outp_size=[18], window=31, layers=[42], groups=[1],
                 conv_activation=['ReLU'], conv_batchnorm=False, conv_dropout=[],
                 lin_layers=[], lin_activation=['ReLU'], lin_dropout=[], conv_layers=None,
                 **kwargs):

        super(CustomConv1D, self).__init__()
        pad = int((window-1)/2)

        if conv_layers is not None:
            layers = conv_layers

        # Convolutional arguments
        self.conv_sizes = inp_size + layers
        self.linear_sizes = [layers[-1]] + lin_layers + outp_size
        self.conv_batchnorm = conv_batchnorm
        self.conv_dropout = conv_dropout
        self.lin_dropout = lin_dropout

        # Set up convolutional layers
        for i in range(0, len(self.conv_sizes)-1):
            # Set current convolutional layer
            cur_name = 'conv' + str(i)
            cur_layer = nn.Conv1d(self.conv_sizes[i], self.conv_sizes[i+1], kernel_size=window,
                                  groups=groups[i], stride=1, padding=pad)
            setattr(self, cur_name, cur_layer)
            # Set current activation layer
            cur_name = 'conv_act' + str(i)
            cur_layer = globals()[conv_activation[i]]()
            setattr(self, cur_name, cur_layer)
            # Set batchnorm layer is requested
            if conv_batchnorm:
                cur_name = 'conv_batchnorm' + str(i)
                cur_layer = nn.BatchNorm1d(self.conv_sizes[i+1])
                setattr(self, cur_name, cur_layer)
            if len(conv_dropout) > 0:
                cur_name = 'conv_dropout' + str(i)
                cur_layer = nn.Dropout(conv_dropout[i])
                setattr(self, cur_name, cur_layer)

        for i in range(0, len(self.linear_sizes)-2):
            # Linear layer
            cur_name = 'lin' + str(i)
            cur_layer = nn.Linear(self.linear_sizes[i], self.linear_sizes[i+1])
            setattr(self, cur_name, cur_layer)
            # Set current activation layer
            cur_name = 'lin_act' + str(i)
            cur_layer = globals()[lin_activation[i]]()
            setattr(self, cur_name, cur_layer)
            if len(lin_dropout) > 0:
                cur_name = 'lin_dropout' + str(i)
                cur_layer = nn.Dropout(lin_dropout[i])
                setattr(self, cur_name, cur_layer)

        # Set final linear output layer
        self.lin_out = nn.Linear(self.linear_sizes[-2], self.linear_sizes[-1])

    def forward(self, x):
        # Transpose inpout for convolutional layers
        x = x.transpose(1, 2)
        # Pass through convolutional layers
        for i in range(0, len(self.conv_sizes)-1):
            # Apply current convolutional layer
            cur_name = 'conv' + str(i)
            cur_layer = getattr(self, cur_name)
            if i == 0:
                out = cur_layer(x)
            else:
                out = cur_layer(out)

            # Apply current convolutional activation layer
            cur_name = 'conv_act' + str(i)
            cur_layer = getattr(self, cur_name)
            out = cur_layer(out)
            # Apply batch norm and dropout if specified
            if self.conv_batchnorm:
                cur_name = 'conv_batchnorm' + str(i)
                cur_layer = getattr(self, cur_name)
                out = cur_layer(out)
            if len(self.conv_dropout) > 0:
                cur_name = 'conv_dropout' + str(i)
                cur_layer = getattr(self, cur_name)
                out = cur_layer(out)

        # Transpose output of the convolutional layer back into original shape
        out = out.transpose(1, 2)
        # Pass through linear layers
        for i in range(0, len(self.linear_sizes)-2):
            # Apply current linear layer
            cur_name = 'lin' + str(i)
            cur_layer = getattr(self, cur_name)
            out = cur_layer(out)
            # Apply current linear activation layer
            cur_name = 'lin_act' + str(i)
            cur_layer = getattr(self, cur_name)
            out = cur_layer(out)
            if len(self.lin_dropout) > 0:
                cur_name = 'lin_dropout' + str(i)
                cur_layer = getattr(self, cur_name)
                out = cur_layer(out)
        # Pass through final output layer
        cur_layer = getattr(self, 'lin_out')
        y = self.lin_out(out)

        return y
