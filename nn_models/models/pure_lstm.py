# -------------------------
#
# Create custom Long Short-term Memory model
#
# --------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F


def rot6_to_rotmat(x):
    """Change rot 6d vector to rotation matrix
    x:      batch X seql X 12
    out:    batch X seql X 18
    """
    batch, seql = x.shape[:2]

    def conversion(x_):
        a1 = x_[:, :, 0]
        a2 = x_[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)

        x_ = torch.stack((b1, b2, b3), dim=-1)
        x_ = x_.view(batch, seql, 3, 3)

        return x_

    x = x.view(-1, 3, 4)
    x1, x2 = x[:, :, :2], x[:, :, 2:]
    
    x1 = conversion(x1)
    x2 = conversion(x2)
    
    x = torch.cat((x1, x2), dim=-1)
    
    return x



class CustomLSTM(nn.Module):
    def __init__(self, inp_size=[42], outp_size=[18], layers=[80, 80],
                 dropout=[0, 0], bidir=False, **kwargs):
        super(CustomLSTM, self).__init__()

        self.sizes = inp_size + layers + outp_size
        self.num_layers = len(layers)
        # Dropout adds after all but last layer, so non-zero dropout requires num_layers>1
        if self.num_layers <= 1:
            dropout = [0]
        self.homogenous = (layers[1:] == layers[:-1]) and (dropout[1:] == dropout[:-1])
        self.lstm0 = nn.LSTM(input_size=self.sizes[0],
                             hidden_size=self.sizes[1],
                             num_layers=self.num_layers,
                             batch_first=True,
                             bidirectional=bidir,
                             dropout=dropout[0])

        # Checks if LSTM is bidirectional to adjust output
        if bidir:
            self.linear_out = nn.Linear(2*self.sizes[-2], self.sizes[-1])
        else:
            self.linear_out = nn.Linear(self.sizes[-2], self.sizes[-1])

    def forward(self, x):
        # Pass through LSTM
        out, hidden = self.lstm0(x)

        # Linear output layer
        y = self.linear_out(out)

        if y.shape[-1] == 12:
            y = rot6_to_rotmat(y)

        return y
