# -------------------------
#
# Create custom Long Short-term Memory model
#
# --------------------------

import torch.nn as nn


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

        return y
