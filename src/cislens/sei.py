"""SEI convolutional encoder; unused classifier and spline head are omitted.

See THIRD_PARTY_NOTICES.md for upstream provenance.
"""

import torch.nn as nn


class SeiEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=9, padding=4),
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )
        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 640, kernel_size=9, padding=4),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )
        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 960, kernel_size=9, padding=4),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )
        self.dconv1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(960, 960, kernel_size=5, dilation=2, padding=4),
            nn.ReLU(inplace=True),
        )
        self.dconv2 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(960, 960, kernel_size=5, dilation=4, padding=8),
            nn.ReLU(inplace=True),
        )
        self.dconv3 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(960, 960, kernel_size=5, dilation=8, padding=16),
            nn.ReLU(inplace=True),
        )
        self.dconv4 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(960, 960, kernel_size=5, dilation=16, padding=32),
            nn.ReLU(inplace=True),
        )
        self.dconv5 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(960, 960, kernel_size=5, dilation=25, padding=50),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward propagation of a batch."""
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)
        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)
        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)
        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        dconv_out5 = self.dconv5(cat_out4)
        out = cat_out4 + dconv_out5
        return out.mean(dim=-1)
