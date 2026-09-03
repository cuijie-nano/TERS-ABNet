# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):

    def __init__(
        self,
        x_channels,
        q_channels=64,
        attention_type="sigmoid"
    ):
        super(AttentionGate, self).__init__()

        self.attention_type = attention_type

        self.conv_x = nn.Conv2d(
            x_channels,
            64,
            kernel_size=3,
            padding=1
        )

        self.conv_q = nn.Conv2d(
            q_channels,
            64,
            kernel_size=3,
            padding=1
        )

        self.conv_out = nn.Conv2d(
            64,
            1,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, q):

        x_conv = self.relu(self.conv_x(x))

        q_upsampled = F.interpolate(
            q,
            size=x.size()[2:],
            mode='bilinear',
            align_corners=False
        )

        q_conv = self.relu(self.conv_q(q_upsampled))

        attention = self.conv_out(x_conv + q_conv)

        if self.attention_type == "sigmoid":

            attention_weights = torch.sigmoid(attention)
            attention_weights = (
                0.1 + 0.9 * attention_weights
            )

        elif self.attention_type == "softmax":

            b, c, h, w = attention.shape

            attention_weights = attention.view(
                b, c, -1
            )

            attention_weights = F.softmax(
                attention_weights,
                dim=2
            )

            attention_weights = attention_weights.view(
                b, c, h, w
            )

        else:
            raise ValueError(
                "attention_type must be 'sigmoid' or 'softmax'"
            )

        return x * attention_weights