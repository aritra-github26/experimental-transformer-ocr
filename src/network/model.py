from pathlib import Path

import torch

from itertools import groupby
import numpy as np
from torch.autograd import Variable
from torch import nn
from torchvision.models import resnet50
import torch.nn.functional as F
import math
from data import preproc as pp


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OCR(nn.Module):

    def __init__(self, vocab_len, hidden_dim, nheads,
                 num_encoder_layers, num_decoder_layers):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a BiLSTM layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)

        # prediction heads with length of vocab
        self.vocab = nn.Linear(hidden_dim * 2, vocab_len)  # Adjusted for bidirectional output

        # output positional encodings (object queries)
        self.decoder = nn.Embedding(vocab_len, hidden_dim)
        self.query_pos = PositionalEncoding(hidden_dim, .2)

        # spatial positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.trg_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def get_feature(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)   
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, inputs, trg):  # Process input images and target sequences
        """
        Forward pass through the model.
        
        Args:
            inputs: Input images (batch of images).
            trg: Target sequences (ground truth labels).
        
        Returns:
            Output predictions for the target sequences.
        """

        # Propagate inputs through ResNet-50
        x = self.get_feature(inputs)

        # Convert from 2048 to hidden_dim feature planes
        h = self.conv(x)

        # Pass through BiLSTM
        h, _ = self.lstm(h.flatten(2).permute(2, 0, 1))

        # Getting positional encoding for target
        trg = self.decoder(trg)
        trg = self.query_pos(trg)

        # Calculate output
        output = self.vocab(h)

        return output


def make_model(vocab_len, hidden_dim=256, nheads=4,
                 num_encoder_layers=4, num_decoder_layers=4):
    
    return OCR(vocab_len, hidden_dim, nheads,
                 num_encoder_layers, num_decoder_layers)
