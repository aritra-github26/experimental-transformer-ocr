# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:15:54 2020

@author: himanshu.chaudhary
"""
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import time


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    
    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))



def train(model, criterion, optimizer, scheduler, dataloader, vocab_length, device):
    """
    Train the model using the provided dataloader, which loads data from HDF5 files.
    """
    model.train()  # Set the model to training mode
    total_loss = 0
    for batch, (imgs, labels_y,) in enumerate(dataloader):  # Iterate through batches of images and labels
          imgs = imgs.to(device)
          labels_y = labels_y.to(device)
    
          optimizer.zero_grad()
          output = model(imgs.float(), labels_y.long()[:, :-1])

          norm = (labels_y != 0).sum()
          loss = criterion(output.log_softmax(-1).contiguous().view(-1, vocab_length), labels_y[:, 1:].contiguous().view(-1).long()) / norm

          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
          optimizer.step()
          total_loss += (loss.item() * norm)

    return total_loss / len(dataloader), output

def evaluate(model, criterion, dataloader, vocab_length, device):
    model.eval()  # Set the model to evaluation mode
    epoch_loss = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch, (imgs, labels_y) in enumerate(dataloader):  # Iterate through batches of images and labels
            imgs = imgs.to(device)
            labels_y = labels_y.to(device)

            output = model(imgs.float(), labels_y.long()[:, :-1])
              
            norm = (labels_y != 0).sum()
            loss = criterion(output.log_softmax(-1).contiguous().view(-1, vocab_length), labels_y[:, 1:].contiguous().view(-1).long()) / norm
  
            epoch_loss += (loss.item() * norm)

    return epoch_loss / len(dataloader)

def get_memory(model, imgs):
    # Refactored get_memory for resnet50-biLSTM model: simply extract features and apply conv and lstm
    with torch.no_grad():
        features = model.get_feature(imgs)
        conv_out = model.conv(features)
        bs, c, h, w = conv_out.size()
        # Flatten spatial dimensions and permute for LSTM input: (batch, seq_len, feature)
        lstm_input = conv_out.flatten(2).permute(0, 2, 1)
        lstm_out, _ = model.lstm(lstm_input)
    return lstm_out.permute(1, 0, 2)  # Return in shape (seq_len, batch, feature) for compatibility

def single_image_inference(model, img, tokenizer, transform, device):
    '''
    Run inference on single image using greedy decoding for resnet50-biLSTM model
    '''
    model.eval()
    img = transform(img)
    imgs = img.unsqueeze(0).float().to(device)
    with torch.no_grad():
        # Forward pass through backbone, conv, and lstm
        features = model.get_feature(imgs)
        conv_out = model.conv(features)
        bs, c, h, w = conv_out.size()
        lstm_input = conv_out.flatten(2).permute(0, 2, 1)
        lstm_out, _ = model.lstm(lstm_input)
        output = model.vocab(lstm_out)  # (batch, seq_len, vocab_size)
        output = output.squeeze(0)  # (seq_len, vocab_size)
        out_indexes = []
        for i in range(output.size(0)):
            out_token = output[i].argmax().item()
            if out_token == tokenizer.chars.index('EOS'):
                break
            out_indexes.append(out_token)
    pre = tokenizer.decode(out_indexes)
    return pre

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run_epochs(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs, tokenizer, target_path, device):
    '''
    run one epoch for a model
    '''
    best_valid_loss = np.inf
    c = 0
    for epoch in range(epochs):     
        print(f'Epoch: {epoch + 1:02}', 'learning rate{}'.format(scheduler.get_last_lr()))
     
        start_time = time.time()
    
        train_loss, outputs = train(model, criterion, optimizer, scheduler, train_loader, tokenizer.vocab_size, device)
        valid_loss = evaluate(model, criterion, val_loader, tokenizer.vocab_size, device)
     
        epoch_mins, epoch_secs = epoch_time(start_time, time.time())

        c += 1
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), target_path)
            c = 0
     
        if c > 4:
            scheduler.step()
            c = 0
     
        print(f'Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {train_loss:.3f}')
        print(f'Val   Loss: {valid_loss:.3f}')    
    
    print(best_valid_loss)
