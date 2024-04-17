import torch
import torch.nn as nn

class CustomMSELoss(nn.Module):
    def __init__(self, exponent, mean = True):
        super(CustomMSELoss, self).__init__()
        self.exponent = exponent
        self.mean = mean
    
    def forward(self, output, target):
        error = target - output
        loss = torch.mean(torch.pow(torch.abs(error), self.exponent))
        if self.mean:
          return torch.mean(loss)
        return loss
