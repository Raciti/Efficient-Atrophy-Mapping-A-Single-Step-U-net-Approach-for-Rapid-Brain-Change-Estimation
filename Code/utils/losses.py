import torch
import torch.nn as nn
import warnings
#import tensorflow.keras.backend as K

class CustomMSELoss(nn.Module):
    def __init__(self, exponent, reduction = "mean"):
        super(CustomMSELoss, self).__init__()
        self.exponent = exponent
        self.reduction = reduction
    
    def forward(self, output, target):
        error = target - output
        loss = torch.mean(torch.pow(torch.abs(error), self.exponent))
        if self.reduction == "mean":
          return torch.mean(loss)
        elif self.reduction == "sum":
          return torch.sum(loss)
        elif self.reduction == "max":
          return torch.max(loss)
        else:
          warnings.warn("The value of self.reduction is not valid. Returning None.")
          return None
