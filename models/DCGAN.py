import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec

def from_pth_file(filename):
    '''
    Instantiate from a pth file.
    '''
    state_dict = torch.load(filename)
    # if 'state_dict' in state_dict:
    #     state_dict = state_dict['state_dict']
    # # Convert old version of parameter names
    # if 'features.0.conv.weight' in state_dict:
    #     state_dict = state_dict_from_old_pt_dict(state_dict)
    # sizes = sizes_from_state_dict(state_dict)
    result = DCGAN_Generator()
    result.load_state_dict(state_dict)
    return result
class DCGAN_Generator(torch.nn.Module):
  def __init__(self, noise_dim=100, output_channels=3):
        super(DCGAN_Generator, self).__init__()    
        self.noise_dim = noise_dim
        self.network = nn.Sequential(
            # input is latent vector Z
            # size of feature maps: 128
            nn.ConvTranspose2d(self.noise_dim, 128 * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128 * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(128 * 16, 128 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(128 * 8, 128 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128 * 4, 128 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128 * 2, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # output is size 3 x 128 x 128
        )
    
  def forward(self, x):
    x = self.network(x)
    return x
    

class DCGAN_Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(DCGAN_Discriminator, self).__init__()
        self.network = nn.Sequential(
            # input dim: 3 x 128 x 128
            # size of feature maps: 32
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32 * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 2, 32 * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 4, 32 * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 8, 32 * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 16, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # output is a scalar
        )
    
    def forward(self, x):
        x = self.network(x)
        return x