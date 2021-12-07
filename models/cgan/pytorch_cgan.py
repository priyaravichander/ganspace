'''
Code from https://github.com/arturml/mnist-cgan repo
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()


def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).cuda())
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).cuda())
    
    # train with fake images
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).cuda())
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()



if __name__ == '__main__':
  generator = Generator().cuda()
  discriminator = Discriminator().cuda()

  criterion = nn.BCELoss()
  d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
  g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

  writer = SummaryWriter()

  num_epochs = 50
  n_critic = 5
  display_step = 50
  for epoch in range(num_epochs):
      print('Starting epoch {}...'.format(epoch), end=' ')
      for i, (images, labels) in enumerate(data_loader):
          step = epoch * len(data_loader) + i + 1
          real_images = Variable(images).cuda()
          labels = Variable(labels).cuda()
          generator.train()
        
          d_loss = discriminator_train_step(len(real_images), discriminator,
                                            generator, d_optimizer, criterion,
                                            real_images, labels)
        
          g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)
        
          writer.add_scalars('scalars', {'g_loss': g_loss, 'd_loss': d_loss}, step)  
        
          if step % display_step == 0:
              generator.eval()
              z = Variable(torch.randn(9, 100)).cuda()
              labels = Variable(torch.LongTensor(np.arange(9))).cuda()
              sample_images = generator(z, labels).unsqueeze(1)
              grid = make_grid(sample_images, nrow=3, normalize=True)
              writer.add_image('sample_image', grid, step)
      print('Done!')
















