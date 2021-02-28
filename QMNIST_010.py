#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
This Script contains the ProgressiveSpinalNet  for QMNIST code based on Spinal net code by @author: Dipu from https://github.com/dipuk0506/SpinalNet.
"""
# import Libraries
import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

# hyper parameters
num_epochs = 100
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001 * 0.8
momentum = 0.5
log_interval = 500


# In[3]:


# train and test loader
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.QMNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.RandomPerspective(), 
                               torchvision.transforms.RandomRotation(10, fill=(0,)), 
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train ,shuffle=True,num_workers=1)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.QMNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


print(example_data.shape)

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig


# In[ ]:


Half_width = 256
layer_width = 64   # hidden layer size

class SpinalVGG(nn.Module):  
    """
    Based on - https://github.com/kkweon/mnist-competition
    
    """
    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
    
    def three_conv_pool(self,in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
 
    def __init__(self, num_classes=10):
        super(SpinalVGG, self).__init__()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)  
        self.dropout = nn.Dropout(p = 0.2)       
        self.fc_spinal_layer1 = nn.Sequential(
              nn.Dropout(p = 0.5),
              nn.Linear(Half_width, layer_width),
              nn.BatchNorm1d(layer_width),
              nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
              nn.Dropout(p = 0.5), 
              nn.Linear(Half_width+ layer_width * 1, layer_width),
              nn.BatchNorm1d(layer_width), 
              nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
              nn.Dropout(p = 0.5), 
              nn.Linear(Half_width+layer_width * 2, layer_width),
              nn.BatchNorm1d(layer_width), 
              nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
              nn.Dropout(p = 0.5), 
              nn.Linear(Half_width+ layer_width * 3, layer_width),
              nn.BatchNorm1d(layer_width), 
              nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
              nn.Dropout(p = 0.5), 
              nn.Linear(Half_width + layer_width*4, num_classes),)
        
    
    def forward(self, x):
      x = self.l1(x)
      x = self.l2(x)
      x = self.l3(x)
      x = self.l4(x)
      # print("x size",x.shape)
      x = x.view(x.size(0), -1)
      # x = x.view(x.size()[0], -1)

      Xaux = self.dropout(x)

      x1 = self.fc_spinal_layer1(x)
      Xaux = torch.cat([Xaux,x1],dim = 1)

      x2 = self.fc_spinal_layer2(Xaux)
      Xaux = torch.cat([Xaux,x2], dim = 1)

      x3 = self.fc_spinal_layer3(Xaux)
      Xaux = torch.cat([Xaux,x3], dim =1)

      x4 = self.fc_spinal_layer4(Xaux)
      Xaux = torch.cat([Xaux,x4], dim = 1)

      x= self.fc_out(Xaux)
      return F.log_softmax(x, dim = 1)


# In[ ]:


import time    
device = 'cuda' 
    
num_epochs = 100

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr2 = learning_rate


# In[ ]:


model2 = SpinalVGG().to(device)


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import torch.optim as optim


# In[ ]:


# Specify a path
PATH_load = "/content/drive/MyDrive/Colab Notebooks/ml2/qmnist/qmnist_009.pt"
PATH = "/content/drive/MyDrive/Colab Notebooks/ml2/qmnist/qmnist_010.pt"


# In[ ]:


# Load
model2 = torch.load(PATH_load)


# In[ ]:


# Load
model = torch.load(PATH_load)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0 
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)    
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Test Accuracy of SpinalNet: {} % '.format(100 * correct / total)) 


# In[ ]:


num_epochs = 100


# In[ ]:


#Train and Test VGG + Spnial FC


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate) 
#optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.001,momentum=0.9)
  
# Train the model
total_step = len(train_loader)


best_accuracy2 =0
since = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)        
        outputs = model2(images)
        loss2 = criterion(outputs, labels)

        # Backward and optimize
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        if i % 500 == 0:           
            print ("Spinal Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss2.item()))        
    
    # Test the model

    model2.eval()
    with torch.no_grad():

        correct2 = 0
        total2 = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            
            outputs = model2(images)
            _, predicted = torch.max(outputs.data, 1)
            total2 += labels.size(0)
            correct2 += (predicted == labels).sum().item()

            
        if best_accuracy2>= correct2 / total2:
            curr_lr2 = learning_rate*np.asscalar(pow(np.random.rand(1),3))
            update_lr(optimizer2, curr_lr2)
            print('Test Accuracy of SpinalNet: {} % Best: {} %'.format(100 * correct2 / total2, 100*best_accuracy2))
        else:
            best_accuracy2 = correct2 / total2
            net_opt2 = model2
            print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct2 / total2))
            torch.save(model2, PATH)

        model2.train()
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# Save
#torch.save(model2, PATH)


# In[ ]:


get_ipython().system('rm /content/drive/MyDrive/Colab\\ Notebooks/ml2/qmnist/qmnist_009.pt')
get_ipython().system('cp /content/drive/MyDrive/Colab\\ Notebooks/ml2/qmnist/qmnist_010.pt /content/drive/MyDrive/Colab\\ Notebooks/ml2/qmnist/qmnist_009.pt')


# In[ ]:


#save the final model in qMNIST_010_final.pt


# In[ ]:


get_ipython().system('cp /content/drive/MyDrive/Colab\\ Notebooks/ml2/qmnist/qmnist_009.pt /content/drive/MyDrive/Colab\\ Notebooks/ml2/qmnist/qmnist_010_final.pt')


# In[ ]:




