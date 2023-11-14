%matplotlib inline
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
#from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
class Dataset(object):
    def __init__(self):
        pass
    
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self,index):
        raise NotImplementedError
        
class DataLoader(object):
    def __init__(self,dataset,batch_size,collate_fn = None,shuffle = True,drop_last = False):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.sampler =torch.utils.data.RandomSampler if shuffle else \
           torch.utils.data.SequentialSampler
        self.batch_sampler = torch.utils.data.BatchSampler
        self.sample_iter = self.batch_sampler(
            self.sampler(self.dataset),
            batch_size = batch_size,drop_last = drop_last)
        self.collate_fn = collate_fn if collate_fn is not None else \
            torch.utils.data._utils.collate.default_collate
        
    def __next__(self):
        indices = next(iter(self.sample_iter))
        batch = self.collate_fn([self.dataset[i] for i in indices])
        return batch
    
    def __iter__(self):
        return self

class Config():
    training_dir = "./dataset/tra/"
    testing_dir = "./dataset/tes/"
    train_batch_size = 64
    train_number_epochs = 50
