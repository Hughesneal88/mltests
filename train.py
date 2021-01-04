import torch
#import torch.nn.Fucntional as F
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import os
import random
from matplotlib import pyplot as plt
import string
from PIL import Image
dataset_path = 'datasets/'
cifar10_train = datasets.CIFAR10(dataset_path, train=True, download=True)
cifar10 = datasets.CIFAR10(dataset_path, train=False, download=True)
alphabets = []
for i in string.ascii_letters:
    alphabets.append(i)
def gen_random_name(image):
    """gen_random_name: This function takes the image as input and generates a random name for it"""
    image_name = random.choice(alphabets) + random.choice(alphabets) + random.choice(alphabets) + random.choice(alphabets) + random.choice(alphabets) + random.choice(alphabets) + '.jpg'
    with open(image_name, 'wb') as f:
        f.write(image)
img = Image.open(os.path.join(os.cwd+"uploads/image_name"))

preprocess = transforms.Compose([
    transforms.resize(25),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )])

batch_t = torch.unsqueeze(img_t, 0)
img_t = preprocess(img)

img, label = cifar10[99]
plt.imshow(img)
plt.show()

class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()


    def forward(self, inputs, outputs, batch_size):
        pass





