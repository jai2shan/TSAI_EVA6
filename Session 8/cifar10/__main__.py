import os
os.chdir(r'C:\Users\jayasans4085\OneDrive - ARCADIS\Documents\Learning\EVA\TSAI_EVA6\Session 8')

from cifar10.dataloader import *
from cifar10.tsaiModels import *
from cifar10.utils import *


trainloader, testloader = cifar_dataloader(150)()

trainNtest = TrainTest(tsaiResNet(), trainloader, testloader)
trainNtest(1)
