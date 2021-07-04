from cifar10.dataloader import *
from cifar10.models import *
from cifar10.utils import *


trainloader, testloader = cifar_dataloader(250)()

trainNtest = TrainTest(ResNet18(), trainloader, testloader)
trainNtest.train_(1)