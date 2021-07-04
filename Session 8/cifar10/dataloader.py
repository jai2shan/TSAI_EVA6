# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:20:46 2021

@author: jayasans4085
"""

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

BatchSize=250


def cutout(mask_size=16, p=1, cutout_inside=False, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout


class cifar_dataloader:
    def __init__(self, BatchSize):
        self.mean = 0
        self.std = 0
        self.BatchSize = BatchSize
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __call__(self):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True,transform=transforms.Compose([transforms.ToTensor()]))
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True,
                                       transform=transforms.Compose([transforms.ToTensor()]))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.BatchSize,
                                          shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.BatchSize,
                                         shuffle=False, num_workers=2)

        print("Calculating mean and std of data")
        for images, _ in trainloader:
            batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            self.mean += images.mean(2).sum(0)
            self.std += images.std(2).sum(0)

        for images, _ in testloader:
            batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            self.mean += images.mean(2).sum(0)
            self.std += images.std(2).sum(0)

        self.mean /= (len(testloader.dataset)+len(trainloader.dataset))
        self.std /= (len(testloader.dataset)+len(trainloader.dataset))

        self.mean = tuple(np.array(self.mean))
        self.std = tuple(np.array(self.std))
        print("Mean and std are calculated")

        transform_params = dict()
        transform_params['train'] = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              #  transforms.Grayscale(num_output_channels=3),
                                               transforms.RandomCrop(32, padding=4),
                                               transforms.RandomRotation(5),
                                               transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                               transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                       saturation=0.2),
                                               cutout(mask_size = 16),
                                               transforms.ToTensor(),
                                               transforms.Normalize(self.mean, self.std)])

        transform_params['test'] = transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize(self.mean, self.std)])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True,transform = transform_params['train'])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform = transform_params['test'])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.BatchSize,
                                                    shuffle=True, num_workers=2)

        testloader = torch.utils.data.DataLoader(testset, batch_size=self.BatchSize,
                                                    shuffle=False, num_workers=2)

        self.View_images(trainloader)
        return trainloader, testloader

    def View_images(self, trainloader):
        # functions to show an image

        def imshow(img):
            img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            # print(label)
            plt.title('classes : {}'.format(' '.join('%5s' % self.classes[labels[j]] for j in range(4))))
            plt.imshow(np.transpose(npimg, (1, 2, 0)))


        # get some random training images
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        class_ = [self.classes[i] for i in labels[:4].tolist()]
        # show images
        imshow(torchvision.utils.make_grid(images[:4]))
        # print labels
        # print(' '.join('%5s' % self.classes[labels[j]] for j in range(4)))

