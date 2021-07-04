import torch
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

def cuda_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Cuda Device : {}'.format(device))
    return device

class TrainTest:

    def __init__(self,):
        pass

    def __init__(self, model, train_loader, test_loader, opt = "SGD",L1 = False, lr = 0.001):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        self.device = cuda_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.opt = opt
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        if self.opt == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

    def train_(self):
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(self.device), target.to(self.device)

            # Init
            self.optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = self.model(data)

            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            if self.L1 == True:
                l1_crit = nn.L1Loss(size_average=False)
                reg_loss = 0
                for param in self.model.parameters():
                    reg_loss += l1_crit(param, target=torch.zeros_like(param))

                factor = 0.0005
                loss += factor * reg_loss

            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Update pbar-tqdm

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
            self.train_acc.append(100 * correct / processed)

    def test_(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        if self.opt ==  "ReduceLROnPlateau":
            self.scheduler.step(test_loss / len(self.test_loader))

        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        self.test_acc.append(100. * correct / len(self.test_loader.dataset))
        accT = 100. * correct / len(self.test_loader.dataset)
        return accT

    def __call__(self, epochs):
        for epoch in range(epochs):
            print("EPOCH:", epoch)
            self.train_()
            self.test_()

        print('Finished Training')


class UnNormalize(object):
    def __init__(self, data):
        self.mean = data.mean
        self.std = data.std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def Misclassified_Images(model, test_loader,device = "cuda"):
  model.eval()
  revnorm = UnNormalize()
  test_loss = 0
  correct = 0
  im_pred = {'Correct': [] ,
            'Wrong': []}
  i = 1
  plt_dt = dict()
  with torch.no_grad():
      for data, target in test_loader:
        if (len(im_pred['Correct'])<21) |  (len(im_pred['Wrong'])<21):
          data, target = data.to(device), target.to(device)
          output = model(data)
          test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(target.view_as(pred)).sum().item()
          i+=1
          plt_dt['Input'], plt_dt['target'], plt_dt['pred'] = revnorm(data.to('cpu')), target.to('cpu'), pred.to('cpu').view(-1,)

          for id in range(len(data)):
            if plt_dt['target'][id] == plt_dt['pred'][id]:
              if (len(im_pred['Correct'])<5):
                im_pred['Correct'] = im_pred['Correct']+ [{'Image':data[id],'pred':pred[id],'actual' : target[id]}]
            else:
              if (len(im_pred['Wrong'])<5):
                im_pred['Wrong'] = im_pred['Wrong']+ [{'Image':data[id],'pred':pred[id],'actual' : target[id]}]

  return im_pred

def plot_Misclassified(im_pred):
  plt.figure(figsize=(16,16))

  for i in range(len(im_pred['Correct'])):
    plt.subplot(1,5,i+1)
    label_ = data.classes[im_pred['Correct'][i]['actual'].cpu()]
    pred_ = data.classes[im_pred['Correct'][i]['pred'].cpu()[0]]
    # Plot
    plt.title('Actual Value is {label}\n Predicted Value is {pred}'.format(label=label_, pred =pred_),  color='b')
    plt.imshow(im_pred['Correct'][i]['Image'].cpu().permute(1, 2, 0))


  plt.show()

  plt.figure(figsize=(16,16))

  for i in range(len(im_pred['Wrong'])):
    plt.subplot(1,5,i+1)
    label_ = data.classes[im_pred['Wrong'][i]['actual'].cpu()]
    pred_ = data.classes[im_pred['Wrong'][i]['pred'].cpu()[0]]
    # Plot
    plt.title('Actual Value is {label}\n Predicted Value is {pred}'.format(label=label_, pred =pred_), color='r')
    plt.imshow(im_pred['Wrong'][i]['Image'].cpu().permute(1, 2, 0))

  plt.show()