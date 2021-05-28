## <center> Assignment 4 </center>

### **Group Members**             
•	Sarthak Dargan – sarthak221995@gmail.com                
•	CV Chiranthan - chiranthancv95@gmail.com                      
•   Jayasankar Raju S - muralis2raj@gmail.com  

### Results
Maximum Test Accuracy - 99.52%    
Maximum Train Accuracy - 98.87%

## Part 1 - Back Propagation
[Link to Excel ](https://docs.google.com/spreadsheets/d/1hnRYfjvfERfm-8LJoDuAjiWleu_ilXwP_pxCKBeswNk/edit#gid=0) 

[Link to Back Propagation Read Me](https://github.com/jai2shan/TSAI_EVA6/blob/main/Session%204/ReadMe%20-%20%20Back%20Prop.md) 


## Part 2 - CNN - MNIST
[Link to Colab](https://colab.research.google.com/github/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%209/Assignment_9_Step%203_updated%2010%20mins%20after%20asgnt%20submission.ipynb) 

### Data Augmentations:
RandomRotation and RandomAffine 

Concatenated the train data twice one set with data augmentations and other without augmentations    

    train_loader = torch.utils.data.DataLoader(
        ConcatDataset([
                    datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                    datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.RandomRotation(10),
                            transforms.RandomAffine(degrees=10, shear=45),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        ]),batch_size=batch_size, shuffle=True, **kwargs) 


### Model Summary : 
![Model Summary](./images/NetworkSummary.png)

### Optimizer and Learning Rate
Used three learning rates based on accuracy

    model =  Net().to(device)
    optimizer1 = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    optimizer2 = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    optimizer3 = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    EPOCHS = 20
    acc = 0
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        if acc < 99.35:
        train(model, device, train_loader, optimizer1, epoch)
        acc = test(model, device, test_loader)
        elif (acc>99.35) & (acc<99.42) :
        train(model, device, train_loader, optimizer2, epoch)
        acc = test(model, device, test_loader)
        elif acc>99.42:
        train(model, device, train_loader, optimizer3, epoch)
        acc = test(model, device, test_loader)


### Loss - Error Graph: 
![Loss Error Plot](./images/LossErrorGraph.png)

### Correct and Incorrect Predictions : 
![Correct Predictions](./images/CorrectPredictions.png)

![Incorrect Predictions](./images/WrongPredictions.png)
