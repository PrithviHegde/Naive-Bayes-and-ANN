import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import random_split
from time import time
import pandas as pd
import numpy as np

# Define the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the transforms to preprocess the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

MNISTdataset = datasets.MNIST('./data', download=True, transform=transform)
# print(len(MNISTdataset))
trainSize = int(len(MNISTdataset) * 0.66)
testSize = len(MNISTdataset) - trainSize
trainDataset, testDataset = random_split(MNISTdataset, [trainSize, testSize])

# Define the data loaders
batchSize = 128
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, shuffle=True)


def executeNN(layers=2, neurons=100, activationMethod1=nn.Tanh(),activationMethod2=nn.Tanh(),activationMethod3=nn.Tanh()):
    # Define the neural network architecture
    class Net(nn.Module):
        if layers ==2:
            #2 hidden layers
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(784, neurons) #Input layer
                self.fc2 = nn.Linear(neurons, neurons) #hidden layer
                self.fc3 = nn.Linear(neurons, neurons) #hidden layer
                self.fc4 = nn.Linear(neurons, 10) #output layer
                self.activation1 = activationMethod1
                self.activation2 = activationMethod2
                self.activation3 = activationMethod3

            def forward(self, x):
                x = x.view(-1, 784)  # Flatten the input
                x = self.activation1(self.fc1(x))
                x = self.activation2(self.fc2(x))
                x = self.activation3(self.fc3(x))
                x = self.fc4(x)
                return x
            
        elif layers == 3:
                #3 hidden layers
                def __init__(self):
                    super(Net, self).__init__()
                    self.fc1 = nn.Linear(784, neurons) #Input layer
                    self.fc2 = nn.Linear(neurons, neurons) #hidden layer
                    self.fc3 = nn.Linear(neurons, neurons) #hidden layer
                    self.fc4 = nn.Linear(neurons, neurons) #hidden layer
                    self.fc5 = nn.Linear(neurons, 10) #output layer
                    self.activation1 = activationMethod1
                    self.activation2 = activationMethod2
                    self.activation3 = activationMethod3

                def forward(self, x):
                    x = x.view(-1, 784)  # Flatten the input
                    x = self.activation1(self.fc1(x))
                    x = self.activation1(self.fc2(x))
                    x = self.activation2(self.fc3(x))
                    x = self.activation3(self.fc4(x))
                    x = self.fc5(x)
                    return x
                

    # Create the neural network
    net = Net().to(device)

    # Define the loss function and optimizer
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Train the neural network
    epochs = 10
    for epoch in range(epochs):
        runningLoss = 0.0
        runningCorrects = 0
        for (inputs, labels) in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = lossFunction(outputs, labels)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
            preds = torch.max(outputs, 1)[1]
            runningCorrects += torch.sum(preds == labels.data)
        epochLoss = runningLoss / len(trainLoader)
        epochAcc = runningCorrects.double() / len(trainLoader.dataset)
        print('Epoch [{}/{}], Loss: {:.4f}%, Accuracy: {:.4f}%'.format(epoch+1, epochs, epochLoss*100, epochAcc*100))

    # Evaluate the neural network on the test dataset
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    confusionMatrix = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix: \n', confusionMatrix)  
    accuracy = (100 *correct/total)
    print('Accuracy on the test dataset: ', accuracy, '%\n')

    return accuracy


output = []
activations = [nn.Tanh(), nn.Sigmoid(), nn.ReLU()]
# Execute all possible NNs
for i in (2,3):
    for j in (100, 150):
        for k in range(3):
            activationMethod1 = activations[k]
            activationMethod2 = activations[k]
            activationMethod3 = activations[k]
            activationMethod3 = activations[k]
            start = time()  
            accuracy = executeNN(i,j,activationMethod1,activationMethod2,activationMethod3)
            end = time()
            if i == 2:
                output.append([2, j, activationMethod1, activationMethod2,'-', str(accuracy) + '%'])
            else:
                output.append([3, j, activationMethod1, activationMethod2, activationMethod3, str(accuracy) + '%'])


#NN#13
start = time()  
accuracy = executeNN(3,100,activationMethod1=nn.Tanh(),activationMethod2=nn.Tanh(), activationMethod3=nn.Sigmoid())
end = time()
output.append([3, 100, nn.Tanh(), nn.Tanh(), nn.Sigmoid(), str(accuracy) + '%'])


#NN#14
start = time()  
accuracy = executeNN(3,100,activationMethod1=nn.Tanh(),activationMethod2=nn.Tanh(), activationMethod3=nn.ReLU())
end = time()
output.append([3, 100, nn.Tanh(), nn.Tanh(), nn.ReLU(), str(accuracy) + '%'])


#NN#15
start = time()  
accuracy = executeNN(3,100,activationMethod1=nn.Tanh(),activationMethod2=nn.ReLU(), activationMethod3=nn.Sigmoid())
end = time()
output.append([3, 100, nn.Tanh(), nn.ReLU(), nn.Sigmoid(), str(accuracy) + '%'])

output = np.array(output)
comparative = pd.DataFrame(output, columns=["No. Hidden Layer","No. Neuron","ActivationFxn1","ActivationFxn2","ActivationFxn3","Test Accuracy"])
print()
print(comparative.to_markdown())
print()

