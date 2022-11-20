import torch
import pandas as pd
from torch.utils.data import Dataset
from  torch import nn 
from torch import optim
import numpy as np
import yaml
class CustomImageDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        # self.labels = torch.nn.functional.one_hot(torch.from_numpy(self.data.iloc[:,0].apply( lambda char:ord(char)-97).to_numpy()))
        self.labels = torch.from_numpy(self.data.iloc[:,0].to_numpy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one_hot_label = self.labels[idx]
        torch_data = torch.from_numpy(self.data.iloc[idx,1:].to_numpy(dtype=np.float32))
        return torch_data, one_hot_label


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        with open("hand_gesture.yaml","r") as f:
            list_label = yaml.full_load(f)["gestures"]
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),            
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, len(list_label)),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    trainset = CustomImageDataset("./landmark.csv")
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=30,shuffle=True)
    model = NeuralNetwork()
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    # loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    statistic_time = 50
    for epoch in range(200):
        running_loss = 0.0
        for i,data in enumerate(trainloader):
            inputs,labels = data
            # print(inputs,labels)
            optimizer.zero_grad()
            logits = model(inputs)
            # outputs = nn.Softmax()(logits)
            outputs = logits
            loss = loss_function(outputs,labels)
            outputs = nn.Softmax()(logits)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % statistic_time == statistic_time-1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/statistic_time :.3f}')
                # print(outputs,labels)
                running_loss = 0.0

    PATH = './test_model.pth'
    torch.save(model.state_dict(), PATH)

