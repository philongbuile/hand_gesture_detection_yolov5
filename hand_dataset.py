from datetime import datetime
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from  torch import nn 
from torch import optim
import torchmetrics
import numpy as np
from torchmetrics import Accuracy
import yaml
import torch


from pytorch_ood.detector import EnergyBased
from pytorch_ood.utils import OODMetrics, ToUnknown, is_known
from pytorch_ood.loss import CACLoss
from torch.utils.tensorboard import SummaryWriter
from models import NeuralNetwork, NeuralNetworkWithCACLoss

from utils import label_dict_from_config_file
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



    

if __name__ == "__main__":
    list_label = label_dict_from_config_file("hand_gesture.yaml")
    DATA_FOLDER_PATH="./data/"
    trainset = CustomImageDataset(os.path.join(DATA_FOLDER_PATH,"landmark_train.csv"))
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=30,shuffle=True,drop_last = True) 

    valset = CustomImageDataset(os.path.join(DATA_FOLDER_PATH,"landmark_val.csv"))
    val_loader = torch.utils.data.DataLoader(valset,batch_size=50, shuffle=False,drop_last=True)
    timestamp = datetime.now().strftime('%d-%m %H:%M')
    

    model = NeuralNetworkWithCACLoss()
    loss_function = model.cac_loss

    # model = NeuralNetwork()
    # loss_function = nn.CrossEntropyLoss(label_smoothing=0.04,ignore_index=-1)
    # loss_function = nn.CrossEntropyLoss()


    optimizer = optim.Adam(model.parameters(),lr=0.0001)

    writer = SummaryWriter('runs/train {} {}'.format(timestamp,model.__class__.__name__))

    
    # for Out of dist detection
    
    # add auroc score
    statistic_time = 20
    best_vloss = 1_000_000
    for epoch in range(200):

        #training step
        running_loss = 0.0
        last_loss = 0.0
        model.train(True)
        acc_train = Accuracy(num_classes=len(list_label))
        for batch_number,data in enumerate(trainloader):
            inputs,labels = data
            optimizer.zero_grad()
            distances = model(inputs)
            loss = loss_function(distances,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            acc_train.update(distances.min(dim=1).indices, labels)
            # acc_train.update(distances.max(dim=1).indices, labels)
            if batch_number % statistic_time == statistic_time-1:    # print every 2000 mini-batches
                # writer.add_scalar("Loss/train",running_loss/statistic_time,epoch*len(trainloader)+batch_number+1)
                last_loss = running_loss/statistic_time
                running_loss = 0.0

        # validating step       
        model.train(False)         # why adding this decrease val loss?????
        running_vloss = 0.0
        acc_val = Accuracy(num_classes=len(list_label))
        val_metrics = OODMetrics()
        # don't tell me it still have some leftover grad from train, or it will push this loss to train?
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            distances = model(vinputs)
            vloss = loss_function(distances, vlabels)
            running_vloss += vloss


            #for CAC
            val_metrics.update(loss_function.score(distances),vlabels)
            #for normal softmax
            # val_metrics.update(-torch.amax(distances,dim=1),vlabels)
            
            #change code here?
            known = is_known(vlabels)
            if known.any():
                acc_val.update(distances[known].min(dim=1).indices, vlabels[known])
                # acc_val.update(distances[known].max(dim=1).indices, vlabels[known])


        # Log the running loss averaged per batch
        # for both training and validation
        print(f"Accuracy train:{acc_val.compute().item()}, val:{acc_train.compute().item()}")
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(last_loss, avg_vloss))
        writer.add_scalars("Out of distribution metrics",val_metrics.compute(),epoch+1)
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : last_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.add_scalars('Training vs. Validation accuracy',
                        { 'Training' : acc_train.compute().item()
                        , 'Validation' : acc_val.compute().item() },
                        epoch + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'models/model_{timestamp}_{model.__class__.__name__}'
            torch.save(model.state_dict(), model_path)

    writer.close()
    model_path = f'models/model_{timestamp}_{model.__class__.__name__}_last'
    torch.save(model.state_dict(), model_path)

