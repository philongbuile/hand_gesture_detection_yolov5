from datetime import datetime
import os
import torch
import pandas as pd
from  torch import nn 
from torch import optim
import numpy as np
from torchmetrics import Accuracy
import torch


from pytorch_ood.utils import OODMetrics, is_known
from torch.utils.tensorboard import SummaryWriter
from models import NeuralNetwork, NeuralNetworkWithCACLoss, RandomClassifier

from utils import CustomImageDataset, EarlyStopper, label_dict_from_config_file




    

if __name__ == "__main__":
    list_label = label_dict_from_config_file("hand_gesture.yaml")
    DATA_FOLDER_PATH="./data/"
    trainset = CustomImageDataset(os.path.join(DATA_FOLDER_PATH,"landmark_train.csv"))
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=40,shuffle=True) 

    valset = CustomImageDataset(os.path.join(DATA_FOLDER_PATH,"landmark_val.csv"))
    val_loader = torch.utils.data.DataLoader(valset,batch_size=50, shuffle=True,drop_last=True)
    timestamp = datetime.now().strftime('%d-%m %H:%M')
    

    model = NeuralNetworkWithCACLoss()
    loss_function = model.cac_loss
    early_stopper = EarlyStopper(patience=30,min_delta=0.01)

    # model = NeuralNetwork()
    # # loss_function = nn.CrossEntropyLoss(label_smoothing=0.04,ignore_index=-1)
    # loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    # early_stopper = EarlyStopper(patience=30,min_delta=0.01)


    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    writer = SummaryWriter('runs/train {} {}'.format(timestamp,model.__class__.__name__))

    
    
    # add auroc score
    statistic_time = 20
    best_vloss = 1_000_000
    for epoch in range(300):

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
            
            acc_train.update(model.predict_with_known_class(inputs), labels) # for CAC

            if batch_number % statistic_time == statistic_time-1:    # print every 2000 mini-batches
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


            val_metrics.update(model.score(distances),vlabels)    #for CAC   
            
            # writer.add_graph(model,vinputs)


            known = is_known(vlabels)
            if known.any():
                acc_val.update(model.predict_with_known_class(vinputs[known]), vlabels[known]) # with CAC


        # Log the running loss averaged per batch
        # for both training and validation
        print(f"Accuracy train:{acc_train.compute().item()}, val:{acc_val.compute().item()}")
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
            model_path = f'models/model_{timestamp}_{model.__class__.__name__}_best'
            torch.save(model.state_dict(), model_path)
        if early_stopper.early_stop(avg_vloss):
            print(f"stopping at epoch {epoch}, minimum: {early_stopper.watched_metrics}")
            break

    writer.close()
    model_path = f'models/model_{timestamp}_{model.__class__.__name__}_last'
    torch.save(model.state_dict(), model_path)


    print(acc_val.compute())
