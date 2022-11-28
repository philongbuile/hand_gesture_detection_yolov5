from utils import label_dict_from_config_file
from torch import nn
import torch
from pytorch_ood.loss import CACLoss


class NeuralNetworkWithCACLoss(nn.Module):
    def __init__(self):
        super(NeuralNetworkWithCACLoss, self).__init__()
        self.flatten = nn.Flatten()
        list_label = label_dict_from_config_file("hand_gesture.yaml")
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(128, len(list_label)),
        )
        self.cac_loss = CACLoss(len(list_label),alpha=2,fixed=False)
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        distances = self.cac_loss.calculate_distances(logits)
        return distances
    def predict(self,x,threshold=0.01):
        distances = self(x)
        softmin = torch.nn.Softmin(dim=1)
        rejection_scores = (distances * (1-softmin(distances)))
        print(rejection_scores)
        chosen_ind = torch.argmin(rejection_scores,dim=1)
        print(chosen_ind)
        return torch.where(rejection_scores[0,chosen_ind]<threshold,chosen_ind,-1)


        
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        list_label = label_dict_from_config_file("hand_gesture.yaml")
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.23),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.23),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(128, len(list_label)),
        )
        self.softmax_ce_loss = nn.Softmax()
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    def predict(self,x,threshold=0.8):
        logits = self(x)
        softmax_prob = nn.Softmax(logits,dim=1)
        chosen_ind = torch.argmax(softmax_prob).tolist()
        return torch.where(softmax_prob[chosen_ind]>threshold,chosen_ind,-1)
        # if softmax_prob[chosen_ind] > threshold:
        #     return chosen_ind
        # else:
        #     return -1