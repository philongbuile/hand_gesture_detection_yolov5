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
        self.cac_loss = CACLoss(len(list_label),magnitude=1,alpha=1)
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        distances = self.cac_loss.calculate_distances(logits)
        return distances
    def predict(self,x,threshold=0.01):
        distances = self(x)
        softmin = torch.nn.Softmin(dim=1)
        rejection_scores = (distances * (1-softmin(distances)))
        chosen_ind = torch.argmin(rejection_scores,dim=1)
        return torch.where(rejection_scores[0,chosen_ind]<threshold,chosen_ind,-1)

    def predict_with_known_class(self,x):
        distances = self(x)
        softmin = torch.nn.Softmin(dim=1)
        rejection_scores = (distances * (1-softmin(distances)))
        return torch.argmin(rejection_scores,dim=1)
    def score(self,distances):
        return self.cac_loss.score(distances)
        
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
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(128, len(list_label)),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    def predict(self,x,threshold=0.8):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        chosen_ind = torch.argmax(softmax_prob,dim=1)
        return torch.where(softmax_prob[0,chosen_ind]>threshold,chosen_ind,-1)
    def predict_with_known_class(self,x):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        return torch.argmax(softmax_prob,dim=1)
    def score(self,logits):
        return -torch.amax(logits,dim=1)

class RandomClassifier(nn.Module):
    def __init__(self):
        super(RandomClassifier, self).__init__()
        list_label = label_dict_from_config_file("hand_gesture.yaml")
        self.num_class = len(list_label)
        self.dummy_stuff =  nn.Linear(63, 128)
        
    def forward(self, x):
        self.eval()
        return torch.rand(x.shape[0],self.num_class)
    def predict(self,x,threshold=0.8):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        chosen_ind = torch.argmax(softmax_prob,dim=1).tolist()
        return torch.where(softmax_prob[chosen_ind]>threshold,chosen_ind,-1)
    def predict_with_known_class(self,x):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        return torch.argmax(softmax_prob,dim=1)
    def score(self,logits):
        return -torch.amax(logits,dim=1)