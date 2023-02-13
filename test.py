import os
from torchmetrics import Accuracy
import torch
from models import NeuralNetwork,NeuralNetworkWithCACLoss,RandomClassifier
from utils import CustomImageDataset, label_dict_from_config_file
from pytorch_ood.utils import OODMetrics, is_known

list_label = label_dict_from_config_file("hand_gesture.yaml")
DATA_FOLDER_PATH="./data/"
testset = CustomImageDataset(os.path.join(DATA_FOLDER_PATH,"landmark_test.csv"))
test_loader = torch.utils.data.DataLoader(testset,batch_size=20,shuffle=True,drop_last = True) 


nn1 = NeuralNetworkWithCACLoss()
nn1.load_state_dict(torch.load("./models/model_05-12 12:15_NeuralNetworkWithCACLoss_best"))

nn2 = NeuralNetwork()
nn2.load_state_dict(torch.load("./models/model_05-12 13:21_NeuralNetwork_best"))
models = [nn1,nn2,RandomClassifier()]




for model in models:
    model.eval()
    acc_val = Accuracy(num_classes=len(list_label))
    test_metrics = OODMetrics()

    for i, test_data in enumerate(test_loader):
        test_input, test_label = test_data
        distances = model(test_input)
        test_metrics.update(model.score(distances),test_label)    #for CAC   
        # print(test_label)        
        known = is_known(test_label)
        # print(test_label[known],model.predict_with_known_class(test_input[known]))   
        if known.any():
            acc_val.update(model.predict_with_known_class(test_input[known]), test_label[known]) # with CAC
    print(model.__class__.__name__)
    print(f"Accuracy of model:{acc_val.compute().item()}")
    print(f"OOD metrics of model: {test_metrics.compute()}")
    print("========================================================================")
