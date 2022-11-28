import os
import cv2
import numpy as np
import torch
from auto_collect_hand_landmark import label_dict_from_config_file

from hand_dataset import NeuralNetworkWithCACLoss
from utils import HandLandmarksDetector

def main(path):
    cam =  cv2.VideoCapture(0)
    detector = HandLandmarksDetector()
    status_text = None
    signs = label_dict_from_config_file("hand_gesture.yaml")


    model = NeuralNetworkWithCACLoss()
    model.load_state_dict(torch.load(path))
    model.eval()

    while cam.isOpened():
        _,frame = cam.read()

        hand,img = detector.detectHand(frame)
        if len(hand) != 0:
            with torch.no_grad():
                hand_landmark = torch.from_numpy(np.array(hand[0],dtype=np.float32).flatten()).unsqueeze(0)
                class_number = model.predict(hand_landmark).item()
                print(class_number)
                if class_number != -1:
                    status_text = signs[class_number]
                else:
                    status_text = "undefined command"
                    
        else:
            status_text = None
        cv2.putText(img, status_text, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("img",img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("./models/model_28-11 15:07_NeuralNetworkWithCACLoss_last")