import os
from threading import Thread
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import torch
from auto_collect_hand_landmark import label_dict_from_config_file

from hand_dataset import NeuralNetworkWithCACLoss
from utils import HandLandmarksDetector


def main():
    cam =  cv2.VideoCapture(0)
    detector = HandLandmarksDetector()
    status_text = None
    model = NeuralNetworkWithCACLoss()
    model.load_state_dict(torch.load("./test_model.pth"))
    model.eval()
    signs = label_dict_from_config_file("hand_gesture.yaml")
    while cam.isOpened():
        _,frame = cam.read()

        hand,img = detector.detectHand(frame)
        if len(hand) != 0:
            with torch.no_grad():
                hand_landmark = torch.from_numpy(np.array(hand[0],dtype=np.float32).flatten()).unsqueeze(0)
                output = torch.nn.Softmax()(model(hand_landmark))[0]
                print(output)
                output_ind = torch.argmax(output).tolist()
                if output[output_ind] > 0.8:
                    status_text = signs[output_ind]
                else:
                    status_text = signs[len(output)-1]

        else:
            status_text = None
        cv2.putText(img, status_text, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("img",img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()