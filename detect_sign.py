import os
from threading import Thread
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import torch
from auto_collect_hand_landmark import label_dict_from_config_file

from hand_dataset import NeuralNetwork


def main():
    cam =  cv2.VideoCapture(0)
    cam.set(3,1920)
    cam.set(4,1080)
    detector = HandDetector(True,maxHands=2)
    status_text = None
    model = NeuralNetwork()
    model.load_state_dict(torch.load("./test_model.pth"))
    model.eval()
    signs = label_dict_from_config_file("hand_gesture.yaml")
    while cam.isOpened():
        _,frame = cam.read()

        hand,img = detector.findHands(frame)
        if len(hand) != 0:
            # print("hand:",hand)
            with torch.no_grad():
                hand_landmark = torch.from_numpy(np.array(hand[0]["lmList"],dtype=np.float32).flatten()).unsqueeze(0)
                output = torch.nn.Softmax()(model(hand_landmark))
                output = torch.argmax(output).tolist()
                status_text = signs[output]
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