import os
import time
import cv2
import numpy as np
import torch
from auto_collect_hand_landmark import label_dict_from_config_file

from models import NeuralNetwork, NeuralNetworkWithCACLoss
from physical import ModbusMaster
from utils import HandLandmarksDetector

def main(model,path):
    cam =  cv2.VideoCapture(0)
    cam.set(3,1280)
    cam.set(4,720)
    detector = HandLandmarksDetector()
    status_text = None
    signs = label_dict_from_config_file("hand_gesture.yaml")

    controller = ModbusMaster()

    light1 = False
    light2 = False
    light3 = False

    model.load_state_dict(torch.load(path))
    model.eval()

    while cam.isOpened():
        _,frame = cam.read()

        hand,img = detector.detectHand(frame)
        if len(hand) != 0:
            with torch.no_grad():
                hand_landmark = torch.from_numpy(np.array(hand[0],dtype=np.float32).flatten()).unsqueeze(0)
                class_number = model.predict(hand_landmark).item()
                if class_number != -1:
                    status_text = signs[class_number]

                    if status_text == "light1":
                        if light1 == False:
                            print("lights on")
                            light1=True
                            controller.switch_actuator_1(True)
                    elif status_text == "light2":
                        if light2 == False:
                            light2=True
                            controller.switch_actuator_2(True)
                    elif status_text == "light3":
                        if light3 == False:
                            light3=True
                            print("cool")
                            controller.switch_actuator_3(True)                    
                    elif status_text == "all shutdown":
                        if not light1 and not light2 and not light3:
                            #print("all lights are off?")
                            pass
                        else:
                            light1 = light2 = light3 = False
                            controller.switch_actuator_1(light1)
                            time.sleep(0.03)
                            controller.switch_actuator_2(light2)
                            time.sleep(0.03)
                            controller.switch_actuator_3(light3)
                            
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
    # model = NeuralNetworkWithCACLoss()
    # main(model,"./models/model_28-11 15:07_NeuralNetworkWithCACLoss_last")

    model = NeuralNetwork()
    main(model,"./models/model_05-12 11:26_NeuralNetwork_best")