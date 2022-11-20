import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import csv
import yaml

from utils import HandDatasetWriter, is_handsign_character, label_dict_from_config_file



def main(split="val",resolution=(1280,720)):
    label_tag = label_dict_from_config_file("hand_gesture.yaml")
    cam =  cv2.VideoCapture(0)
    detector = HandDetector(True,maxHands=2)
    hand_dataset = HandDatasetWriter()
    current_letter= None
    status_text = None
    cannot_switch_char = False


    while cam.isOpened():
        _,frame = cam.read()

        hand,img = detector.findHands(frame)
        # rgb_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        if(current_letter is None):
            status_text = "press a character to record"
        else:
            label =  ord(current_letter)-ord("a")
            status_text = "Recording {letter}, press again to stop".format(letter= label_tag[label])
        
        key = cv2.waitKey(1)
        # not pressing any key, push the data to roboflow if current letter is not ''
        if(key == -1):
            if(current_letter is None ):
                # no current letter recording, just skip it
                pass
            else:
                if len(hand) != 0:
                    hand_dataset.add(hand=hand[0]["lmList"],label=label)

        # some key is pressed
        else:
            # pressed some key, do not push this image, assign current letter to the key just pressed
            key = chr(key)
            if (is_handsign_character(key)):
                if(current_letter is None):
                    current_letter = key
                elif(current_letter == key):
                    # pressed again?, reset the current state
                    cannot_switch_char=False
                    current_letter = None
                else:
                    cannot_switch_char = True
                    # warned user to unbind the current_letter first
        if(cannot_switch_char):
            cv2.putText(img, f"please press {current_letter} again to unbind", (0,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(img, status_text, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("img",img)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()