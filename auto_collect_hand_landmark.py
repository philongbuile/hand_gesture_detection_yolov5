import cv2


import numpy as np
import csv
import yaml
from utils import HandDatasetWriter, HandLandmarksDetector, is_handsign_character, label_dict_from_config_file

def main(split="val",resolution=(1280,720)):

    hand_detector = HandLandmarksDetector()
    label_tag = label_dict_from_config_file("hand_gesture.yaml")
    cam =  cv2.VideoCapture(0)
    cam.set(3,resolution[0])
    cam.set(4,resolution[1])


    dataset_path = f"./data/landmark_{split}.csv"
    hand_dataset = HandDatasetWriter(dataset_path)
    current_letter= None
    status_text = None
    cannot_switch_char = False


    saved_frame = None
    while cam.isOpened():
        _,frame = cam.read()

        hands,annotated_image = hand_detector.detectHand(frame)
        if(current_letter is None):
            status_text = "press a character to record"
        else:
            label =  ord(current_letter)-ord("a")
            if label == -65:
                status_text = f"Recording unknown, press spacebar again to stop"
                label = -1
            else:
                status_text = f"Recording {label_tag[label]}, press {current_letter} again to stop"
        
        key = cv2.waitKey(1)
        # not pressing any key, push the data to roboflow if current letter is not ''
        if(key == -1):
            if(current_letter is None ):
                # no current letter recording, just skip it
                pass
            else:
                if len(hands) != 0:
                    hand = hands[0]
                    hand_dataset.add(hand=hand,label=label)
                    saved_frame = frame
        # some key is pressed
        else:
            # pressed some key, do not push this image, assign current letter to the key just pressed
            key = chr(key)
            if (is_handsign_character(key)):
                if(current_letter is None):
                    current_letter = key
                elif(current_letter == key):
                    # pressed again?, reset the current state
                    if saved_frame is not None:
                        if label >=0:
                            cv2.imwrite(f"./sign_imgs/{label_tag[label]}.jpg",saved_frame)

                    cannot_switch_char=False
                    current_letter = None
                    saved_frame = None
                else:
                    cannot_switch_char = True
                    # warned user to unbind the current_letter first
        if(cannot_switch_char):
            cv2.putText(annotated_image, f"please press {current_letter} again to unbind", (0,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(annotated_image, status_text, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(f"{split}",annotated_image)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("train",(1280,720))
    # main("val",(1280,720))
    # main("test",(1280,720))