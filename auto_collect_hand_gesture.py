import os
from threading import Thread
import uuid
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import requests
from requests_toolbelt import MultipartEncoder
# from numpy.random import choice
import random
import io
from PIL import Image
from pascal_voc_writer import Writer


MY_KEY= os.environ['ROBOFLOW_KEY']

def upload_image(src_img,name,split):
    image = Image.fromarray(src_img).convert("RGB")
    # Convert to JPEG Buffer
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format="JPEG")


    # Construct the URL
    upload_url = "".join([
        "https://api.roboflow.com/dataset/asl-letter/upload",
        "?api_key=" + MY_KEY,
        f"&name={name}",
        f"&split={split}"
    ])

    m = MultipartEncoder(fields={'file': (name, buffered.getvalue(), "image/jpeg")})
    r = requests.post(upload_url, data=m, headers={'Content-Type': m.content_type})

    # Output result
    print(r.status_code)
    id = r.json()['id']
    return id

def upload_annotation(content,annotation_filename,image_id):
    # Read Annotation as String
    # Construct the URL
    upload_url = "".join([
        f"https://api.roboflow.com/dataset/asl-letter/annotate/{image_id}",
        "?api_key=" + MY_KEY,
        "&name=", annotation_filename
    ])

    # POST to the API
    r = requests.post(upload_url, data=content, headers={
        "Content-Type": "text/plain"
    })


def send_annotation(label,src_image,x,y,w,h):
    img_name = f"{label}_{uuid.uuid1()}.jpg"
    annotation_name = f"{label}.xml"               
    split = random.choices(population=["train","valid","test"],weights=[0.8,0.1,0.1],k=1)[0]
    image_id = upload_image(src_image,img_name,split)
    height,width = src_image.shape[0],src_image.shape[1]
    writer = Writer(img_name,width,height)
    writer.addObject(label,x,y,x+w,y+h)
    annotation_content = writer.print()
    upload_annotation(annotation_content,annotation_name,image_id)
    print(f"upload img and annotation successfully to {split}")

def is_handsign_character(char:str):
    return ord('a') <= ord(char) <=ord("z") or char == " "


def main():
    cam =  cv2.VideoCapture(0)
    detector = HandDetector()
    current_letter= None
    status_text = None
    cannot_switch_char = False


    while cam.isOpened():
        _,frame = cam.read()
        hand,img = detector.findHands(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        rgb_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        if(current_letter is None):
            status_text = "press a character to record"
        else:
            status_text = "Recording {letter}, press again to stop".format(letter= current_letter)
        
        key = cv2.waitKey(1)
        # not pressing any key, push the data to roboflow if current letter is not ''
        if(key == -1):
            if(current_letter is None ):
                # no current letter recording, just skip it
                pass
            else:
                # push to roboflow x,y,w,h,current_letter,current image.
                if len(hand) != 0:
                    x,y,w,h = hand[0]["bbox"]
                    #padding, because the current bbox only surround/circumscribe the landmarks, which may leave out a small portion of fingertips!
                    x = np.clip(x-20,0,None)
                    y = np.clip(y-20,0,None)
                    w = np.clip(w+40,0,rgb_image.shape[1])
                    h = np.clip(h+40,0,rgb_image.shape[0])
                    thread = Thread(target=send_annotation,args=(current_letter,rgb_image,x,y,w,h))
                    thread.start()

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
        cv2.imshow("img",cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()