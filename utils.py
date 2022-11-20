import csv
import numpy as np
import yaml


def is_handsign_character(char:str):
    return ord('a') <= ord(char) <=ord("z") or char == " "
def label_dict_from_config_file(relative_path):
    with open(relative_path,"r") as f:
       label_tag = yaml.full_load(f)["gestures"]
    return label_tag
class HandDatasetWriter():
    def __init__(self) -> None:
        self.csv_file = open("landmark.csv","a")
        self.file_writer = csv.writer(self.csv_file,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    def add(self,hand,label):
        self.file_writer.writerow([label,*np.array(hand).flatten().tolist()])
    def close(self):
        self.csv_file.close()