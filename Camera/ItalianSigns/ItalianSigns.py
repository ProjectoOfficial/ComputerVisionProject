from fileinput import filename
from genericpath import isfile
import os
import sys
from pathlib import Path

import cv2
import csv

current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(os.path.dirname(parent))

from traffic.traffic_video import Sign_Detector, Annotator

if __name__ == "__main__":
    labels = dict()
    if os.path.isfile(os.path.join(current, "labels", "ItalianSigns.csv")):
        flabels = open(os.path.join(current, 'labels', 'ItalianSigns.csv'), 'r')
        reader = csv.reader(flabels)
        next(reader, None) # skip header
        for row in reader:
            if row != []:
                labels[row[0]] = row[1:]

    sd = Sign_Detector()

    for (dirpath, dirname, filenames) in os.walk(os.path.join(current, "images")):
        for fname in filenames:    
            frame = cv2.imread(os.path.join(current, "images", fname))
            
            height, width, _ = frame.shape
            h = height // 4
            w = width // 10*4 #~40%
            an = Annotator(width, height)
            an.org = (20, 50)

            frame = cv2.line(frame, (w, 0), (w, height), (255, 0, 0), 2)

            speed, bbox, valid = None, None, None
            if fname in labels.keys():
                bbox = labels[fname][0:4]
                bbox = ((int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])))
                speed = int(labels[fname][4])
                valid = int(labels[fname][5])
            else:
                found, circles, speed, updates = sd.detect(frame, h, w, show_results = False)
                bbox = sd.extract_bb(circles, h, w)

            if bbox is not None:
                frame = an.draw_bb(frame, bbox)
                cv2.putText(frame, "speed: {}".format(speed), (10,40), cv2.FONT_HERSHEY_COMPLEX, 1.6 , (255,0,0), 3, cv2.LINE_AA, False)

            cv2.imshow("frame", frame)
            if cv2.waitKey(0) == ord('q'):
                sys.exit(0)
            cv2.destroyAllWindows()