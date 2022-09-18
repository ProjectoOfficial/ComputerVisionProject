import os
import sys
import numpy as np
import shutil

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import cv2
import csv
import argparse

current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(os.path.dirname(parent))

from traffic.traffic_video import Sign_Detector, Annotator

def make_dataset():
    # already checked labels
    gtlabels = dict()
    if not os.path.isfile(os.path.join(current, "labels", "ItalianSigns.csv")):
        f = open(os.path.join(current, 'labels', 'ItalianSigns.csv'), 'w')
        w = csv.writer(f)
        header = ["filename", "x top left", "y top left", "x bottom right",  "y bottom right", "speed limit"]
        w.writerow(header)
        f.close()
    else:
        f = open(os.path.join(current, 'labels', 'ItalianSigns.csv'), 'r')
        r = csv.reader(f)
        next(r, None) # skip header
        for row in r:
            if row != []:
                gtlabels[row[0]] = row[1:]
        f.close()

    fwrite = open(os.path.join(current, 'labels', "ItalianSigns.csv"), 'a')
    writer = csv.writer(fwrite)

    # Labels not checked
    rawlabels = dict()
    if os.path.isfile(os.path.join(current, "labels", "RawLabels.csv")):
        fread = open(os.path.join(current, 'labels', 'RawLabels.csv'), 'r')
        reader = csv.reader(fread)
        next(reader, None) # skip header
        for row in reader:
            if row != []:
                rawlabels[row[0]] = row[1:]

    sd = Sign_Detector()

    for (dirpath, dirname, filenames) in os.walk(os.path.join(current, "rawimages")):
        for fname in filenames:
            if fname in gtlabels.keys():
                continue    

            frame = cv2.imread(os.path.join(current, "rawimages", fname))
            
            height, width, _ = frame.shape
            h = height // 4
            w = width // 3
            an = Annotator(width, height)
            an.org = (20, 50)

            #frame = frame[h : round(h*3), w : , :]
            frame = cv2.line(frame, (w, h), (width, h), (255, 0, 0), 2)
            frame = cv2.line(frame, (w, h), (w, round(h*3)), (255, 0, 0), 2)
            frame = cv2.line(frame, (w, round(h*3)), (width, round(h*3)), (255, 0, 0), 2)

            speed, bbox, valid = None, None, None
            if fname in rawlabels.keys():
                bbox = rawlabels[fname][0:4]
                bbox = ((int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])))
                speed = int(rawlabels[fname][4])
                valid = int(rawlabels[fname][5])

            if bbox is not None:
                frame = an.draw_bb(frame, bbox)
                cv2.putText(frame, "speed: {}".format(speed), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1.6 , (0, 255, 255), 3, cv2.LINE_AA, False)
            else:
                found, circles, speed, updates = sd.detect(frame, h, w, show_results = False)
                sdbbox = sd.extract_bb(circles, h, w)

                if sdbbox is not None:
                    frame = an.draw_bb(frame, sdbbox, (255, 0, 0), 2)
                    cv2.putText(frame, "sd speed: {}".format(speed), (10, 150), cv2.FONT_HERSHEY_COMPLEX, 1.6 , (255,0,0), 2, cv2.LINE_AA, False)

                    if bbox is None:
                        bbox = sdbbox
                        
            cv2.imshow("frame", cv2.resize(frame,(1280, 720)))
            valid = 0

            key = cv2.waitKey(0)
            if key == ord('q'):
                sys.exit(0)
            elif key == ord('s'): # save
                valid = 1
            elif key == ord('c'): # correct
                roi = cv2.selectROI("frame", frame) # x1,y1,w,h
                print(roi)
                bbox = ((roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]))
                speed = input("please insert the correct speed: ")
                valid = 1

            if valid:
                shutil.copy(os.path.join(current, "rawimages", fname), os.path.join(current, "images", fname))
                sign_label = [fname, bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], speed]
                writer.writerow(sign_label)

            cv2.destroyAllWindows()

def remove_images_not_in_dataset():
    gtlabels = dict()
    if not os.path.isfile(os.path.join(current, "labels", "ItalianSigns.csv")):
        print("Italian Signs not found")
        sys.exit(1)
    else:
        f = open(os.path.join(current, 'labels', 'ItalianSigns.csv'), 'r')
        r = csv.reader(f)
        next(r, None) # skip header
        for row in r:
            if row != []:
                gtlabels[row[0]] = row[1:]
        f.close()

    for (dirpath, dirname, filenames) in os.walk(os.path.join(current, "images")):
        for fname in filenames:
            if fname in gtlabels.keys():
                continue

            os.remove(os.path.join(current, "images", fname))

def test(show:bool=False):
    gtlabels = dict()
    if not os.path.isfile(os.path.join(current, "labels", "ItalianSigns.csv")):
        print("Italian Signs not found")
        sys.exit(1)
    else:
        f = open(os.path.join(current, 'labels', 'ItalianSigns.csv'), 'r')
        r = csv.reader(f)
        next(r, None) # skip header
        for row in r:
            if row != []:
                gtlabels[row[0]] = row[1:]
        f.close()

    sd = Sign_Detector()
    y, y_true = [], []
    for (dirpath, dirname, filenames) in os.walk(os.path.join(current, "images")):
        for fname in filenames:
            frame = cv2.imread(os.path.join(current, "images", fname))
            original = frame.copy()

            height, width, _ = frame.shape
            h = height // 4
            w = width // 3
            an = Annotator(width, height)
            an.org = (20, 50)

            if show:
                #frame = frame[h : round(h*3), w : , :]
                frame = cv2.line(frame, (w, h), (width, h), (255, 0, 0), 2)
                frame = cv2.line(frame, (w, h), (w, round(h*3)), (255, 0, 0), 2)
                frame = cv2.line(frame, (w, round(h*3)), (width, round(h*3)), (255, 0, 0), 2)

            speed, bbox, valid = None, None, None
            bbox = gtlabels[fname][0:4]
            bbox = ((int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])))
            speed = int(gtlabels[fname][4])
            y_true.append(speed)

            frame = an.draw_bb(frame, bbox, (0, 255, 255), 3)
            cv2.putText(frame, "gt speed: {}".format(speed), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.6 , (0, 255,255), 2, cv2.LINE_AA, False)

            found, circles, sdspeed, updates = sd.detect(original, h, w, show_results = False)
            sdbbox = sd.extract_bb(circles, h, w)
            y.append(sdspeed)

            if show:
                if sdbbox is not None:
                    frame = an.draw_bb(frame, sdbbox, (255, 0, 0), 2)
                    cv2.putText(frame, "sd speed: {}".format(sdspeed), (10, 150), cv2.FONT_HERSHEY_COMPLEX, 1.6 , (255,0,0), 2, cv2.LINE_AA, False)
            
            if show:
                cv2.imshow("frame", frame)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

    y_true = np.array(y_true)
    y = np.array(y)
    accuracy = accuracy_score(y_true, y)
    cf_mat = confusion_matrix(y_true, y)
    report = classification_report(y_true, y)

    print("SPEED LIMITS")
    print(accuracy)
    print(cf_mat)
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', action='store_true', default=False, help='improve Italian Signs')
    parser.add_argument('-t', '--test', action='store_true', default=False, help='test traffic on Italian Signs')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='verbosity')
        
    opt = parser.parse_args()
    opt.dataset = True
    #opt.test = True
    if opt.test:
        test(opt.verbose)
    elif opt.dataset:
        make_dataset()



