import os
import sys
import numpy as np
import shutil

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report

import cv2
import csv
import argparse

current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(os.path.dirname(parent))

from traffic.traffic_video import Sign_Detector, Annotator

def standard_bbox(bbox: tuple) -> np.array:
    x1 = bbox[0][0]
    y1 = bbox[0][1]
    x2 = bbox[1][0]
    y2 = bbox[1][1]
    return np.array([x1, y1, x2, y2], np.float32)

def xyxy2xywh(x: np.array) -> np.array:
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y

def intersection_over_union(box_a, box_b):
    # Determine the coordinates of each of the two boxes
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[0]+box_a[2], box_b[0]+box_b[2])
    yB = min(box_a[1]+box_a[3], box_b[1]+box_b[3])

    # Calculate the area of the intersection area
    area_of_intersection = (xB - xA + 1) * (yB - yA + 1)

    # Calculate the area of both rectangles
    box_a_area = (box_a[2] + 1) * (box_a[3] + 1)
    box_b_area = (box_b[2] + 1) * (box_b[3] + 1)
    # Calculate the area of intersection divided by the area of union
    # Area of union = sum both areas less the area of intersection
    iou = area_of_intersection / float(box_a_area + box_b_area - area_of_intersection)

    # Return the score
    return iou

def bbox_metrics(b_true_list: list, b_list: list) -> float:
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(b_true_list)):
        b_true = xyxy2xywh(standard_bbox(b_true_list[i]))
        b = xyxy2xywh(standard_bbox(b_list[i]))

        iou = intersection_over_union(b_true, b) if b.sum() != 0 else 0
        
        if b.sum() > 0 and iou >= 0.6:
            tp += 1

        elif b.sum() > 0  and iou < 0.6:
            fp +=1

        elif b.sum() == 0 and b_true.sum() != 0:
            fn += 1

        elif b.sum() == 0 and b_true.sum() == 0:
            tn +=1

    precision = tp / (tp+fp) if tp != 0 else 0
    recall = tp / (tp+fn) if tp != 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    return accuracy, precision, recall
        

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
    b, b_true = [], []
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
            b_true.append(bbox)

            frame = an.draw_bb(frame, bbox, (0, 255, 255), 3)
            cv2.putText(frame, "gt speed: {}".format(speed), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.6 , (0, 255,255), 2, cv2.LINE_AA, False)

            found, circles, sdspeed, updates = sd.detect(original, h, w, show_results = False)
            sdbbox = sd.extract_bb(circles, h, w)
            y.append(sdspeed if sdspeed is not None else 0)
            b.append(sdbbox if sdbbox is not None else ((0, 0), (0, 0)))

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
    accuracy = balanced_accuracy_score(y_true, y)
    cf_mat = confusion_matrix(y_true, y)
    report = classification_report(y_true, y)
    box_accuracy, box_precision, box_recall = bbox_metrics(b_true, b)

    print("SPEED LIMITS")
    print("Balanced accuracy: {:.2f}".format(accuracy))
    print("BBox Accuracy: {:.2f} - precision: {:.2f} - recall: {:.2f}".format(box_accuracy, box_precision, box_recall))
    print(cf_mat)
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', action='store_true', default=False, help='improve Italian Signs')
    parser.add_argument('-t', '--test', action='store_true', default=False, help='test traffic on Italian Signs')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='verbosity')
        
    opt = parser.parse_args()
    #opt.dataset = True
    opt.test = True
    if opt.test:
        test(opt.verbose)
    elif opt.dataset:
        make_dataset()



