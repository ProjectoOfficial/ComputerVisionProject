import os
import sys
import numpy as np
import shutil

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import cv2
import csv
import argparse

from tqdm import tqdm
from time import process_time_ns

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
        
def classification_metrics(y_true: list, y_pred: list, classes):
    cf_matrix = confusion_matrix(y_true, y_pred)
    c_tp = dict()
    c_tn = dict()
    c_fp = dict()
    c_fn = dict()

    for i, cls in enumerate(classes):
        c_tp[cls] = cf_matrix[i, i]
        c_fp[cls] = cf_matrix[:, i].sum() - c_tp[cls]
        c_fn[cls] = cf_matrix[i, :].sum() - c_tp[cls]
        c_tn[cls] = cf_matrix.sum(axis=(0,1)) - c_tp[cls] - c_fn[cls] - c_fp[cls]

    recalls = []
    precisions = []

    print("class\t\tprecision\t\trecall\t\tf1\t\tsupport")
    for cls in classes:
        support = c_tp[cls] + c_fn[cls]
        
        precision = 0 if c_tp[cls] == 0 else c_tp[cls] / (c_tp[cls] + c_fp[cls])
        precisions.append(precision)

        recall = 0 if c_tp[cls] == 0 else c_tp[cls] / (c_tp[cls] + c_fn[cls])
        recalls.append(recall)

        f1 = 0 if c_tp[cls] == 0 else (2 * c_tp[cls]) / ((2 * c_tp[cls]) + c_fp[cls] + c_fn[cls])
        print("{}\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t{}".format(cls, precision, recall, f1, support))

    print("Accuracy: {:.2f}".format(accuracy_score(y_true, y_pred)))

    print("\n")

    ap = 0
    for i in range(len(classes) - 1):
        ap += (recalls[i] + recalls[i + 1]) * precisions[i]
    print("mAP: {:.2f}".format(ap)) 

def make_dataset(isvideo: bool=False, path: str=""):
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
    filenames = os.listdir(os.path.join(current, "rawimages"))
    index = 0
    cap = None
    stop_condition = None
    if isvideo:
        cap = cv2.VideoCapture(path)
        stop_condition = cap.isOpened()
    else:
        stop_condition = not(index >= len(filenames))

    while stop_condition:
        frame = None

        if not isvideo:
            fname = filenames[index]
            if fname in gtlabels.keys():
                index += 1
                continue    
            frame = cv2.imread(os.path.join(current, "rawimages", fname))
        else:
            _, frame = cap.read()
            cap.set(cv2.CAP_PROP_FPS, 60)
            frame = cv2.resize(frame, (1280, 720))
        
        original = frame.copy()

        height, width, _ = frame.shape
        h_perc = 5
        w_perc = 50 
        h = (height * h_perc) // 100
        w = (width * w_perc) // 100
        an = Annotator(width, height)
        an.org = (20, 50)

        #frame = frame[h : round(h*3), w : , :]
        frame = cv2.line(frame, (w, h), (width, h), (255, 0, 0), 2)
        frame = cv2.line(frame, (w, h), (w, frame.shape[0] - h), (255, 0, 0), 2)
        frame = cv2.line(frame, (w, frame.shape[0] - h), (width, frame.shape[0] - h), (255, 0, 0), 2)

        speed, bbox, valid = None, None, None
        if not isvideo:
            if fname in rawlabels.keys():
                bbox = rawlabels[fname][0:4]
                bbox = ((int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])))
                speed = int(rawlabels[fname][4])
                valid = int(rawlabels[fname][5])

        if bbox is not None:
            frame = an.draw_bb(frame, bbox)
            cv2.putText(frame, "speed: {}".format(speed), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1.6 , (0, 255, 255), 3, cv2.LINE_AA, False)
        else:
            found, circles, speed, updates = sd.detect(original.copy(), h_perc, w_perc, show_results = False)
            sdbbox = sd.extract_bb(circles, h, w)

            if sdbbox is not None:
                frame = an.draw_bb(frame, sdbbox, (255, 0, 0), 2)
                cv2.putText(frame, "sd speed: {}".format(speed), (10, 150), cv2.FONT_HERSHEY_COMPLEX, 1.6 , (255,0,0), 2, cv2.LINE_AA, False)

                if bbox is None:
                    bbox = sdbbox
                    
        cv2.imshow("frame", cv2.resize(frame,(1280, 720)))
        valid = 0

        key = None
        key = cv2.waitKey(1) if isvideo else cv2.waitKey(0)
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

        index += 1

        if not isvideo:
            stop_condition = not(index >= len(filenames))
        else:
            stop_condition = cap.isOpened()
        

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

def test(show:bool=False, speed: bool=True):
    speed = not show
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

    avg_time = 0
    lower_time = 999999999999
    upper_time = 0
    filenames = os.listdir(os.path.join(current, "images"))
    for fname in tqdm(filenames, desc="testing: "):
        frame = cv2.imread(os.path.join(current, "images", fname))
        original = frame.copy()

        #START
        start_time = process_time_ns()
        height, width, _ = frame.shape
        h_perc = 5
        w_perc = 50
        h = (height * h_perc) // 100
        w = (width * w_perc) // 100
        an = Annotator(width, height)
        an.org = (20, 50)

        if show:
            #frame = frame[h : round(h*3), w : , :]
            frame = cv2.line(frame, (w, h), (width, h), (255, 0, 0), 2)
            frame = cv2.line(frame, (w, h), (w, frame.shape[0] - h), (255, 0, 0), 2)
            frame = cv2.line(frame, (w, frame.shape[0] - h), (width, frame.shape[0] - h), (255, 0, 0), 2)

        speed, bbox, valid = None, None, None
        bbox = gtlabels[fname][0:4]
        bbox = ((int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])))
        speed = int(gtlabels[fname][4])
        y_true.append(speed)
        b_true.append(bbox)

        if not speed:
            frame = an.draw_bb(frame, bbox, (0, 255, 255), 3)
            cv2.putText(frame, "gt speed: {}".format(speed), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.6 , (0, 255,255), 2, cv2.LINE_AA, False)

        found, circles, sdspeed, updates = sd.detect(original, h_perc, w_perc, show_results = False)
        sdbbox = sd.extract_bb(circles, h, w)
        y.append(sdspeed if sdbbox is not None else 0)
        b.append(sdbbox if sdbbox is not None else ((0, 0), (0, 0)))

        if speed:
            avg_time += process_time_ns() - start_time
            lower_time = process_time_ns() - start_time if process_time_ns() - start_time < lower_time else lower_time
            upper_time = process_time_ns() - start_time if process_time_ns() - start_time > upper_time else upper_time
            continue

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
    classification_metrics(y_true, y, ["10", "20", "30", "40", "50", "60", "70", "80", "90"])
    box_accuracy, box_precision, box_recall = bbox_metrics(b_true, b)

    print("SPEED LIMITS")
    print("BBox Accuracy: {:.2f} - precision: {:.2f} - recall: {:.2f}".format(box_accuracy, box_precision, box_recall))

    
    if speed:
        print("Average Speed: {:.4f}ms".format((avg_time /10**6)/len(filenames)))
        print("Upper bound Speed: {:.4f}ms".format((upper_time /10**6)))
        print("Lower bound Speed: {:.4f}ms".format((lower_time /10**6)))

def infer(path: str):
    assert path != "" and path is not None, "You must specify image path"

    frame = cv2.imread(path)
    assert frame is not None, "Your path is wrong"

    original = frame.copy()

    sd = Sign_Detector()

    height, width, _ = frame.shape
    h_perc = 5
    w_perc = 50
    h = (height * h_perc) // 100
    w = (width * w_perc) // 100
    an = Annotator(width, height)
    an.org = (20, 50)

    frame = cv2.line(frame, (w, h), (width, h), (255, 0, 0), 2)
    frame = cv2.line(frame, (w, h), (w, frame.shape[0] - h), (255, 0, 0), 2)
    frame = cv2.line(frame, (w, frame.shape[0] - h), (width, frame.shape[0] - h), (255, 0, 0), 2)

    speed, bbox, valid = None, None, None
    frame = an.draw_bb(frame, bbox, (0, 255, 255), 3)
    cv2.putText(frame, "gt speed: {}".format(speed), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.6 , (0, 255,255), 2, cv2.LINE_AA, False)

    found, circles, sdspeed, updates = sd.detect(original, h_perc, w_perc, show_results = False)
    sdbbox = sd.extract_bb(circles, h, w)

    if sdbbox is not None:
        frame = an.draw_bb(frame, sdbbox, (255, 0, 0), 2)
        cv2.putText(frame, "sd speed: {}".format(sdspeed), (10, 150), cv2.FONT_HERSHEY_COMPLEX, 1.6 , (255,0,0), 2, cv2.LINE_AA, False)
    
    cv2.imshow("frame", frame)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', action='store_true', default=False, help='add data to Italian Signs (images)')
    parser.add_argument('-i', '--infer', action='store_true', default=False, help='infer traffic on a single image')
    parser.add_argument('-p', '--path', type=str, default="", help='path to a resource (image, video)')
    parser.add_argument('-s', '--sequence', action='store_true', default=False, help='add data to Italian Signs (video)')
    parser.add_argument('-t', '--test', action='store_true', default=False, help='test traffic on Italian Signs')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='verbosity')
        
    opt = parser.parse_args()

    if opt.sequence or opt.infer:
        assert opt.path, "You have to specify the path of the video"

    #opt.dataset = True
    opt.test = True
    #opt.infer = True
    #opt.path = r"C:\Users\daniel\Documents\GitHub\ComputerVisionProject\Camera\ItalianSigns\rawimages\1492.jpg"
    if opt.test:
        test(opt.verbose)
    elif opt.dataset:
        make_dataset(opt.sequence, opt.path)
    elif opt.infer:
        infer(opt.path)
