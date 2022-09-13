import argparse
import os
import sys
import csv
import cv2
import torch
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from Camera.RTCamera import RTCamera
from Geometry import Geometry
from Distance import Distance
from traffic.traffic_video import Sign_Detector, Annotator
from Tracking import Tracking
from pynput.keyboard import Listener

from tqdm import tqdm
from threading import Thread
from Models.YOLOv7 import yolo_test
from Models.YOLOv7.BDDDataset import BDDDataset
from Preprocessing import Preprocessing
from Models.YOLOv7.utils.general import check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from Models.YOLOv7.utils.metrics import ap_per_class, ConfusionMatrix
from Models.YOLOv7.utils.plots import plot_images, output_to_target
from Models.YOLOv7.utils.torch_utils import select_device, time_synchronized

PRESSED_KEY = ''
RECORDING = False
BLUR = False
TRANSFORMS = False
CHESSBOARD = False
ROTATION = None

def on_press(key):
    global PRESSED_KEY
    if hasattr(key, 'char'):
        if key.char is not None:
            if key.char in "qrgescibtf": # add here a letter if you want to insert a new command
                PRESSED_KEY = key.char


listener = Listener(on_press=on_press)
preprocessor = Preprocessing((640, 640))

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='main.py')

    # input choice
    parser.add_argument('-d', '--dataset', action='store_true', help='the source is bdd100k')
    parser.add_argument('-c', '--camera', action='store_true', help='the source is the camera')

    # required parameters
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='YOLOv7 batch-size')
    parser.add_argument('-ct', '--conf-thres', type=float, default=0.001, help='YOLOv7 conf threshold')
    parser.add_argument('-dev', '--device', type=str, default='0', help='cuda device(s)')
    parser.add_argument('-it', '--iou-thres', type=float, default=0.65, help='YOLOv7 iou threshold')
    parser.add_argument('-n', '--name', type=str, default='test_dir', help='YOLOv7 result test directory name')
    parser.add_argument('-p', '--project', type=str, default=os.path.join(parent, 'Models', 'YOLOv7', 'runs', 'test') , help='YOLOv7 project save directory')
    parser.add_argument('-sh', '--save-hybrid', action='store_true', default=False, help='YOLOv7 save hybrid')
    parser.add_argument('-st', '--save-txt', action='store_true', default=False, help='YOLOv7 save txt')
    parser.add_argument('-w', '--weights', type=str, default=os.path.join(parent, 'Models', 'YOLOv7', 'last.pt'),
                        help='YOLOv7 weights')

    # yolo_test parameters
    parser.add_argument('-a', '--augment', action='store_true', help='augmented inference')
    parser.add_argument('-comp', '--compute-loss', default=None, help='')
    parser.add_argument('-dt', '--data', type=str, default=os.path.join(current, 'data', 'bdd100k'), help='*.data path')
    parser.add_argument('-ok', '--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('-hy', '--hyp', type=str, default=os.path.join(current, 'data', 'hyp.scratch.p5.yaml'), help='')
    parser.add_argument('-iw', '--image-weights', type=bool, default=False, help='')
    parser.add_argument('-is', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('-nt', '--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('-plt', '--plots', action='store_true', help='')
    parser.add_argument('-sc', '--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('-sj', '--save-json', default=True, action='store_true', help='save a compatible JSON results file')
    parser.add_argument('-sng', '--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('-str', '--stride', type=int, default=32, help='')
    parser.add_argument('-t', '--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('-v', '--verbose', default=True, action='store_true', help='report mAP by class')
    parser.add_argument('-wr', '--workers', type=int, default=6, help='')

    # camera parameters
    parser.add_argument('-cal', '--calibrate', action='store_true', default=False, help='true if you want to calibrate the camera')
    parser.add_argument('-cd', '--camera-device', type=int, default=0, help='Camera device ID')
    parser.add_argument('-f', '--filename', type=str, default='out', help='filename for recordings')
    parser.add_argument('-j', '--jetson', action='store_true', default=False, help='true if you are using the Nvidia Jetson Nano')
    parser.add_argument('-l', '--label', action='store_true', default=False, help='true if you want to save labelled signs')
    parser.add_argument('-r', '--resolution', type=tuple, default=(1280, 720), help='camera resolution')
    parser.add_argument('-rt', '--rotate', action='store_true', default=False, help='rotate frame for e-con camera')
    parser.add_argument('-s', '--save-sign', action='store_true', default=False, help='save frames which contain signs')


    opt = parser.parse_args()

    print(opt)

    assert opt.dataset is not None or opt.camera is not None, 'specify if you want to use bdd100k dataset or the camera'

    if opt.dataset:

        data_size = (1280, 720)
        preprocessor = Preprocessing((opt.img_size, opt.img_size))
        valset = BDDDataset(opt.data, opt.task, opt.hyp, data_size, preprocessor=preprocessor, mosaic=False,
                            augment=False, rect=True, image_weights=opt.image_weights, stride=opt.stride,
                            batch_size=opt.batch_size, concat_coco_names=False)
        valloader = torch.utils.data.DataLoader(valset, opt.batch_size, collate_fn=BDDDataset.collate_fn,
                                                num_workers=opt.workers)

        # Set save directory
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=False))  # increment run
        (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        tester = yolo_test.Test(opt.weights, opt.batch_size, opt.device, save_dir)

        seen = 0
        confusion_matrix = ConfusionMatrix(nc=tester.nc)
        names = {k: v for k, v in
                 enumerate(tester.model.names if hasattr(tester.model, 'names') else tester.model.module.names)}

        iouv = torch.linspace(0.5, 0.95, 10).to(tester.device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        loss = torch.zeros(3, device=tester.device)
        jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(valloader, desc=s)):
            targets = targets.to(tester.device)
            nb, _, height, width = img.shape  # batch size, channels, height, width

            with torch.no_grad():
                t = time_synchronized()
                out, train_out = tester.predict(img)  # inference and training outputs
                t0 += time_synchronized() - t

                if opt.compute_loss:
                    loss += opt.compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

                # Run NMS
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(tester.device)  # to pixels
                lb = [targets[targets[:, 0] == i, 1:] for i in
                      range(nb)] if opt.save_hybrid else []  # for autolabelling
                t = time_synchronized()
                out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=lb,
                                          multi_label=True)
                t1 += time_synchronized() - t

            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path = Path(paths[si])
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Append to text file
                if opt.save_txt:
                    gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                    for *xyxy, conf, cls in predn.tolist():
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # Append to pycocotools JSON dictionary
                if opt.save_json:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                    box = xyxy2xywh(predn[:, :4])  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for p, b in zip(pred.tolist(), box.tolist()):
                        jdict.append({'image_id': image_id,
                                      'category_id': int(p[5]),
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=tester.device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                    if opt.plots:
                        confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # opt.iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # Plot images
            if opt.plots and batch_i < 3:
                f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
                Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
                f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
                Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=opt.plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=tester.nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if (opt.verbose or tester.nc < 50) and tester.nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (opt.img_size, opt.img_size, opt.batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

        if opt.plots:
            confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

        if opt.save_json and len(jdict):
            w = Path(opt.weights[0] if isinstance(opt.weights,
                                                  list) else opt.weights).stem if opt.weights is not None else ''  # opt.weights
            pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json

        # Return results
        tester.model.float()  # for training
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if opt.save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        maps = np.zeros(tester.nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]

    if opt.camera:

        opt.save_txt |= opt.save_hybrid

        if opt.rotate:
            ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE

        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=False))  # increment run
        (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        if not os.path.isdir(os.path.join(current, "signs")):
            os.makedirs(os.path.join(current, "signs"))

        if not os.path.isdir(os.path.join(current, "Calibration")):
            os.makedirs(os.path.join(current, "Calibration"))

        if not os.path.isdir(os.path.join(current, "Recordings")):
            os.makedirs(os.path.join(current, "Recordings"))

        if not os.path.isdir(os.path.join(current, "ItalianSigns")):
            os.makedirs(os.path.join(current, "ItalianSigns"))

        if not os.path.isdir(os.path.join(current, "ItalianSigns", 'images')):
            os.makedirs(os.path.join(current, "ItalianSigns", 'images'))

        if not os.path.isdir(os.path.join(current, "ItalianSigns", 'labels')):
            os.makedirs(os.path.join(current, "ItalianSigns", 'labels'))

        if not os.path.isfile(os.path.join(current, "ItalianSigns", 'labels', 'ItalianSigns.csv')):
            f = open(os.path.join(current, "ItalianSigns", 'labels', 'ItalianSigns.csv'), 'w')
            writer = csv.writer(f)
            writer.writerow(
                ["filename", "x top left", "y top left", "x bottom right", "y bottom right", "speed limit", "valid"])
            f.close()

        camera = RTCamera(opt.camera_device, fps=30, resolution=opt.resolution, cuda=True, auto_exposure=False,
                          rotation=ROTATION)
        camera.start()

        start_fps = time.monotonic()
        fps = 0
        listener.start()

        sd = Sign_Detector()
        an = Annotator(*opt.resolution)
        an.org = (20, 50)
        circles = None
        speed = 0
        updates = 0

        if opt.calibrate:
            geometry = Geometry(os.path.join(current, 'Calibration'))
            calibrated, mtx, dist, rvecs, tvecs = geometry.get_calibration()
            camera.calibrate(calibrated, mtx, dist, rvecs, tvecs)

        tester = None
        names = None
        tracker = None

        if not opt.jetson:
            tester = yolo_test.Test(opt.weights, opt.batch_size, opt.device, save_dir)
            names = tester.model.names
            tracker = Tracking()

        label_file = None
        label_writer = None
        if opt.label:
            label_file = open(os.path.join(current, "ItalianSigns", 'labels', 'ItalianSigns.csv'), 'a')
            label_writer = csv.writer(label_file)

        # Main infinite loop
        while True:
            frame = camera.get_frame()

            if camera.available():
                original = frame.copy()

                if time.monotonic() - start_fps > 1:
                    fps = camera.get_fps()
                    start_fps = time.monotonic()

                if PRESSED_KEY == 'q':  # QUIT
                    if label_file is not None:
                        label_file.close()
                    listener.stop()
                    listener.join()
                    print("closing!")
                    break

                elif PRESSED_KEY == 'r':  # REGISTER/STOP RECORDING
                    if not RECORDING:
                        print("recording started...")
                        camera.register(os.path.join(current, "Recordings", "{}__{}.mp4".format(opt.filename,
                                                                                                datetime.now().strftime(
                                                                                                    "%d_%m_%Y__%H_%M_%S"))))
                        RECORDING = True
                    else:
                        camera.stop_recording()
                        print("recording stopped!")
                        RECORDING = False

                elif PRESSED_KEY == 'g' and not RECORDING:  # CHANGE GAIN
                    gain = int(input("please insert the gain: "))
                    camera.set_gain(gain)

                elif PRESSED_KEY == 'e' and not RECORDING:  # CHANGE EXPOSURE
                    exp = int(input("please insert the exposure: "))
                    camera.set_exposure(exp)

                elif PRESSED_KEY == 's' and not RECORDING:  # SAVE CURRENT FRAME
                    path = os.path.join(current, 'Calibration',
                                        'frame_{}.jpg'.format(datetime.now().strftime("%d_%m_%Y__%H_%M_%S")))
                    camera.save_frame(path)

                    print("saved frame {} ".format(path))

                elif PRESSED_KEY == 'c' and not RECORDING:  # CALIBRATE CAMERA
                    print("Calibration in process, please wait...\n")
                    cv2.destroyAllWindows()
                    geometry = Geometry(os.path.join(current, 'Calibration'))
                    calibrated, mtx, dist, rvecs, tvecs = geometry.get_calibration()
                    camera.calibrate(calibrated, mtx, dist, rvecs, tvecs)

                elif PRESSED_KEY == 'i':  # SHOW MEAN VALUE OF CURRENT FRAME
                    print("Frame AVG value: {}".format(frame.mean(axis=(0, 1, 2))))

                elif PRESSED_KEY == 'b':  # BLUR FRAME
                    BLUR = not BLUR
                    print("blur: {}".format(BLUR))

                elif PRESSED_KEY == 't' and not RECORDING:  # APPLY TRANSFORMS TO FRAME
                    TRANSFORMS = not TRANSFORMS
                    print("transform: {}".format(TRANSFORMS))

                elif PRESSED_KEY == 'f' and not RECORDING:  # SHOW CHESSBOARD
                    CHESSBOARD = not CHESSBOARD
                    print("Chessboard: {}".format(CHESSBOARD))
                    cv2.destroyAllWindows()

                if BLUR:
                    frame = preprocessor.GaussianBlur(frame, 1)

                if TRANSFORMS:
                    (frame, _) = preprocessor.Transform_base(frame)

                if CHESSBOARD:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray, (7, 9),
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
                    if ret:
                        cv2.drawChessboardCorners(frame, (7, 9), corners, ret)

                if PRESSED_KEY != '':
                    PRESSED_KEY = ''

                if not opt.jetson:
                    # Object Recognition
                    img, _ = preprocessor.Transform_base(frame)
                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(dim=0)
                    out, train_out = tester.predict(img)
                    out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, multi_label=True)

                    detections = []
                    for si, pred in enumerate(out):
                        predn = pred.clone()
                        ratio = ((1, 1), (0, 0))
                        scale_coords(img.shape[1:], predn[:, :4], (640, 640), ratio)  # native-space pred

                        for *xyxy, conf, cls in predn.tolist():
                            if conf > 0.7:
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1)  # xywh
                                xywh = [int(x) for x in xywh]
                                x, y, w, h = xywh

                                detections.append((cls, xywh))
                                distance = Distance().get_Distance(xywh)
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 0, 255), 2)
                                cv2.circle(frame, (x + (w // 2), y + (h // 2)), 4, (40, 55, 255), 4)
                                cv2.putText(frame, "{:.2f} {} {:.2f}".format(conf, names[int(cls)], distance),
                                            (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 255), 1)

                    # Tracking
                    hsvframe = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    tracker.zero_objects()
                    for cls, box in detections:
                        x, y, w, h = box
                        box[0] = int(box[0] + box[2] / 2)
                        box[1] = int(box[1] + box[3] / 2)
                        id = tracker.update_obj(cls, box)

                        prediction, pts = tracker.track(hsvframe, box)
                        cv2.putText(frame, "ID: {}".format(id), (x - 60, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                    (255, 0, 255), 1)
                        cv2.putText(frame, "ID: {}".format(id),
                                    (int(prediction[0] - (0.5 * w)) + 5, int(prediction[1] - (0.5 * h)) + 20),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 255), 1)
                        cv2.rectangle(frame, (int(prediction[0] - (0.5 * w)), int(prediction[1] - (0.5 * h))),
                                      (int(prediction[0] + (0.5 * w)), int(prediction[1] + (0.5 * h))), (0, 255, 0), 2)
                    tracker.clear_objects()

                # traffic sign detection
                height, width, _ = frame.shape
                h = height // 4
                w = width // 3
                found, c, s, u = sd.detect(frame, h, w, show_results=False)
                if found and s != 0:
                    circles, speed, updates = c, s, u

                    if circles is not None:
                        sign_bb = sd.extract_bb(circles, h, w)
                        frame = an.draw_bb(frame, sign_bb)

                        if opt.label:
                            fname = 'frame_{}.jpg'.format(datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
                            fpath = os.path.join(current, "ItalianSigns", 'images', fname)
                            if not os.path.isfile(fpath):
                                saved = cv2.imwrite(fpath, original)
                                if saved:
                                    sign_label = [fname, sign_bb[0][0], sign_bb[0][1], sign_bb[1][0], sign_bb[1][1],
                                                  speed, 1]
                                    label_writer.writerow(sign_label)

                    if opt.save_sign:
                        path = os.path.join(current, 'signs',
                                            'sign_{}.jpg'.format(datetime.now().strftime("%d_%m_%Y__%H_%M_%S")))
                        cv2.imwrite(path, frame)

                an.write(frame, speed, updates)
                cv2.putText(frame, str(fps) + " fps", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2,
                            cv2.LINE_AA)
                cv2.imshow("frame", frame)

        camera.stop()
        cv2.destroyAllWindows()
        print("closed")



