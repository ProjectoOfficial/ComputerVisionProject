import os
import sys
import platform

current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(parent)

import torch

from Backbones import Resnet
from BDDDataset import BDDDataset
from utils import collate_fn

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision import ops

from tensorboardX import SummaryWriter

from AnchorBoxesGenerator import AnchorBoxesGenerator
from FasterRCNN import FasterRCNN, FastRCNNPredictor
from plot import plot_map, plot_loss_and_lr
import matplotlib.pyplot as plt
from train_utils import *

def show_image(img, labels):
    img = img.swapaxes(0,1)
    img = img.swapaxes(1,2)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')

    for idx in range(labels['boxes'].shape[0]):
        bbox = labels['boxes'][idx]
        class_name = BDD_CLASSES[labels['labels'][idx]]
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,edgecolor='red', linewidth=3.5))
        ax.text(bbox[0], bbox[1], '{:s}'.format(class_name), bbox=dict(facecolor='blue', alpha=0.5), fontsize=14, color='white')

    plt.show()
    plt.close()



if __name__ == '__main__':

    if platform.system() == "Linux":
        DATA_DIR = os.path.dirname(os.path.abspath(__file__)).replace("/faster_rcnn","/data/bdd100k")
    elif platform.system() == "Windows":
        DATA_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\faster_rcnn","\\data\\bdd100k")

    BDD_CLASSES = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
    BATCH_SIZE = 2
    DEVICE = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Data path: {}".format(DATA_DIR))
    print("Fetching data...")

    data_transform = transforms.Compose([transforms.ToTensor()])

    trainset = BDDDataset(DATA_DIR, transforms=data_transform, flag='train', label_list=BDD_CLASSES)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    
    valset = BDDDataset(DATA_DIR, transforms=data_transform, flag='train', label_list=BDD_CLASSES)
    valloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    #print(next(iter(trainloader)))

    # anchor configuration
    anchor_size = [64, 128, 256]
    anchor_ratio = [0.5, 1, 2.0]
    
    # roi configuration
    roi_out_size = [7, 7]
    roi_sample_rate = 2

    # FasterRCNN configuration
    in_channels = 3
    num_classes = 10

    train_horizon_flip_prob = 0.0  # data horizon flip probility in train transform
    min_size = 800
    max_size = 1000
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    # rpn process parameters
    rpn_pre_nms_top_n_train = 2000
    rpn_post_nms_top_n_train = 2000

    rpn_pre_nms_top_n_test = 1000
    rpn_post_nms_top_n_test = 1000

    rpn_nms_thresh = 0.7
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5

    # remove low threshold target
    box_score_thresh = 0.05
    box_nms_thresh = 0.5
    box_detections_per_img = 100
    box_fg_iou_thresh = 0.5
    box_bg_iou_thresh = 0.5
    box_batch_size_per_image = 512
    box_positive_fraction = 0.25
    bbox_reg_weights = None

    resume = ''  # pretrained_weights
    start_epoch = 0  # start epoch
    num_epochs = 10  # train epochs

    # learning rate parameters
    lr = 5e-3
    momentum = 0.9
    weight_decay = 0.0005

    # learning rate schedule
    lr_gamma = 0.33
    lr_dec_step_size = 100

    batch_size = 1

    num_class = 10  # foreground + 1 background
    data_root_dir = " "
    model_save_dir = os.path.dirname(os.path.abspath(__file__))
    
    writer = SummaryWriter(os.path.join(model_save_dir, 'epoch_log'))

    anchor_sizes = tuple((f,) for f in anchor_size)
    aspect_ratios = tuple((f,) for f in anchor_ratio) * len(anchor_sizes)
    anchor_generator = AnchorBoxesGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    roi_pooler = ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=roi_out_size, sampling_ratio=roi_sample_rate)

    backbone = Resnet(in_channels, num_classes)

    model = FasterRCNN(backbone=backbone, num_classes=num_classes,
                           # transform parameters
                           min_size=min_size, max_size=max_size,
                           image_mean=image_mean, image_std=image_std,
                           # rpn parameters
                           rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
                           rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                           rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                           rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                           rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
                           rpn_nms_thresh=rpn_nms_thresh,
                           rpn_fg_iou_thresh=rpn_fg_iou_thresh,
                           rpn_bg_iou_thresh=rpn_bg_iou_thresh,
                           rpn_batch_size_per_image=rpn_batch_size_per_image,
                           rpn_positive_fraction=rpn_positive_fraction,
                           # Box parameters
                           box_head=None, box_predictor=None,

                           # remove low threshold target
                           box_score_thresh=box_score_thresh,
                           box_nms_thresh=box_nms_thresh,
                           box_detections_per_img=box_detections_per_img,
                           box_fg_iou_thresh=box_fg_iou_thresh,
                           box_bg_iou_thresh=box_bg_iou_thresh,
                           box_batch_size_per_image=box_batch_size_per_image,
                           box_positive_fraction=box_positive_fraction,
                           bbox_reg_weights=bbox_reg_weights
                           )
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_dec_step_size, gamma=lr_gamma)

    # train from pretrained weights
    if resume != "":
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(start_epoch))

    train_loss = []
    learning_rate = []
    train_mAP_list = []
    val_mAP = []

    best_mAP = 0
    for epoch in range(start_epoch, num_epochs):
        loss_dict, total_loss = train_one_epoch(model, optimizer, trainloader,
                                                DEVICE, epoch, train_loss=train_loss, train_lr=learning_rate,
                                                print_freq=50, warmup=False)

        lr_scheduler.step()

        '''
        print("------>Starting training data valid")
        _, train_mAP = evaluate(model, trainloader, device=DEVICE, mAP_list=train_mAP_list)

        print("------>Starting validation data valid")
        _, mAP = evaluate(model, valloader, device=DEVICE, mAP_list=val_mAP)
        print('training mAp is {}'.format(train_mAP))
        print('validation mAp is {}'.format(mAP))
        print('best mAp is {}'.format(best_mAP))

        board_info = {'lr': optimizer.param_groups[0]['lr'],
                      'train_mAP': train_mAP,
                      'val_mAP': mAP}

        for k, v in loss_dict.items():
            board_info[k] = v.item()
        board_info['total loss'] = total_loss.item()
        write_tb(writer, epoch, board_info)

        if mAP > best_mAP:
            best_mAP = mAP
            # save weights
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            model_save_dir = model_save_dir
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save(save_files,
                       os.path.join(model_save_dir, "{}-model-{}-mAp-{}.pth".format(backbone, epoch, mAP)))
        '''
    writer.close()
    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate, model_save_dir)

    # plot mAP curve
    if len(val_mAP) != 0:
        plot_map(val_mAP, model_save_dir)
