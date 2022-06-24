import os
import sys
import json
from pathlib import Path

current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import random

from BDDDataset import BDDDataset
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from DabDetr import DABDETR, SetCriterion, PostProcess
from backbone import Backbone, FrozenBatchNorm2d, Joiner
from position_encoding import  PositionEmbeddingSineHW
from transformer import Transformer
from segmentation import DETRsegm, PostProcessSegm
from matcher import HungarianMatcher
from logger import setup_logger

from utils import *

DATA_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\DABDETR","\data\\bdd100k")

# classes_BDD
BDD_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'bus', 'traffic light', 'traffic sign',
    'pedestrian', 'bicycle', 'truck', 'motorcycle', 'car', 'train', 'rider'
]

def show(img, labels):
    img = img.swapaxes(0,1)
    img = img.swapaxes(1,2)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')

    for idx in range(labels['boxes'].shape[0]):
        bbox = labels['boxes'][idx]
        class_name = BDD_INSTANCE_CATEGORY_NAMES[labels['labels'][idx]]
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,edgecolor='red', linewidth=3.5))
        ax.text(bbox[0], bbox[1], '{:s}'.format(class_name), bbox=dict(facecolor='blue', alpha=0.5), fontsize=14, color='white')

    plt.show()
    plt.close()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    seed = random.seed()
    
    # USING DEFAULT PARAMETERS
    num_classes                     = 11
    lr_backbone                     = 1e-5      # backbone learning rate
    lr                              = 1e-4      # learning rate
    weight_decay                    = 1e-4
    epochs                          = 50
    batch_size                      = 1
    lr_drop                         = 40
    drop_lr_now                     = True      # load checkpoint and drop for 12epoch setting
    save_checkpoint_interval        = 10
    clip_max_norm                   = 0.1       # gradient clipping max norm
    start_epoch                     = 0
    resume                          = ''        # resume from checkpoint
    eval                            = False      # eval only. w/o Training.
    pretrain_model_path             = ''        # load from other checkpoint
    finetune_ignore                 = ''        # A list of keywords to ignore when loading pretrained models. use +
    save_log                        = True      # If save the training prints to the log file.
    amp                             = True      # Train with mixed precision
    scalar                          = 5         # number of dn groups
    label_noise_scale               = 0.2       # label noise ratio to flip
    box_noise_scale                 = 0.4       # box noise scale to shift and scale

    hidden_dim                      = 256       # Size of the embedding (dimension of the transformer)
    num_select                      = 100       # the number of predictions selected for evaluation
    dropout                         = 0.0       # Dropout applied in the transformer
    nheads                          = 8         # Number of attention heads inside the transformer's attentions
    num_queries                     = 100       # Number of query slots
    dim_feedforward                 = 1024      # Intermediate size of the feedforward layers in the transformer blocks
    enc_layers                      = 6         # Number of encoding layers in the transformer
    dec_layers                      = 6         # Number of decoding layers in the transformer
    pre_norm                        = True      # Using pre-norm in the Transformer blocks.
    transformer_activation          = 'prelu'   # activation function used inside transformer
    num_patterns                    = 0         # number of pattern embeddings.

    masks                           = False     # Train segmentation head if the flag is provided
    use_dn                          = True      # use denoising training.
    frozen_weights                  = None      # Path to the pretrained model. If set, only the mask head will be trained

    cls_loss_coef                   = 1         # loss coefficient for cls
    bbox_loss_coef                  = 5         # loss coefficient for bbox L1 loss
    giou_loss_coef                  = 2         # loss coefficient for bbox GIOU loss
    mask_loss_coef                  = 1         # loss coefficient for mask
    dice_loss_coef                  = 1         # loss coefficient for dice

    set_cost_class                  = 2         # Class coefficient in the matching cost
    set_cost_bbox                   = 5         # L1 box coefficient in the matching cost
    set_cost_giou                   = 2         # giou box coefficient in the matching cost
    focal_alpha                     = 0.25      # alpha for focal loss

    N_steps                         = hidden_dim // 2
    pe_temperatureH                 = 20        # Temperature for height positional encoding.
    pe_temperatureW                 = 20        # Temperature for width positional encoding.

    aux_loss                        = False     # Disables auxiliary decoding losses (loss at each layer)
    random_refpoints_xy             = True      # Random init the x,y of anchor boxes and freeze them.

    distributed                     = False     # Set if you want to train on distribuited GPUs
    gpu                             = [0]       # gpu IDs
    find_unused_params              = True
    output_dir                      = parent
    rank                            = 0         # number of distributed processes
    num_workers                     = 10

    # modelname                       = 'dn_dab_detr' # choices=['dn_dab_detr', 'dn_dab_deformable_detr', 'dn_dab_deformable_detr_deformable_encoder_only']

    position_embedding = PositionEmbeddingSineHW(
            N_steps, 
            temperatureH=pe_temperatureH,
            temperatureW=pe_temperatureW,
            normalize=True
        )

    backbone = Backbone("resnet50", lr_backbone, True, True, batch_norm=FrozenBatchNorm2d)
    backbone_model = Joiner(backbone, position_embedding)
    backbone_model.num_channels = backbone.num_channels

    transformer = Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        num_queries=num_queries,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
        query_dim=4,
        activation=transformer_activation,
        num_patterns=num_patterns,
    )

    # BUILD MODEL MAIN
    model = DABDETR(
        backbone_model,
        transformer,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=aux_loss,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=random_refpoints_xy,
    )

    if masks:
        model = DETRsegm(model, freeze_detr=(frozen_weights is not None))

    matcher = HungarianMatcher(
        cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou,
        focal_alpha=focal_alpha
    )
    weight_dict = {'loss_ce': cls_loss_coef, 'loss_bbox': bbox_loss_coef}
    weight_dict['loss_giou'] = giou_loss_coef

    # dn loss
    if use_dn:
        weight_dict['tgt_loss_ce'] = cls_loss_coef
        weight_dict['tgt_loss_bbox'] = bbox_loss_coef
        weight_dict['tgt_loss_giou'] = giou_loss_coef

    if masks:
        weight_dict["loss_mask"] = mask_loss_coef
        weight_dict["loss_dice"] = dice_loss_coef

    # TODO this is a hack
    if aux_loss:
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=focal_alpha, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=num_select)}
    if masks:
        postprocessors['segm'] = PostProcessSegm()

    wo_class_error = False
    model.to(device)
    model_without_ddp = model

    logger = setup_logger(output=os.path.join(output_dir, 'info.txt'), distributed_rank=rank, color=False, name="DAB-DETR")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        }
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                  weight_decay=weight_decay)

    # LOAD DATA
    train_transform = transforms.Compose([transforms.ToTensor()])

    trainset = BDDDataset(DATA_DIR, transforms=train_transform, flag='train', label_list=BDD_INSTANCE_CATEGORY_NAMES)
    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    
    valset = BDDDataset(DATA_DIR, transforms=train_transform, flag='train', label_list=BDD_INSTANCE_CATEGORY_NAMES)
    valloader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    if distributed:
        sampler_train = DistributedSampler(trainset)
        sampler_val = DistributedSampler(valset, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(trainset)
        sampler_val = torch.utils.data.SequentialSampler(valset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True)

    data_loader_train = DataLoader(trainset, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=num_workers)
    data_loader_val = DataLoader(valset, batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=num_workers)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)

    if frozen_weights is not None:
        checkpoint = torch.load(frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(output_dir)
    if resume:
        if resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

            if drop_lr_now:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

    if not resume and pretrain_model_path:
        checkpoint = torch.load(pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = finetune_ignore if finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))
        # import ipdb; ipdb.set_trace()

        '''
        if eval:
            os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if output_dir:
           save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if output_dir and is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            sys.exit(1)

        '''

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        if distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model, criterion=criterion, data_loader=data_loader_train, optimizer=optimizer, device=device, epoch=epoch,
            max_norm=clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler,
            amp=True, label_noise_scale=label_noise_scale, box_noise_scale=box_noise_scale,
            num_patterns=num_patterns, debug=False, scalar=scalar, logger=(logger if save_log else None))

        if output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % lr_drop == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}_beforedrop.pth')
            for checkpoint_path in checkpoint_paths:
                save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        lr_scheduler.step()
        if output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % lr_drop == 0 or (epoch + 1) % save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     #**{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if output_dir and is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            '''
            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name)

            '''

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("Now time: {}".format(str(datetime.datetime.now())))