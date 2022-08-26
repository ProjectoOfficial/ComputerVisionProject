import argparse
import logging
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread, local

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))  
parent = os.path.dirname(current)
sys.path.append(str(Path(parent).parent))

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from Preprocessing import Preprocessing

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, check_dataset, check_file, check_img_size, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.plots import plot_images, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger
from BDDDataset import BDDDataset

logger = logging.getLogger(__name__)


def train(settings: dict):
    adam = settings['ADAM']
    artifact_alias = settings['ARTIFACT_ALIAS']
    batch_size = settings['BATCH_SIZE']
    bbox_interval = settings['BBOX_INTERVAL']
    bucket = settings['BUCKET']
    cache_images = settings['CACHE_IMAGES']
    cfg = settings['CFG']
    data_dir = settings['DATA_DIR']
    device = settings['DEVICE']
    entity = settings['ENTITY']
    epochs = settings['EPOCHS']
    evolve = settings['EVOLVE']
    exist_ok = settings['EXIST_OK']
    hyp = settings['HYP']
    image_weights = settings['IMAGE_WEIGHTS']
    img_size = settings['IMG_SIZE']
    label_smoothing = settings['LABEL_SMOOTHING']
    linear_lr = settings['LINEAR_LR']
    local_rank = settings['LOCAL_RANK']
    multi_scale = settings['MULTI_SCALE']
    name = settings['NAME']
    noautoanchor = settings['NOAUTOANCHOR']
    nosave = settings['NO_SAVE']
    no_test = settings['NO_TEST']
    project = settings['PROJECT']
    quad = settings['QUAD']
    rank = settings['GLOBAL_RANK'] 
    rect = settings['RECT']
    resume = settings['RESUME']
    save_dir = settings['SAVE_DIR']
    save_period = settings['SAVE_PERIOD']
    single_cls = settings['SINGLE_CLS']
    stride = settings['STRIDE']
    sync_bn = settings['SYNC_BN']
    use_coco_labels = settings['USE_COCO_LABELS']
    upload_dataset = settings['UPLOAD_DATASET']
    weights = settings['WEIGHTS']
    workers = settings['WORKERS']
    world_size = settings['WORLD_SIZE']
    

    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Directories
    wdir = Path(os.path.join(save_dir, 'weights'))
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = Path(os.path.join(save_dir, 'last.pt'))
    best = Path(os.path.join(save_dir, 'best.pt'))
    results_file = Path(os.path.join(save_dir, 'results.txt'))

    # Save run settings

    with open(Path(os.path.join(save_dir, 'hyp.yaml')), 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    '''
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    '''

    # Configure
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)

    # TrainSet

    data_size = (1280, 720)
    preprocess = Preprocessing((640, 640))
    trainset = BDDDataset(data_dir, 'train', hyp, data_size, preprocessor=preprocess, mosaic=False, augment=False, rect=True, image_weights=image_weights, stride=stride, 
    batch_size=batch_size, concat_coco_names=use_coco_labels)
    
    nc = 1 if single_cls else len(trainset.names)
    names = ['item'] if single_cls and len(trainset.names) != 1 else trainset.names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data_dir)  # check

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    if adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), stride)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    #Trainloader
    sampler = torch.utils.data.distributed.DistributedSampler(trainset) if rank != -1 else None
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, collate_fn=BDDDataset.collate_fn, num_workers=workers, sampler=sampler)

    mlc = np.concatenate(trainset.labels, 0)[:, 0].max()  # max label class
    nb = len(trainloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g . Possible class labels are 0-%g' % (mlc, nc, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        if not resume:
            labels = np.concatenate(trainset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                #plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not noautoanchor:
                check_anchors(trainset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[int(local_rank)], output_device=int(local_rank),
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(trainset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {trainloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    torch.save(model, Path(os.path.join( wdir, 'init.pt')))
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(trainset.labels, nc=nc, class_weights=cw)  # image weights
                trainset.indices = random.choices(range(trainset.n), weights=iw, k=trainset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(trainset.indices) if rank == 0 else torch.zeros(trainset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    trainset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            trainloader.sampler.set_epoch(epoch)
        pbar = enumerate(trainloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                if rank != -1:
                    loss *= world_size  # gradient averaged between devices in DDP mode
                if quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 10:
                    f = Path(os.path.join(save_dir, 'train_batch{}.jpg'.format(ni)))  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(name) and bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, bucket, name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt, Path(os.path.join(wdir, 'best_{:03d}.pt'.format(epoch))))
                if epoch == 0:
                    torch.save(ckpt, Path(os.path.join(wdir, 'epoch_{:03d}.pt'.format(epoch))))
                elif ((epoch+1) % 25) == 0:
                    torch.save(ckpt, Path(os.path.join(wdir, 'epoch_{:03d}.pt'.format(epoch))))
                elif epoch >= (epochs-5):
                    torch.save(ckpt, Path(os.path.join(wdir, 'epoch_{:03d}.pt'.format(epoch))))
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if bucket:
            os.system(f'gsutil cp {final} gs://{bucket}/weights')  # upload
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    MULTI_GPU = False

    ADAM = True
    ARTIFACT_ALIAS = 'latest'
    BATCH_SIZE = 30 if MULTI_GPU else 8
    BBOX_INTERVAL = -1
    BUCKET = ''
    CACHE_IMAGES = False
    CFG = os.path.join(current, 'cfg', 'training', 'yolov7.yaml')
    DATA_DIR = os.path.join(current, 'data', 'bdd100k')
    DEVICE = '0, 1' if MULTI_GPU else '0'
    ENTITY = None
    EPOCHS = 2
    EVOLVE = False
    EXIST_OK = False
    HYP = os.path.join(current, 'data', 'hyp.scratch.custom.yaml')
    IMAGE_WEIGHTS = False
    IMG_SIZE = (640, 640)
    LABEL_SMOOTHING = 0.0
    LINEAR_LR = False
    LOCAL_RANK = os.environ['LOCAL_RANK'] if MULTI_GPU else -1
    MULTI_SCALE = False
    NAME = 'custom'
    NOAUTOANCHOR = True
    NO_SAVE = False
    NO_TEST = False
    PROJECT = os.path.join(current, 'runs', 'train')
    QUAD = False
    RECT = False
    RESUME = False
    SAVE_PERIOD = 1
    SINGLE_CLS = False
    STRIDE = 32
    SYNC_BN = False # True with multiple GPUs
    UPLOAD_DATASET = False
    USE_COCO_LABELS = False
    WEIGHTS = os.path.join(current, 'yolov7_training.pt')
    WORKERS = 4 if MULTI_GPU else 6

    settings = {'ADAM': ADAM, 'ARTIFACT_ALIAS': ARTIFACT_ALIAS, 'BATCH_SIZE': BATCH_SIZE, 'BBOX_INTERVAL':BBOX_INTERVAL, 'BUCKET': BUCKET, 'CACHE_IMAGES': CACHE_IMAGES, 
    'CFG': CFG, 'DATA_DIR': DATA_DIR, 'DEVICE': DEVICE, 'ENTITY': ENTITY, 'EPOCHS': EPOCHS, 'EVOLVE': EVOLVE, 'EXIST_OK': EXIST_OK, 'HYP': HYP, 'IMAGE_WEIGHTS': IMAGE_WEIGHTS,
    'IMG_SIZE': IMG_SIZE, 'LABEL_SMOOTHING': LABEL_SMOOTHING, 'LINEAR_LR': LINEAR_LR, 'LOCAL_RANK': LOCAL_RANK, 'MULTI_SCALE': MULTI_SCALE, 'NAME': NAME, 'NOAUTOANCHOR': NOAUTOANCHOR,
    'NO_SAVE': NO_SAVE, 'NO_TEST': NO_TEST, 'PROJECT': PROJECT, 'QUAD': QUAD, 'RECT': RECT, 'RESUME': RESUME, 'SAVE_PERIOD': SAVE_PERIOD, 'SINGLE_CLS': SINGLE_CLS, 'STRIDE': STRIDE,
    'SYNC_BN': SYNC_BN, 'UPLOAD_DATASET': UPLOAD_DATASET, 'USE_COCO_LABELS': USE_COCO_LABELS, 'WEIGHTS': WEIGHTS, 'WORKERS': WORKERS, }

    # Set DDP variables
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(global_rank)
    #if opt.global_rank in [-1, 0]:
    #    check_git_status()
    #    check_requirements()
    settings['WORLD_SIZE'] = world_size
    settings['GLOBAL_RANK'] = global_rank

    weights = check_file(WEIGHTS)
    cfg = check_file(CFG)
    hyp = check_file(HYP)
    assert len(cfg) or len(weights), 'either cfg or weights must be specified'

    save_dir = increment_path(Path(os.path.join(PROJECT, NAME)), exist_ok=False)  # increment run
    settings['SAVE_DIR'] = save_dir

    # DDP mode
    total_batch_size = BATCH_SIZE
    device = select_device(DEVICE, batch_size=BATCH_SIZE)
    if int(LOCAL_RANK) != -1:
        assert torch.cuda.device_count() > int(LOCAL_RANK)
        torch.cuda.set_device(int(LOCAL_RANK))
        device = torch.device('cuda', int(LOCAL_RANK))
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert BATCH_SIZE % world_size == 0, '--batch-size must be multiple of CUDA device count'
        BATCH_SIZE = total_batch_size // world_size
        print("{} {} {} {}".format(LOCAL_RANK, BATCH_SIZE, world_size, global_rank))
    settings['TOTAL_BATCH_SIZE'] = total_batch_size
    settings['BATCH_SIZE'] = BATCH_SIZE
    settings['DEVICE'] = device

    # Hyperparameters
    with open(HYP) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    settings['HYP'] = hyp
    # Train
    logger.info(settings)
    if not EVOLVE:
        tb_writer = None  # init loggers
        if global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {PROJECT}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(save_dir)  # Tensorboard
        train(settings)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),   # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
                'paste_in': (1, 0.0, 1.0)}    # segment copy-paste (probability)
        
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
                
        assert int(LOCAL_RANK) == -1, 'DDP mode not implemented for --evolve'
        NO_TEST, NO_SAVE = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(os.path.join(save_dir, 'hyp_evolved.yaml'))  # save best result here
        if BUCKET:
            os.system('gsutil cp gs://%s/evolve.txt .' % BUCKET)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            settings['BUCKET'] = BUCKET
            settings['NO_SAVE'] = NO_SAVE
            settings['NO_TEST'] = NO_TEST
            settings['HYP'] = HYP
            # Train mutation
            results = train(settings)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, BUCKET)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
