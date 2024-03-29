import math
import sys
import os.path as osp
from typing import Iterable, List, Optional, Tuple
from tqdm import tqdm
import time
import datetime
from collections import Counter

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, SequenceSampler,DistributedBatchSampler,RandomSampler,BatchSampler




from losses import DistillationLoss

import numpy as np
import utils

from models.GatherLayer import GatherLayer
import paddle.nn.functional as F


def labels2idxs(labels: paddle.Tensor):
    #labels = paddle.cast(labels,paddle.int32)
    buff = [paddle.cast(labels[i] == labels,dtype=paddle.int32) for i in range(labels.shape[0])]
    targets = paddle.stack(buff)
    return targets


def train_one_epoch(model: paddle.nn.Layer, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: paddle.optimizer.Optimizer,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema  = None, mixup_fn = None,
                    args=None):
    set_training_mode = args.train_mode
    fp32 = args.fp32_resume
    pretrain_cvlp = args.pretrain_cvlp
    two_branch = args.two_branch

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    text_tokens = getattr(data_loader.dataset, 'text_tokens', None)
    sent_idxs = getattr(data_loader.dataset, 'end_idxs', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        #target dimension
        if pretrain_cvlp:
            idxs = [np.random.randint(sent_idxs[t]) for t in targets]
            tokens = paddle.stack([text_tokens[targets[i]][idxs[i]] for i in range(len(targets))])

            if dist.is_initialized():
                targets = paddle.concat(GatherLayer.apply(targets), 0)
            targets = labels2idxs(targets)
            targets = paddle.cast(targets,samples.dtype)

            if mixup_fn is not None:
                targets_o = targets
                if dist.is_initialized():
                    samples = paddle.concat(GatherLayer.apply(samples), 0)
                samples, targets = mixup_fn(samples, targets)
                if dist.is_initialized():
                    gpu_idx = utils.get_rank()
                    gpu_num = utils.get_world_size()
                    samples = paddle.reshape(samples,(gpu_num, -1, samples.shape[1], samples.shape[2], samples.shape[3]))[gpu_idx]
                    #samples = samples.view(gpu_num, -1, samples.shape[1], samples.shape[2], samples.shape[3])[gpu_idx]
            samples = (samples, tokens)
        elif mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        outputs = model(samples)
        with paddle.amp.auto_cast():
            
            if two_branch:
                loss0 = criterion(samples, outputs[0], targets)
                loss1 = criterion(samples, outputs[1], targets)
                loss = loss0 + loss1
                metric_logger.update(loss0=loss0)
                metric_logger.update(loss1=loss1)
                metric_logger.update(loss=loss)
            elif pretrain_cvlp:
                loss, distill_loss = criterion(samples, outputs, targets)
                metric_logger.update(distill_loss=distill_loss)
            else:
                loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        optimizer.clear_grad()

        paddle.device.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if pretrain_cvlp:
            if mixup_fn is not None:
                targets = targets_o
            img_acc1 = multi_label_acc1(output=outputs[0], target=targets)
            text_acc1 = multi_label_acc1(output=outputs[1], target=targets)
            batch_size = samples[0].shape[0] / utils.get_world_size()
            metric_logger.meters['img_acc1'].update(img_acc1.item(), n=batch_size)
            metric_logger.meters['text_acc1'].update(text_acc1.item(), n=batch_size)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.get_lr())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@paddle.no_grad()
def evaluate(data_loader, model, args=None, tokens=None):
    two_branch = args.two_branch

    criterion = paddle.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    texts = tokens

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images
        target = target

        # compute output
        with paddle.amp.auto_cast():
            output = model((images, texts))
            if two_branch:
                batch_size = images.shape[0]
                loss0 = criterion(output[0], target)
                loss1 = criterion(output[1], target)
                acc0_1, acc0_5 = utils.accuracy(output[0], target, topk=(1,5))
                acc1_1, acc1_5 = utils.accuracy(output[1], target, topk=(1,5))

                metric_logger.update(loss0=loss0.item())
                metric_logger.update(loss1=loss1.item())
                metric_logger.meters['acc0_1'].update(acc0_1.item(), n=batch_size)
                metric_logger.meters['acc0_5'].update(acc0_5.item(), n=batch_size)
                metric_logger.meters['acc1_1'].update(acc1_1.item(), n=batch_size)
                metric_logger.meters['acc1_5'].update(acc1_5.item(), n=batch_size)
            else:
                loss = criterion(output, target)
                acc1, acc5 = utils.accuracy(output, target, topk=(1,5))
                batch_size = images.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg*100 for k, meter in metric_logger.meters.items()}


def shot_acc(preds, labels, train_class_count, many_shot_thr=100, low_shot_thr=20):
    # _, preds = output.topk(1, 1, True, True)
    # preds = preds.squeeze(-1)

    # [min_shot, max_shot, correct, total, acc]
    shot_cnt_stats = {
        "many": [many_shot_thr - 1, max(train_class_count), 0, 0, 0.],
        "median": [low_shot_thr, many_shot_thr - 1, 0, 0, 0.],
        "low": [0, low_shot_thr, 0, 0, 0.],
        "10-shot": [0, 10, 0, 0, 0.],
        "5-shot": [0, 5, 0, 0, 0.],
    }
    for l in paddle.unique(labels):
        class_correct = paddle.sum((preds[labels == l] == labels[labels == l])).item()
        test_class_count = len(labels[labels == l])
        for stat_name in shot_cnt_stats:
            stat_info = shot_cnt_stats[stat_name]
            if train_class_count[l] > stat_info[0] and train_class_count[l] <= stat_info[1]:
                stat_info[2] += class_correct
                stat_info[3] += test_class_count
    for stat_name in shot_cnt_stats:
        shot_cnt_stats[stat_name][-1] = shot_cnt_stats[stat_name][2] / shot_cnt_stats[stat_name][3] * \
                                        100.0 if shot_cnt_stats[stat_name][3] != 0 else 0.
    return shot_cnt_stats


@paddle.no_grad()
def evaluate_LT(data_loader, model, args=None, tokens=None, labels=None, prefix='val'):
    two_branch = args.two_branch

    criterion = paddle.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    texts = tokens if tokens is not None else None

    training_labels = np.array(labels).astype(int)
    train_class_count = [len(training_labels[training_labels == l]) for l in range(args.nb_classes)]
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images
        target = paddle.reshape(target,(-1,1))

        inputs = (images, texts) if texts is not None else images
        # compute output
        output = model(inputs)
        with paddle.amp.auto_cast():
            if two_branch:
                batch_size = images.shape[0]
                loss0 = criterion(output[0], target)
                loss1 = criterion(output[1], target)
                loss = loss0 + loss1
                acc0_1, acc0_5 = utils.accuracy(output[0], target, topk=(1,5))
                acc1_1, acc1_5 = utils.accuracy(output[1], target, topk=(1,5))
                alpha = 0.7 if 'INAT' in args.data_set else 0.2
                buff_output = F.softmax(output[0],axis=1) * alpha + F.softmax(output[1],axis=1)*(1-alpha)
                acc1, acc5 = utils.accuracy(buff_output, target, topk=(1,5))
                #acc1, acc5 = accuracy(output[0].softmax(1) * alpha + output[1].softmax(1) * (1-alpha), target, topk=(1, 5))

                metric_logger.update(loss=loss.item())
                metric_logger.update(loss0=loss0.item())
                metric_logger.update(loss1=loss1.item())
                metric_logger.meters['acc0_1'].update(acc0_1.item(), n=batch_size)
                metric_logger.meters['acc0_5'].update(acc0_5.item(), n=batch_size)
                metric_logger.meters['acc1_1'].update(acc1_1.item(), n=batch_size)
                metric_logger.meters['acc1_5'].update(acc1_5.item(), n=batch_size)
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

                output_ = output[0] + output[1]
                _, preds = paddle.topk(output_,1,1)
                #_, preds = output_.topk(1, 1, True, True)
                preds = paddle.squeeze(preds,axis=-1)
                #preds = preds.squeeze(-1)
                target = paddle.squeeze(target,axis=-1)
                shot_cnt_stats = shot_acc(preds, target, train_class_count)
                for stat_name in shot_cnt_stats:
                    metric_logger.meters[stat_name].update(shot_cnt_stats[stat_name][-1],
                                                           n=shot_cnt_stats[stat_name][-2])
            else:
                loss = criterion(output, target)
                acc1, acc5 = paddle.metric.accuracy(output, target, k=1),paddle.metric.accuracy(output, target, k=1)
                batch_size = images.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

                _, preds = paddle.topk(output_,1,1)
                preds = paddle.squeeze(preds,axis=-1)
                shot_cnt_stats = shot_acc(preds, target, train_class_count)
                for stat_name in shot_cnt_stats:
                    metric_logger.meters[stat_name].update(shot_cnt_stats[stat_name][-1],
                                                           n=shot_cnt_stats[stat_name][-2])

    if two_branch:
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} '
              'Acc0@1 {top01.global_avg:.3f} Acc0@5 {top05.global_avg:.3f} '
              'Acc1@1 {top11.global_avg:.3f} Acc1@5 {top15.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5,
                      top01=metric_logger.acc0_1, top05=metric_logger.acc0_5,
                      top11=metric_logger.acc1_1, top15=metric_logger.acc1_5,
                      losses=metric_logger.loss))
    else:
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg*100 for k, meter in metric_logger.meters.items()}


@paddle.no_grad()
def calc_class_acc(data_loader, model, args=None, tokens=None, prefix='val'):
    '''calculate accuracy for each class separately'''
    criterion = paddle.nn.CrossEntropyLoss()
    two_branch = args.two_branch
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    texts = tokens if tokens is not None else None

    labels = data_loader.dataset.targets
    labels = np.array(labels).astype(int)
    cnt_per_class = [len(labels[labels == l]) for l in range(args.nb_classes)]
    true_per_class = [0] * args.nb_classes
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images
        target = target

        inputs = (images, texts) if texts is not None else images
        # compute output
        with paddle.amp.auto_cast():
            output = model(inputs)
            if two_branch:
                loss0 = criterion(output[0], target)
                loss1 = criterion(output[1], target)
                loss = loss0 + loss1
                alpha = 0.7 if 'INAT' in args.data_set else 0.2
                buff_output = F.softmax(output[0],axis=1) * alpha + F.softmax(output[1],axis=1)*(1-alpha)
                acc1, acc5 = utils.accuracy(buff_output, target, topk= (1,5))
                output = output[0] + output[1]
            else:
                loss = criterion(output, target)
                acc1, acc5 = utils.accuracy(output, target, topk= (1,5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        _, preds = paddle.topk(output,1,1)
        preds = paddle.squeeze(preds,axis=-1)
        acc = preds == target
        for l in paddle.unique(target):
            true_per_class[l] += paddle.sum(acc[target == l]).item()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return [true_per_class[i] / cnt_per_class[i] for i in range(args.nb_classes)]


def multi_label_acc1(output: paddle.Tensor, target: paddle.Tensor):
    # target is a matrix of [0,1] with the same shape as output
    # print("multi_label_acc1:", target.shape)
    assert output.shape == target.shape
    _, pred = paddle.topk(output,1,1)
    pred = paddle.squeeze(pred.T)
    #pred = pred.t().squeeze()
    return (target[paddle.arange(0, target.shape[0]), pred] == 1).sum(
        0) * 100. / target.shape[0]


@paddle.no_grad()
def evaluate_pretrain(data_loader: DataLoader, model, labels=None, args=None, load_cache=True, topk=(5, 1),
                      prefix='val'):
    # switch to evaluation mode
    start_time = time.time()
    model.eval()
    if args.distributed: model = model.module

    text_tokens = getattr(data_loader.dataset, 'text_tokens', None)
    assert text_tokens is not None and isinstance(text_tokens, List), \
        "text_tokens is None, This function only supports pretraining phase"
    text_tokens = paddle.concat(text_tokens)
    sent_idxs = getattr(data_loader.dataset, 'end_idxs', None)
    assert sent_idxs is not None and isinstance(sent_idxs, List)
    targets = paddle.to_tensor(data_loader.dataset.targets)
    text_targets = paddle.empty((sum(sent_idxs),), dtype=paddle.int64)  # [Nt,]
    left = 0
    for i in range(len(sent_idxs)):
        text_targets[left : left + sent_idxs[i]] = i
        left += sent_idxs[i]
    
    # step 1. obtain all embeddings of image and text
    image_embeddings, text_embeddings = None, None
    cache_dir = osp.dirname(args.resume)
    img_embed_path = osp.join(cache_dir, "%s_img_embed.npy" % prefix)
    txt_embed_path = osp.join(cache_dir, "txt_embed.npy")
    if args.resume:
        cache_dir = osp.dirname(args.resume)
        img_embed_path = osp.join(cache_dir, "%s_img_embed.npy" % prefix)
        txt_embed_path = osp.join(cache_dir, "txt_embed.npy")
    if load_cache and osp.exists(img_embed_path):
        print("using cached image embeddings")
        image_embeddings = paddle.to_tensor(np.load(img_embed_path))
    if load_cache and osp.exists(txt_embed_path):
        print("using cached text embeddings")
        text_embeddings = paddle.to_tensor(np.load(txt_embed_path))

    # image
    if image_embeddings is None:
        image_embeddings = []
        iter = tqdm(data_loader, desc="image embeddings") if load_cache else data_loader
        for images, target in iter:
            images = images
            # compute output
            with paddle.amp.auto_cast():
                image_features = model.encode_image(images)
            image_embeddings.append(image_features.detach())
        image_embeddings = paddle.concat(image_embeddings)
        if utils.is_main_process(): np.save(img_embed_path, image_embeddings.cpu().numpy())
    # print("image_embeddings.shape: ", image_embeddings.shape) # [Ni, 1024]

    # text
    if text_embeddings is None:
        text_embeddings = []
        tokens_loader_val = DataLoader(dataset= text_tokens,batch_sampler= BatchSampler(sampler = SequenceSampler(text_tokens),batch_size= int(8 * args.batch_size)),
            num_workers=args.num_workers,
        )
        iter = tqdm(tokens_loader_val, desc="text embeddings") if load_cache else tokens_loader_val
        print(tokens_loader_val)
        for batch_tokens in iter:
            batch_tokens = batch_tokens
            # compute output
            with paddle.amp.auto_cast():
                text_features = model.encode_text(batch_tokens)
            text_embeddings.append(text_features.detach())
        text_embeddings = paddle.concat(text_embeddings)
        if utils.is_main_process(): np.save(txt_embed_path, text_embeddings.cpu().numpy())
    # print("text_embeddings.shape: ", text_embeddings.shape) # [Nt, 1024]
    if args.ensemble:
        print("using ensemble")
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        n_text_embeddings = []
        left = 0
        for i in range(len(sent_idxs)):
            n_text_embeddings.append(paddle.mean(text_embeddings[left : left + sent_idxs[i], :], axis=0))
            left += sent_idxs[i]
        text_embeddings = paddle.stack(n_text_embeddings)
        text_targets = paddle.arange(len(sent_idxs))

    # step 2. compute cosine similarity for image and text
    text_embeddings /= paddle.norm(text_embeddings,axis=-1,keepdim=True)
    #text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    #image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    image_embeddings /= paddle.norm(image_embeddings,axis=-1,keepdim=True)
    # image
    def get_pred(embeddings_A, embeddings_B, topk=1, desc=''):
        embeddings_loader = DataLoader(
            embeddings_A, batch_sampler= BatchSampler(sampler = SequenceSampler(embeddings_A),batch_size= int(8 * args.batch_size)),
            num_workers=args.num_workers,
        )
        iter = tqdm(embeddings_loader, desc=desc) if load_cache else embeddings_loader
        preds = []
        for batch_embeddings in iter:
            batch_embeddings = batch_embeddings
            batch_logits = batch_embeddings @ embeddings_B.t()
            _, batch_preds = paddle.topk(batch_logits,topk,1)
            #_, batch_preds = batch_logits.topk(topk, dim=1, largest=True, sorted=True)  # [BN, topk]
            preds.append(batch_preds)
        preds = paddle.concat(preds)
        return preds
    
    pred_image = get_pred(image_embeddings, text_embeddings, 
                            topk=max(topk), desc="preds of image embeddings") # [Ni, topk]
    print("pred_image.shape:", pred_image.shape)
    # print("logits_per_image.shape", logits_per_image.shape)

    pred_label = text_targets[pred_image]  # [Ni, topk]
    image_acc1 = paddle.sum(pred_label[:, 0] == targets) * 100.0 / pred_image.shape[0]
    # shot acc
    img_shot_acc, knn_shot_acc = {}, {}
    if labels is not None:
        training_labels = np.array(labels).astype(int)
        train_class_count = [len(training_labels[training_labels == l]) for l in range(args.nb_classes)]
        img_shot_acc = shot_acc(pred_label[:, 0], targets, train_class_count=train_class_count)
        img_shot_acc = {k: v[-1] for k, v in img_shot_acc.items()}
    # knn
    vote_result = paddle.to_tensor([Counter(label.tolist()).most_common(1)[0][0] for label in pred_label])
    if labels is not None:
        knn_shot_acc = shot_acc(vote_result, targets, train_class_count=train_class_count)
        knn_shot_acc = {f"knn_{k}": v[-1] for k, v in knn_shot_acc.items()}
    knn_acc = paddle.sum(vote_result == targets) * 100.0 / pred_image.shape[0]
    pred_text = get_pred(text_embeddings, image_embeddings, topk=1, desc="preds of text embeddings")
    pred_text = paddle.squeeze(pred_text) # [Nt, ]
    print("pred_text.shape:", pred_text.shape)
    pred_text = targets[pred_text]
    text_acc1 = paddle.sum(pred_text == text_targets) * 100.0 / pred_text.shape[0]

    paddle.device.cuda.synchronize()
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print("* image_Acc@1: {:.3f}% text_Acc@1 {:.3f}% knn_Acc@5 {:.3f}% Total time: {}".format(
        image_acc1.item(), text_acc1.item(), knn_acc.item(), total_time))
    return {"image_acc1": image_acc1.item(), "text_acc1": text_acc1.item(),
            f"knn_{max(topk)}": knn_acc.item(), **img_shot_acc, **knn_shot_acc}


@paddle.no_grad()
def select_sent(data_loader: DataLoader, model, args=None, load_cache=True, topk=(5, 1), prefix='val'):
    # switch to evaluation mode
    start_time = time.time()
    model.eval()
    if args.distributed: model = model.module

    text_tokens = getattr(data_loader.dataset, 'text_tokens', None)
    assert text_tokens is not None and isinstance(text_tokens, List), \
        "text_tokens is None, This function only supports pretraining phase"
    text_tokens = paddle.concat(text_tokens)
    sent_idxs = getattr(data_loader.dataset, 'end_idxs', None)
    assert sent_idxs is not None and isinstance(sent_idxs, List)
    text_targets = paddle.empty((sum(sent_idxs),), dtype=paddle.int64)  # [Nt,]
    left = 0
    for i in range(len(sent_idxs)):
        text_targets[left : left + sent_idxs[i]] = i
        left += sent_idxs[i]

    # step 1. obtain all embeddings of image and text
    image_embeddings, text_embeddings, image_targets = None, None, None
    if args.resume:
        cache_dir = osp.dirname(args.resume)
        img_embed_path = osp.join(cache_dir, "%s_img_embed.npy" % prefix)
        img_target_path = osp.join(cache_dir, "%s_img_target.npy" % prefix)
        txt_embed_path = osp.join(cache_dir, "%s_txt_embed.npy" % prefix)
    if load_cache and osp.exists(img_embed_path):
        print("using cached image embeddings")
        image_embeddings = paddle.to_tensor(np.load(img_embed_path))
    if load_cache and osp.exists(img_target_path):
        print("using cached image targets")
        image_targets = paddle.to_tensor(np.load(img_target_path))
    if load_cache and osp.exists(txt_embed_path):
        print("using cached text embeddings")
        text_embeddings = paddle.to_tensor(np.load(txt_embed_path))
    # image
    if image_embeddings is None or image_targets is None:
        image_embeddings = []
        image_targets = []
        iter = tqdm(data_loader, desc="image embeddings") if load_cache else data_loader
        for images, target in iter:
            images = images
            image_targets.append(target)
            # compute output
            with paddle.amp.auto_cast():
                image_features = model.encode_image(images)
            image_embeddings.append(image_features.detach())
        image_embeddings = paddle.concat(image_embeddings)
        image_targets = paddle.concat(image_targets)
        if utils.is_main_process(): np.save(img_embed_path, image_embeddings.cpu().numpy())
        if utils.is_main_process(): np.save(img_target_path, image_targets.cpu().numpy())
    # print("image_embeddings.shape: ", image_embeddings.shape) # [Ni, 1024]

    # text
    if text_embeddings is None:
        text_embeddings = []
        tokens_loader_val = DataLoader(
            batch_sampler=SequenceSampler(text_tokens),
            num_workers=args.num_workers
        )
        iter = tqdm(tokens_loader_val, desc="text embeddings") if load_cache else tokens_loader_val
        for batch_tokens in iter:
            batch_tokens = batch_tokens
            # compute output
            with paddle.amp.auto_cast():
                text_features = model.encode_text(batch_tokens)
            text_embeddings.append(text_features.detach())
        text_embeddings = paddle.concat(text_embeddings)
        if utils.is_main_process(): np.save(txt_embed_path, text_embeddings.cpu().numpy())
    # print("text_embeddings.shape: ", text_embeddings.shape) # [Nt, 1024]

    # step 2. compute cosine similarity for image and text
    text_embeddings /= text_embeddings.norm(axis=-1, keepdim=True)
    image_embeddings /= image_embeddings.norm(axis=-1, keepdim=True)

    text_ces = []
    iter = tqdm(range(text_embeddings.shape[0]), desc="ce for text embeddings") if load_cache else range(text_embeddings.shape[0])
    for i in iter:
        text_embedding = text_embeddings[i]
        logit = text_embedding @ image_embeddings.t() * model.logit_scale.exp()
        label = image_targets == text_targets[i]
        label = label / label.sum()
        ce = paddle.sum(-label * F.log_softmax(logit, axis=-1), axis=-1)
        text_ces.append(ce)

    text_ces = paddle.to_tensor(text_ces)

    txt_ce_path = osp.join(cache_dir, "%s_txt_ce.npy" % prefix)
    if utils.is_main_process(): np.save(txt_ce_path, text_ces.cpu().numpy())
    exit(0)