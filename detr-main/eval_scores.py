import math
import os
import sys
from typing import Iterable
import torchvision.transforms as T
import numpy as np
import torch
import matplotlib.pyplot as plt
import util.misc as utils
from util import box_ops
from models.matcher import build_matcher
from tqdm import tqdm

@torch.no_grad()
def eval_score(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args, data_dir):
    model.eval()
    criterion.eval()

    ma_conf_noiou = np.zeros((4,3))
    conf_ma = np.zeros((4,3))

    for s_idx, (samples, targets, records, *_) in enumerate(tqdm(data_loader)):
        targets_new = []
        # NEW TARGET IS LIST(DICTIONARY(TENSOR)))
        for i, target in enumerate(targets):
            boxes = target[:, :2]
            cxs = boxes.mean(dim=1)
            cys = torch.zeros(cxs.size(dim=0)).add(0.5)
            ws = boxes[:, 1] - boxes[:, 0]
            hs = torch.ones(cxs.size(dim=0))
            boxes = torch.column_stack((cxs, cys, ws, hs))
            labels = target[:, 2].long()
            dict_t = {'boxes': boxes, 'labels': labels}
            targets_new.append(dict_t)

        samples = samples[:, None, :, :]
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets_new]
        outputs = model(samples)

        matcher = build_matcher(args)
        indices = matcher(outputs, targets)
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        idx = batch_idx, src_idx
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        giou = torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        out_idx = indices[0][0]
        t_idx = indices[0][1]

        out_prob = outputs["pred_logits"].flatten(0, 1).argmax(-1)

        for i in range(len(giou)):
            label = targets[0]['labels'][t_idx[i]]
            pred_label = out_prob[out_idx[i]]
            if pred_label > 3:
                pred_label = 3
            if giou[i] >= 0.3:
                conf_ma[pred_label, label] += 1
            ma_conf_noiou[pred_label, label] += 1



    if data_dir =="D:/10channel":
        np.save('D:/predictions/' + args.backbone + '_conf_noiou.npy', ma_conf_noiou)
        np.save('D:/predictions/' + args.backbone + '_conf_matrix.npy', conf_ma)
    else:
        np.save('/scratch/s203877/' + args.backbone + '_conf_noiou.npy', ma_conf_noiou)
        np.save('/scratch/s203877/' + args.backbone + '_conf_matrix.npy', conf_ma)

    return 'fuck off'
