import torch
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
from utils import *
from model.common import *


def compute_debias_loss(logits_neg, a):
    _, top_ans_ind = torch.topk(F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(F.softmax(logits_neg,dim=-1), 1, top_ans_ind).sum(1)
    qice_loss = neg_top_k.mean()
    return qice_loss


def loss_calu(predict, target, config):
    loss_fn = CrossEntropyLoss()

    batch_attr, batch_obj, batch_target = target[1], target[2], target[3]
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()
    _, logits_att, logits_obj, logits_soft_prompt, other_info = predict

    loss = []
    loss_att = loss_fn(logits_att, batch_attr)
    loss_obj = loss_fn(logits_obj, batch_obj)
    loss_logit_sp = loss_fn(logits_soft_prompt, batch_target)

    loss_logit_sp = (1- config.att_obj_w) * loss_logit_sp
    loss_att = config.att_obj_w * loss_att
    loss_obj = config.att_obj_w * loss_obj

    loss.append(loss_att)
    loss.append(loss_obj)
    loss.append(loss_logit_sp)

    # debias loss
    if 'logits_att_img_obj_txt' in other_info:
        loss_att = compute_debias_loss(other_info['logits_obj_img_att_txt'], F.one_hot(batch_attr, logits_att.shape[1]).float())
        loss_obj = compute_debias_loss(other_info['logits_att_img_obj_txt'], F.one_hot(batch_obj, logits_obj.shape[1]).float())
        dis_loss = config.debias_loss_weight * (loss_att + loss_att)
        loss.append(dis_loss)

    # retrieval loss
    if 'info_retrieval_loss_obj' in other_info:
        
        # obj
        info = other_info['info_retrieval_loss_obj']
        features_retrieval = info['retrieved_features']
        targets_retrieval = info['targets']
        features_ori = info['ori_features'].unsqueeze(1).repeat(1,info['retrieved_features'].shape[1],1)
        features_retrieval = features_retrieval/features_retrieval.norm(dim=-1, keepdim=True)
        features_ori = features_ori/features_ori.norm(dim=-1, keepdim=True)
        targets_ori = batch_obj.unsqueeze(1).repeat(1,info['retrieved_features'].shape[1])

        features_ori = features_ori.view(-1,features_ori.shape[-1])
        features_retrieval = features_retrieval.view(-1,features_retrieval.shape[-1])
        targets_ori = targets_ori.view(-1)
        targets_retrieval = targets_retrieval.view(-1)
        loss_tri_obj = pairwise_loss(features_ori,features_retrieval,targets_ori,targets_retrieval)

        # att
        info = other_info['info_retrieval_loss_att']
        features_retrieval = info['retrieved_features']
        targets_retrieval = info['targets']
        features_ori = info['ori_features'].unsqueeze(1).repeat(1,info['retrieved_features'].shape[1],1)
        features_retrieval = features_retrieval/ features_retrieval.norm(dim=-1, keepdim=True)
        features_ori = features_ori/ features_ori.norm(dim=-1, keepdim=True)
        targets_ori = batch_attr.unsqueeze(1).repeat(1,info['retrieved_features'].shape[1])

        features_ori = features_ori.view(-1,features_ori.shape[-1])
        features_retrieval = features_retrieval.view(-1,features_retrieval.shape[-1])
        targets_ori = targets_ori.view(-1)
        targets_retrieval = targets_retrieval.view(-1)
        loss_tri_att = pairwise_loss(features_ori,features_retrieval,targets_ori,targets_retrieval)

        retrieval_loss = config.retrieval_loss_weight *  (loss_tri_att + loss_tri_obj)
        loss.append(retrieval_loss)

    loss = sum(loss)

    return loss