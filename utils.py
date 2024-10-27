import os
import torch
import numpy as np
import random
import os
import yaml

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)

def query(xd,xq,k):
    
    xd = xd.astype('float32')
    xq = xq.astype('float32')
    xd = torch.tensor(xd)
    xq = torch.tensor(xq)

    similarity = xq @ xd.T
    D,I = torch.sort(similarity, descending=True)
    D, I = D[:, : k], I[:, : k]

    return D, I

def query_img(query_features, keys_features, k, topk_loss=None):

    keys_features = keys_features.detach()
    keys_features_re = keys_features.clone().detach()
    keys_features_re = keys_features_re / keys_features_re.norm(dim=-1, keepdim=True)
    keys_features_re = keys_features_re.cpu().numpy()

    topk_sim_loss, topk_index_loss = query(keys_features_re, query_features.detach().cpu().numpy(), topk_loss)
    topk_sim, topk_index = topk_sim_loss[:, : k], topk_index_loss[:, : k]

    topk_sim = topk_sim.cuda()  
    topk_index = topk_index.cuda() 
    keys_features = keys_features.cuda()
    retrieval_features = keys_features[topk_index]  
    retrieval_features_loss = keys_features[topk_index_loss]

    return topk_sim, topk_index, retrieval_features, topk_sim_loss, topk_index_loss, retrieval_features_loss

def pairwise_loss(outputs1,outputs2,label1,label2):
    similarity = (label1.data.float() == label2.data.float()).float()
    dot_product = (outputs1 * outputs2).sum(1)

    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    exp_loss = torch.log(1+torch.exp(-torch.abs(dot_product))) + torch.max(dot_product, torch.FloatTensor([0.]).cuda())-similarity * dot_product

    S1 = torch.sum(mask_positive.float())
    S0 = torch.sum(mask_negative.float())
    S = S0+S1
    exp_loss[similarity.data > 0] = exp_loss[similarity.data > 0] * (S / S1)
    exp_loss[similarity.data <= 0] = exp_loss[similarity.data <= 0] * (S / S0)

    loss = torch.sum(exp_loss) / S

    return loss
