# packages
import os
import sys
import re
import numpy as np 
import json
import pickle
import copy
import pickle
import os.path as osp
import torch
import math
import random


# required infos 
root = './'
dataset =  'mit-states'
split_info = torch.load(osp.join(root,'data',dataset,'metadata_compositional-split-natural.t7'))
split_info_dict = {x['image']:x for x in split_info}

split_info_train = [x for x in split_info if x['set'] == 'train']
split_info_train_dict = {}
for i,v in enumerate(split_info_train):
    split_info_train_dict[v['image']] = v
    split_info_train_dict[v['image']]['idx'] = i

pair2idx = {}
obj2idx = {}
att2idx = {}

for k,v in split_info_train_dict.items():
    att =  v['attr']
    obj =  v['obj']
    pair = att + ' ' + obj
    _ = pair2idx.setdefault(pair,[])
    pair2idx[pair].append(v['idx'])
    _ = att2idx.setdefault(att,[])
    att2idx[att].append(v['idx'])
    _ = obj2idx.setdefault(obj,[])
    obj2idx[obj].append(v['idx'])

att2pair_train = {}
obj2pair_train = {}
for p in pair2idx.keys():
    att, obj = p.split(' ')
    _ = att2pair_train.setdefault(att,[])
    att2pair_train[att].append(p)
    _ = obj2pair_train.setdefault(obj,[])
    obj2pair_train[obj].append(p)

# allocation function 
def obtain_allocated_num_new(pair_nums,max_num,balance_ratio=1.5):
    count = 0
    sorted_id = sorted(range(len(pair_nums)), key=lambda k: pair_nums[k])
    allocated = np.zeros(len(pair_nums)).astype(int)
    for i,idx in enumerate(sorted_id):
        required_num = math.ceil(max_num/(len(pair_nums)-count))
        num_idx = pair_nums[idx] if pair_nums[idx] <= required_num else required_num
        if i > 0:
            max_ = math.ceil(balance_ratio*allocated[sorted_id[i-1]])
            num_idx = num_idx if num_idx <= max_ else max_
        allocated[idx] = num_idx
        max_num -= num_idx
        count += 1
        
    return allocated.tolist()

# build db 
num_each_class = 128
obj_indices = []
for obj in obj2pair_train:
    obj_pairs = obj2pair_train[obj]
    pair_nums = [ len(pair2idx[x]) for x in obj_pairs ]
    allocated_num = obtain_allocated_num_new(pair_nums,num_each_class)
    indices = []
    for i,p in enumerate(obj_pairs):
        indices += random.sample(pair2idx[p],allocated_num[i])
    obj_indices += indices
random.shuffle(obj_indices)

att_indices = []
for att in att2pair_train:
    att_pairs = att2pair_train[att]
    pair_nums = [ len(pair2idx[x]) for x in att_pairs ]
    allocated_num = obtain_allocated_num_new(pair_nums,num_each_class)
    indices = []
    for i,p in enumerate(att_pairs):
        indices += random.sample(pair2idx[p],allocated_num[i])
    att_indices += indices
random.shuffle(att_indices)

# save db
indices = {'obj_indices':obj_indices,'att_indices':att_indices}
torch.save(indices,osp.join(root,'data',database,dataset + '.t7'))