import os
import numpy as np
import torch
import tqdm
from torch.utils.data.dataloader import DataLoader
from model.rapr_model import RAPR
from parameters import parser, YML_PATH
from loss import loss_calu
import test as test
from dataset import CompositionDataset
from utils import *
from utils import *

def train_model(model, optimizer, config, train_dataset, val_dataset):
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    train_dataloader_bd = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=False
    )

    indices = torch.load(os.path.join(config.database_path,config.dataset + '.t7')) 

    model.train()
    best_metric = 0
    best_loss = 0

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()
    train_losses = []

    for i in range(config.epoch_start, config.epochs):

        model.eval()
        target_att_all = torch.zeros(len(train_dataloader_bd.dataset))
        target_obj_all = torch.zeros(len(train_dataloader_bd.dataset))
        batch_img_att_all = torch.zeros(len(train_dataloader_bd.dataset),768)
        batch_img_obj_all = torch.zeros(len(train_dataloader_bd.dataset),768)

        count = 0
        progress_bar_bd = tqdm.tqdm(total=len(train_dataloader_bd), desc='exp: ' + config.exp_id + ' | ' + "epoch % 3d" % (i + 1)  +  '| build datbase',ncols=150)
        for bid, batch in enumerate(train_dataloader_bd):
            batch_img = batch[0].cuda()
            batch_img_retrieval = model.forward_bd(batch_img)
            batch_attr, batch_obj, _ = batch[1], batch[2], batch[3]
            target_att_all[count:count+batch_img.shape[0]] = batch_attr
            target_obj_all[count:count+batch_img.shape[0]] = batch_obj
            batch_img_att_all[count:count+batch_img.shape[0],:] = batch_img_retrieval[1].cpu().data
            batch_img_obj_all[count:count+batch_img.shape[0],:] = batch_img_retrieval[2].cpu().data
            count += batch_img.shape[0]
            progress_bar_bd.update()

        progress_bar_bd.close()
        model.db = {}
        model.db['features_att'] = batch_img_att_all[indices['att_indices']]
        model.db['targets_att'] = target_att_all[indices['att_indices']]
        model.db['features_obj'] = batch_img_obj_all[indices['att_indices']]
        model.db['targets_obj'] = target_obj_all[indices['att_indices']]

        model.train()
        progress_bar = tqdm.tqdm(total=len(train_dataloader), desc='exp: ' + config.exp_id + ' | ' + "epoch % 3d" % (i + 1),ncols=150)
        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):
            batch_img = batch[0].cuda()
            batch_attr, batch_obj, batch_target = batch[1], batch[2], batch[3]
            predict = model(batch_img, train_pairs,train=True)
            loss = loss_calu(predict, batch, config)

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps
            
            # backward pass
            loss.backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            postfix = {"loss": np.mean(epoch_train_losses[-50:])}
            progress_bar.set_postfix(postfix)
            progress_bar.update()

        progress_bar.close()
        progress_bar.write(f"epoch {i +1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))

        if (i + 1) % config.save_every_n == 0:
            data = {'model':model.state_dict(),'database':model.db}
            torch.save(data, os.path.join(config.save_path, f"epoch_{i}.pt"))

        print("Evaluating val dataset:")
        loss_avg, val_result = evaluate(model, val_dataset)


        if config.best_model_metric == "best_loss":
            if loss_avg.cpu().float() < best_loss:
                best_loss = loss_avg.cpu().float()
                data = {'model':model.state_dict(),'database':model.db}
                torch.save(data, os.path.join(config.save_path, f"best.pt"))
        elif val_result[config.best_model_metric] > best_metric:
                best_metric = val_result[config.best_model_metric]
                data = {'model':model.state_dict(),'database':model.db}
                torch.save(data, os.path.join(config.save_path, f"best.pt"))
       

def evaluate(model, dataset):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
            model, dataset, config)
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    result = ""
    key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
    for key in test_stats:
        if key in key_set:
            result = result + key + "  " + str(round(test_stats[key], 4)) + " | "
    
    result = dataset.phase + " split: " + result

    print(result)
    model.train()

    return loss_avg, test_stats

if __name__ == "__main__":
    
    config = parser.parse_args()
    load_args(YML_PATH[config.dataset], config)
    config.save_path = 'saved_models/' + config.dataset + '/' + config.model_name + config.exp_id
    set_seed(config.seed)
    os.makedirs(config.save_path, exist_ok=True)

    with open(os.path.join(config.save_path, "config.yaml"),"w",encoding="utf-8") as f:
        yaml.dump(vars(config),f)

    dataset_path = config.dataset_path
    train_dataset = CompositionDataset(dataset_path,phase='train')
    val_dataset = CompositionDataset(dataset_path,phase='val')

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)
    model = RAPR(config, attributes=attributes, classes=classes, offset=offset).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    model.train_pairs = val_dataset.train_pairs
    model.attr2idx = val_dataset.attr2idx
    model.obj2idx = val_dataset.obj2idx
    model.idx2attr = {v:k for k,v in model.attr2idx.items()}
    model.idx2obj = {v:k for k,v in model.obj2idx.items()}

    train_model(model, optimizer, config, train_dataset, val_dataset)
    print("done!")
