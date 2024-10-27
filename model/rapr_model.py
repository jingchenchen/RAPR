import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.common import *
import numpy as np
from utils import *
from model.backbone import CLIP, LoRA_Encoder


class RAPR(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()

        clip_model = CLIP(
                init_checkpoint = 'clip_models/ViT-L-14.pt',
                image_resolution=224,
                vision_patch_size=14,
                vision_width=1024,
                vision_layers=24,
                embed_dim=768,
                context_length = 8,
                vocab_size = 49408,
                transformer_width = 768,
                transformer_heads = 12,
                transformer_layers = 12
            )
        
        clip_model.visual = LoRA_Encoder(clip_model.visual, r=4, topk=config.LR_K)
        self.clip = clip_model.cuda()
        self.clip.text_projection.data = self.clip.text_projection.data.half()
        self.clip.visual.lora_vit.proj.data = self.clip.visual.lora_vit.proj.data.half()

        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.token_ids, self.soft_att_obj, ctx_vectors, ctx_vectors_att = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        self.dtype = torch.float16

        self.text_encoder = CustomTextEncoder(self.clip, self.dtype)
        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.soft_prompt = nn.Parameter(ctx_vectors).cuda()
        self.soft_att_obj1 = nn.Parameter(self.soft_att_obj.clone()) # (360,768)
        self.soft_prompt1 = nn.Parameter(ctx_vectors.clone()).cuda() # (3,768)
        self.soft_prompt1_att = nn.Parameter(ctx_vectors_att.clone()).cuda()  # (1,768)

        width_img = config.width_img
        if config.projection == 'mlp':
            self.proj_img = MLP(width_img)
            self.proj_img_att = MLP(width_img)
            self.proj_img_obj = MLP(width_img)
        elif config.projection == 'sa':
            self.proj_img = ResidualAttentionBlock(width_img, width_img//64, None)
            self.proj_img_obj = ResidualAttentionBlock(width_img, width_img//64, None)
            self.proj_img_att = ResidualAttentionBlock(width_img, width_img//64, None)

    def construct_soft_prompt(self):
        token_ids = clip.tokenize("a photo of x x",
                              context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                clip.tokenize(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        ) 
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda()) 

        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        ) 
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = "a photo of"
        n_ctx = len(ctx_init.split())
        prompt = clip.tokenize(ctx_init,
                            context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

        ctx_init = "object"
        prompt = clip.tokenize(ctx_init,
                            context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)
        ctx_vectors_att = embedding[0, 1 : 2, :]

        return token_ids, soft_att_obj, ctx_vectors, ctx_vectors_att

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip.token_embedding(
            class_token_ids.cuda()
        ).type(self.clip.dtype)
        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        eos_idx = int(self.token_ids[0].argmax())
        token_tensor[:, eos_idx - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_att_obj[
            obj_idx + self.offset 
        ].type(self.clip.dtype)

        token_tensor[
            :, 1 : len(self.soft_prompt) + 1, :
        ] = self.soft_prompt.type(self.clip.dtype)

        return token_tensor

    def construct_token_tensors_att_obj(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        soft_att_obj = self.attr_dropout(self.soft_att_obj1)
        eos_idx = int(self.token_ids[0].argmax())
        
        attr_set = torch.unique(attr_idx)
        obj_set = torch.unique(obj_idx)

        token_ids_att = self.token_ids.repeat(len(attr_set), 1)
        token_att_tensors = self.clip.token_embedding(
            token_ids_att.cuda()
        ).type(self.clip.dtype)

        token_att_tensors[:, eos_idx - 2, :] = soft_att_obj[:self.offset].type(self.clip.dtype) # attr 
        token_att_tensors[
            :, 1 : len(self.soft_prompt) + 1, :
        ] = self.soft_prompt1.type(self.clip.dtype)

        token_att_tensors[:, eos_idx - 1, :] = self.soft_prompt1_att.type(self.clip.dtype)

        token_ids_obj = self.token_ids.repeat(len(obj_set), 1)
        token_obj_tensors = self.clip.token_embedding(
            token_ids_obj.cuda()
        ).type(self.clip.dtype)

        token_obj_tensors[:, eos_idx - 1, :] = soft_att_obj[self.offset:].type(self.clip.dtype) # obj

        # adding the correct learnable context
        token_obj_tensors[
            :, 1 : len(self.soft_prompt) + 1, :
        ] = self.soft_prompt1.type(self.clip.dtype)

        return token_att_tensors, token_obj_tensors

    def decompose_token_tensors(self, token_tensors, idx):
        att_idx, obj_idx = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        token_att_tensors = torch.zeros(len(self.attributes),token_tensors.shape[1],token_tensors.shape[2]).cuda()
        token_obj_tensors = torch.zeros(len(self.classes),token_tensors.shape[1],token_tensors.shape[2]).cuda()
        for i in range(len(self.attributes)):
            token_att_tensors[i,:,:] = token_tensors[np.where(att_idx==i)[0],:,: ].mean(0)
        for i in range(len(self.classes)):
            token_obj_tensors[i,:,:] = token_tensors[np.where(obj_idx==i)[0],:,: ].mean(0)        
        return token_att_tensors, token_obj_tensors

    def visual(self, x: torch.Tensor):
        x = self.clip.visual.lora_vit.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.clip.visual.lora_vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.clip.visual.lora_vit.positional_embedding.to(x.dtype)
        x = self.clip.visual.lora_vit.ln_pre(x)
        x = x.permute(1, 0, 2)
        img_feature = self.clip.visual.lora_vit.transformer(x)
        x = img_feature.permute(1, 0, 2)
        x = self.clip.visual.lora_vit.ln_post(x[:, 0, :])
        if self.clip.visual.lora_vit.proj is not None:
            x = x @ self.clip.visual.lora_vit.proj

        return x, img_feature

    def ft_to_logit_img(self,img):
        img_feature = img.permute(1, 0, 2)
        img_feature = self.clip.visual.lora_vit.ln_post(img_feature[:, 0, :])
        if self.clip.visual.lora_vit.proj is not None:
            img_feature = img_feature @ self.clip.visual.lora_vit.proj
        return img_feature

    def ft_to_logit_txt(self,txt):
        txt_feature = txt.permute(1, 0, 2)
        txt_feature = self.text_encoder.ln_final(txt_feature)
        txt_tf = (
            txt_feature[
                torch.arange(txt_feature.shape[0]), self.token_ids.argmax(dim=-1)
            ]
            @ self.text_encoder.text_projection
        )
        return txt_tf

    def compose_logits(self, logits_att, logits_obj, idx):
        attr_idx, obj_idx = idx[:, 0], idx[:, 1]
        logits_att_full = logits_att[:,attr_idx]
        logits_obj_full = logits_obj[:,obj_idx]
        logits = logits_att_full * logits_obj_full
        return logits, logits_att_full, logits_obj_full

    def obtain_img_embs(self,img_ft,batch_img):
        img_ft_att_ = img_ft.clone()
        img_ft_obj_ = img_ft.clone()

        img_ft_img = self.proj_img(img_ft.type(torch.float))
        img_ft_img =  self.ft_to_logit_img(img_ft_img.type(self.clip.dtype))
        img_emb = self.config.res_w_vis * batch_img + (1 - self.config.res_w_vis) * img_ft_img

        img_ft_att = self.proj_img_att(img_ft_att_.type(torch.float))
        img_ft_att = self.ft_to_logit_img(img_ft_att.type(self.clip.dtype))
        img_emb_att = self.config.res_w_vis_att * batch_img + (1 - self.config.res_w_vis_att) * img_ft_att

        img_ft_obj = self.proj_img_obj(img_ft_obj_.type(torch.float))
        img_ft_obj =  self.ft_to_logit_img(img_ft_obj.type(self.clip.dtype))
        img_emb_obj = self.config.res_w_vis_obj * batch_img + (1 - self.config.res_w_vis_obj) * img_ft_obj

        return img_emb, img_emb_att, img_emb_obj

    def forward(self, batch_img, idx, train=False):

        scores = None
        other_info = {}

        token_tensors = self.construct_token_tensors(idx)
        text_features, _ = self.text_encoder(self.token_ids,token_tensors,enable_pos_emb=self.enable_pos_emb,)
        token_att_tensors, token_obj_tensors = self.construct_token_tensors_att_obj(idx)
        text_att_features, _ = self.text_encoder(self.token_ids,token_att_tensors,enable_pos_emb=self.enable_pos_emb,)
        text_obj_features, _ = self.text_encoder(self.token_ids,token_obj_tensors,enable_pos_emb=self.enable_pos_emb,)

        txt_emb = text_features
        txt_emb_att = text_att_features
        txt_emb_obj = text_obj_features
        
        # normlization
        txt_emb = txt_emb/txt_emb.norm(dim=-1, keepdim=True)
        txt_emb_obj = txt_emb_obj/txt_emb_obj.norm(dim=-1, keepdim=True)
        txt_emb_att = txt_emb_att/txt_emb_att.norm(dim=-1, keepdim=True)

        batch_img, img_ft = self.visual(batch_img.type(self.clip.dtype))
        img_emb, img_emb_att, img_emb_obj = self.obtain_img_embs(img_ft, batch_img)

        # retrieval 
        img_emb_att, _, _, info_retrieval_loss_att = self.perform_retrieval(img_emb_att, postfix='_att', retrieval_loss_topk = self.config.retrieval_loss_topk)
        info_retrieval_loss_att['ori_features'] = img_emb_att
        other_info['info_retrieval_loss_att'] = info_retrieval_loss_att
                
        img_emb_obj, _, _, info_retrieval_loss_obj = self.perform_retrieval(img_emb_obj, postfix='_obj', retrieval_loss_topk = self.config.retrieval_loss_topk)
        info_retrieval_loss_obj['ori_features'] = img_emb_obj
        other_info['info_retrieval_loss_obj'] = info_retrieval_loss_obj

        # normlization
        img_emb = img_emb/img_emb.norm(dim=-1, keepdim=True)
        img_emb_att = img_emb_att/img_emb_att.norm(dim=-1, keepdim=True)
        img_emb_obj = img_emb_obj/img_emb_obj.norm(dim=-1, keepdim=True)

        logits_soft_prompt = (self.clip.logit_scale.exp() * img_emb @ txt_emb.t())
        logits_att = (self.clip.logit_scale.exp() * img_emb_att @ txt_emb_att.t())
        logits_obj = (self.clip.logit_scale.exp() * img_emb_obj @ txt_emb_obj.t())

        _, logits_att_full, logits_obj_full = self.compose_logits(logits_att,logits_obj,idx)  
        scores_soft_prompt = logits_soft_prompt
        scores_obj_att = logits_att_full + logits_obj_full
        scores = (1 - self.config.att_obj_w) * scores_soft_prompt + self.config.att_obj_w * scores_obj_att

        # debias loss
        logits_att_img_obj_txt = (self.clip.logit_scale.exp() * img_emb_att @ txt_emb_obj.t())
        logits_obj_img_att_txt = (self.clip.logit_scale.exp() * img_emb_obj @ txt_emb_att.t())
        other_info['logits_att_img_obj_txt'] = logits_att_img_obj_txt
        other_info['logits_obj_img_att_txt'] = logits_obj_img_att_txt

        return (scores, logits_att, logits_obj, logits_soft_prompt, other_info)

    def perform_retrieval(self, img_emb, postfix='', retrieval_loss_topk=None):

        img_emb_re = img_emb / img_emb.norm(dim=-1, keepdim=True)

        topk_sim, topk_index, retrieval_image_features, topk_sim_loss, topk_index_loss, retrieval_image_features_loss = query_img(img_emb_re, self.db['features'+postfix], self.config.retrieval_topk, topk_loss = retrieval_loss_topk)

        retrieval_image_features = retrieval_image_features.cuda().detach()
        att = F.softmax(topk_sim*self.config.retrieval_temperature, dim=-1)
        aggragated_feature = torch.bmm(att[:, None, :], retrieval_image_features).squeeze(1)
        aggragated_feature = aggragated_feature.type(self.clip.dtype)
        img_emb = self.config.retrieval_weight * aggragated_feature + (1 - self.config.retrieval_weight) * img_emb

        info_retrieval_loss = {}
        retrieval_image_targets = self.db['targets' + postfix][topk_index_loss.cpu().data].cuda()
        info_retrieval_loss['retrieved_features'] = retrieval_image_features_loss.type(self.clip.dtype).detach()
        info_retrieval_loss['targets'] = retrieval_image_targets

        return img_emb, topk_sim, topk_index, info_retrieval_loss

    def forward_bd(self,batch_img):
        with torch.no_grad():
            batch_img, img_ft = self.visual(batch_img.type(self.clip.dtype))
            img_emb, img_emb_att, img_emb_obj = self.obtain_img_embs(img_ft,batch_img)

        return img_emb, img_emb_att, img_emb_obj
