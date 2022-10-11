import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from transformers import BertTokenizer

def src_to_mask(src,x,y):
    #src = src.cpu().numpy()
    batch_data_mask_tok = []
    for encode_sen_idxx in src:
        #print('encode_sen_idxx',type(encode_sen_idxx),encode_sen_idxx.size(),len(list(encode_sen_idxx)))
        #encode_sen_idx = list(encode_sen_idxx["input_ids"][0].numpy())
        encode_sen_idx = list(encode_sen_idxx.cpu().numpy())
        token = 1
        mask = [0] * len(encode_sen_idx)
        for num in range(len(encode_sen_idx)):
            mask[num] = token
            #x = int(tokenizer("．", is_split_into_words=True,return_tensors="pt", add_special_tokens=False)["input_ids"])
            #y = int(tokenizer("，", is_split_into_words=True,return_tensors="pt", add_special_tokens=False)["input_ids"])
            if (encode_sen_idx[num] == x or encode_sen_idx[num] == y) \
                    and num != len(encode_sen_idx) - 1:
                token += 1
            if encode_sen_idx[num]==0:mask[num] = 0
        for num in range(len(encode_sen_idx)):
            if mask[num] == token and token != 1:
                mask[num] = 1000
        batch_data_mask_tok.append(mask)
        #print('batch_data_mask_tok',len(batch_data_mask_tok),len(batch_data_mask_tok[0]))
    return np.array(batch_data_mask_tok)


def group_mask(batch,type="self",pad=0):
    length = batch.shape[1]
    lis = []
    if type=="self":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    if ele != 1000:copy[copy == 1000] = 0
                    copy[copy != ele] = 0
                    copy[copy == ele] = 1
                    #print("self copy",copy)
                '''
                if ele == 1000:
                    copy[copy != ele] = 1
                    copy[copy == ele] = 0
                '''
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    elif type=="between":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy==1000] = 0
                    copy[copy ==ele] = 0
                    copy[copy!= 0] = 1
                    '''
                    copy[copy != ele and copy != 1000] = 1
                    copy[copy == ele or copy == 1000] = 0
                    '''
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    elif type == "question":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy != 1000] = 0
                    copy[copy == 1000] = 1
                if ele==1000:
                	copy[copy==0] = -1
                	copy[copy==1] = 0
                	copy[copy==-1] = 1
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    else:return "error"
    return res

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             /math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.cuda() == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionwiseFeedForwardd(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForwardd, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        #print(type(d_ff),type(d_model),d_ff,d_model)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class GroupAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(GroupAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.tokenizer = BertTokenizer.from_pretrained("/home/yjzhang/Paper/Model/MacBert")
        self.x = int(self.tokenizer("．", is_split_into_words=True,return_tensors="pt", add_special_tokens=False)["input_ids"])
        self.y = int(self.tokenizer("，", is_split_into_words=True,return_tensors="pt", add_special_tokens=False)["input_ids"])
        #print(self.x,self.y)

    def get_mask(self,src,pad=0):
        mask = src_to_mask(src,self.x,self.y)
        #print('mask',len(mask),len(mask[0]))
        self.src_mask_self = torch.from_numpy(group_mask(mask,"self",pad).astype('uint8')).unsqueeze(1)
        self.src_mask_between = torch.from_numpy(group_mask(mask,"between",pad).astype('uint8')).unsqueeze(1)
        self.src_mask_question = torch.from_numpy(group_mask(mask, "question", pad).astype('uint8')).unsqueeze(1)
        self.src_mask_global = (src != pad).unsqueeze(-2).unsqueeze(1)
        self.src_mask_global = self.src_mask_global.expand(self.src_mask_self.shape).type(torch.uint8)
        self.final = torch.cat((self.src_mask_between.cuda(),self.src_mask_self.cuda(),self.src_mask_global.cuda(),self.src_mask_question.cuda()),1)
        return self.final.cuda()

    def forward(self, query, key, value, mask=None):
        #print("query",query,"\nkey",key,"\nvalue",value)
        "Implements Figure 2"

        if mask is not None and len(mask.shape)<4:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        else:
            mask = torch.cat((mask, mask), 1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # which is linears(query, key, value)


        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)


        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)