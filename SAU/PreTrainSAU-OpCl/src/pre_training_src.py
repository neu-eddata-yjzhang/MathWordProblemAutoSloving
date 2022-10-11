from cProfile import label
import time
import torch.optim
import json
import torch
import random
import copy
import re
from src.expressions_transfer import *
from src.train_and_evaluate import *
from src.pre_data import *
from copy import deepcopy
import torch.nn.functional as f
import torch.nn as nn
from transformers import BertModel, BertForMaskedLM, BertTokenizer

def prepare_data_pretraining_all(pairs_trained, pairs_tested, pairs_tested_ape, trim_min_count, generate_nums, copy_nums, tree=False):
    id2data = {}
    from tqdm import tqdm
    train_pairs = []
    test_pairs = []
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-macbert-base")
    print("Indexing words...")
    one, two = 0, 0
    for pair in pairs_trained:
        try:
            inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)
            ans_nums = len(pair[4])
            eq = ''.join(pair[1])
            que_id = pair[5]
            Nnums = 0
            for i in eq:
                if i == 'N':
                    Nnums += 1
            train_pairs.append((inputs, ans_nums, que_id, Nnums, eq))
            if ans_nums == 1:
                one += 1
            else:
                two += 1
        except:
            continue
    print('Number of training data %d' % (len(train_pairs)))
    print('There are ',one,' questions with one var in train data')
    print('There are ',two,' questions with two var in train data')
    one, two = 0, 0
    for pair in pairs_tested:
        try:
            inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)
            ans_nums = len(pair[4])
            eq = ''.join(pair[1])
            que_id = pair[5]
            Nnums = 0
            for i in eq:
                if i =='N':
                    Nnums += 1
            test_pairs.append((inputs, ans_nums, que_id, Nnums, eq))
            if ans_nums == 1:
                one += 1
            else:
                two += 1
        except:
            continue    
    print('Number of testing data %d' % (len(test_pairs)))
    print('There are ',one,' questions with one var in test data')
    print('There are ',two,' questions with two var in test data')
    return train_pairs, test_pairs


def prepare_train_batch_all(pairs_to_batch, batch_size, epoch):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    input_batches = []
    input_batch = []
    pos_input_batches = []
    pos_input_batch = []
    neg_input_batches = []
    neg_input_batch = []
    OpNum_batches = []
    OpNum_batch = []
    total = 0
    no_pos = 0
    no_neg = 0
    for batch in pairs:
        inputs, ans_nums, que_id, Nnums, eq = batch[0], batch[1], batch[2], batch[3], batch[4]
        op_nums = [0,0,0,0,0] # +-*/^
        for i in eq:
            if i == '+':
                op_nums[0] += 1
            elif i == '-':
                op_nums[1] += 1
            elif i == '*':
                op_nums[2] += 1
            elif i == '/':
                op_nums[3] += 1
            elif i == '^':
                op_nums[4] += 1
        enough = 1
        pos, neg = True, True
        for i in pairs:
            if que_id == i[2]:
                continue
            if (eq in i[4] or i[4] in eq) and ans_nums == i[1] and pos:
                pos_temp = i[0]
                pos = False
                enough += 1
            elif ans_nums != i[1] and neg:
                if ans_nums == 1:
                    if eq not in i[4] :
                        neg = False
                        neg_temp = i[0]
                        enough += 1
                else:
                    if i[4] not in eq :
                        neg = False
                        neg_temp = i[0]
                        enough += 1
            if enough == 3:
                total += 1
                input_temp = inputs
                input_batch.append(input_temp)
                pos_input_batch.append(pos_temp)
                neg_input_batch.append(neg_temp)
                OpNum_batch.append(op_nums)
                break
        if pos:
            no_pos += 1
        if neg:
            no_neg += 1
        if len(input_batch) == batch_size:
            input_batches.append(input_batch)
            pos_input_batches.append(pos_input_batch)
            neg_input_batches.append(neg_input_batch)
            OpNum_batches.append(OpNum_batch)
            input_batch = []
            pos_input_batch = []
            neg_input_batch = []
            OpNum_batch = []
    if epoch == 0:
        print('After selecting, there are ',total,' datas are remained, total data are ',len(pairs))
        print(no_pos,' datas have not positive data')
        print(no_neg,' datas have not negative data')
    return input_batches, pos_input_batches, neg_input_batches, OpNum_batches


class PreTrainBert_all_Mac(nn.Module):
    def __init__(self, hidden_size, batch_size):
        super(PreTrainBert_all_Mac, self).__init__()

        # self.bert_model = BertForMaskedLM.from_pretrained("hfl/chinese-bert-wwm-ext")
        # self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        print('Start to load model')
        self.bert_model = BertModel.from_pretrained("hfl/chinese-macbert-base")
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.5)
        self.dist_loss = torch.nn.MSELoss()
        self.fc_op_num = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=5))                                           

    def forward(self, bert_input, op_num, device, check=False):
        length_max = max([i['input_ids'].squeeze().size(0) for i in bert_input])
        input_ids = []
        attention_mask = []
        for i in bert_input:
            input_id = i['input_ids'].squeeze()
            mask = i['attention_mask'].squeeze()
            zeros = torch.zeros(length_max - input_id.size(0))
            padded = torch.cat([input_id.long(), zeros.long()])
            input_ids.append(padded)
            padded = torch.cat([mask.long(), zeros.long()])
            attention_mask.append(padded)
        input_ids = torch.stack(input_ids,dim = 0).long().to(device)#.cuda()
        attention_mask = torch.stack(attention_mask,dim = 0).long().to(device)#.cuda()
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)[0].transpose(0,1) 
        problem_output = bert_output.mean(0)
        op_loss = self.dist_loss(self.fc_op_num(problem_output), torch.FloatTensor(op_num).to(device))
        return problem_output, op_loss

