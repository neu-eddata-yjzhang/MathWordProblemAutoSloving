# coding: utf-8
from src.expressions_transfer import *
from src.train_and_evaluate import *
# from src.models import *
import time
import torch.optim
import json
import torch
from src.pre_training_src import *
import numpy as np
from copy import deepcopy
from transformers import BertModel
import json
import re
from torch.nn.modules.module import Module

def remove_brackets(x):
    y = x
    if x[0] == "(" and x[-1] == ")":
        x = x[1:-1]
        flag = True
        count = 0
        for s in x:
            if s == ")":
                count -= 1
                if count < 0:
                    flag = False
                    break
            elif s == "(":
                count += 1
        if flag:
            return x
    return y


def load_hmwp_data(filename): # load the json data to list(dict()) for hmwp data
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    f.close()
    out_data = []
    for d in data:
        if "equation" not in d or "ans" not in d or d["ans"] == []:
            continue
        x = d['equation']
        if len(set(x) - set("0123456789.+-*/^()=xXyY; ")) != 0:
            continue
        count1 = 0
        count2 = 0
        for elem in x:
            if elem == '(':
                count1 += 1
            if elem == ')':
                count2 += 1
        if count1 != count2:
            continue

        eqs = x.split(';')
        new_eqs = []
        for eq in eqs:
            sub_eqs = eq.split('=')
            new_sub_eqs = []
            for s_eq in sub_eqs:
                new_sub_eqs.append(remove_brackets(s_eq.strip()))
            new_eqs.append(new_sub_eqs[0] + ' = ' + new_sub_eqs[1])
        if len(new_eqs) == 1:
            d['equation'] = new_eqs[0]
        else:
            d['equation'] = ' ; '.join(new_eqs)

        seg = d['original_text'].strip().split()
        new_seg = []
        for s in seg:
            if len(s) == 1 and s in ",.?!;":
                continue
            new_seg.append(s)
        d['original_text'] = ' '.join(new_seg)
        out_data.append(d)
    return out_data

def transfer_hmwp_num(data_list):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data_list:
        nums = []
        input_seq = []
        seg = d["original_text"].strip().split()
        equations = d["equation"]

        for s in seg:
            pos = re.search(pattern, s) # 搜索每个词的数字位置
            if pos and pos.start() == 0:
                nums.append(s[pos.start():pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True) # 从大到小排序

        # print(nums)
        # print(nums_fraction)
        float_nums = []
        for num in nums:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums.append(str(float(eval(num.strip()))))
            elif '%' in num:
                float_nums.append(str(float(eval(num[:-1].strip()) / 100)))
            elif len(num) > 1 and num[0] == '0':
                float_nums.append(str(float(eval(num[1:].strip()))))
            else:
                float_nums.append(str(float(eval(num.strip()))))

        float_nums_fraction = []
        for num in nums_fraction:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums_fraction.append(str(float(eval(num.strip()))))
            elif '%' in num:
                float_nums.append(str(float(eval(num[:-1].strip()) / 100)))
            else:
                float_nums_fraction.append(str(float(eval(num.strip()))))
        # print(float_nums)
        # print(float_nums_fraction)
        nums = float_nums
        nums_fraction = float_nums_fraction

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N")
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res

            pos_st = re.search("\d+\.\d+%?|\d+%?", st) # 带百分号的数字数
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N")
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        new_out_seq = []
        for seq in out_seq:
            if seq == ' ' or seq == '':
                continue
            if seq == ';':
                new_out_seq.append('SEP')
                continue
            new_out_seq.append(seq)
        out_seq = new_out_seq
        # print(equations)
        # print(' '.join(out_seq))
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        ans = d['ans']
        d_id = d['id']
        pairs.append((input_seq, out_seq, nums, num_pos, ans, d_id))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    max_num_list_len = copy_nums

    return pairs, temp_g, max_num_list_len

batch_size = 8
embedding_size = 128
hidden_size = 768
n_epochs = 150
learning_rate = 1e-5
weight_decay = 1e-5
beam_size = 5
n_layers = 2
min_cl_loss = 99




var_nums = ['x','y']
data_path = "data/questions.json"
data = load_hmwp_data(data_path)
pairs, generate_nums, copy_nums = transfer_hmwp_num(data)

#####(input_seq, out_seq, nums, num_pos, ans, d_id)

pairs_tested = pairs[:400]
pairs_trained = pairs[400:]



best_acc_fold = []

print('Prepare Data ')
train_pairs, test_pairs = prepare_data_pretraining_all(pairs_trained, pairs_tested, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True)
torch.cuda.set_device(7)
device = torch.device("cuda:7")
pre_bert = PreTrainBert_all_Mac(hidden_size, batch_size).to(device)
#model = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext").to(device)
#pre_bert = torch.nn.DataParallel(pre_bert,device_ids=[1,2,3,4,5,6,7])
print('Load Model Success!')
model_optimizer = torch.optim.AdamW(pre_bert.parameters(), lr=learning_rate, weight_decay=weight_decay)
#ffn_optimizer = torch.optim.Adam(ffn_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

#model_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=30, gamma=0.5)
model_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=50, gamma=0.5)

contra_loss_func = torch.nn.MarginRankingLoss(0.2)
print('Start to Train')
for epoch in range(n_epochs):
    loss_total = 0
    input_batches, pos_input_batches, neg_input_batches = prepare_train_batch_all(train_pairs, batch_size,epoch)
    print("epoch:", epoch + 1)
    start = time.time()
    #result_list = []
    total_loss = 0
    for idx in range(len(input_batches)):
        pre_bert.train()
        model_optimizer.zero_grad()
        src = pre_bert(input_batches[idx],device = device)
        pos = pre_bert(pos_input_batches[idx],device = device)
        neg = pre_bert(neg_input_batches[idx],device = device)
        pos_sim = torch.cosine_similarity(src, pos, dim=0)
        neg_sim = torch.cosine_similarity(src, neg, dim=0)
        target = [1 for _ in range(pos_sim.shape[0])]
        target = torch.LongTensor(target).cuda()
        loss = contra_loss_func(pos_sim, neg_sim, target)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pre_bert.parameters(), 5)
        model_optimizer.step()
    print('train loss:', total_loss/len(input_batches))
    model_scheduler.step()

    ###########test
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    total_loss = 0
    if epoch % 1 == 0:
        input_batches, pos_input_batches, neg_input_batches = prepare_train_batch_all(test_pairs, batch_size, epoch)
        with torch.no_grad():
            for idx in range(len(input_batches)):
                pre_bert.eval()
                src = pre_bert(input_batches[idx],device = device)
                pos = pre_bert(pos_input_batches[idx],device = device)
                neg = pre_bert(neg_input_batches[idx],device = device)
                pos_sim = torch.cosine_similarity(src, pos, dim=0)
                neg_sim = torch.cosine_similarity(src, neg, dim=0)
                target = [1 for _ in range(pos_sim.shape[0])]
                target = torch.LongTensor(target).cuda()
                loss = contra_loss_func(pos_sim, neg_sim, target)
                total_loss += loss.item()
        print('test loss:', total_loss/len(input_batches))
        print("--------------------------------")
        if  cl_total_loss/len(input_batches) < min_cl_loss:
                pre_bert.bert_model.save_pretrained('./Macmodels/MiniCL')