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
    from tqdm import tqdm
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    test_pairs_ape = []
    tokenizer = BertTokenizer.from_pretrained("/home/yjzhang/Paper/Model/MacBert")
    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
    for pair in pairs_trained:
        op_nums = [0,0,0,0,0] # +-*/^
        for i in pair[1]:
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
        try:
            num_stack = []
            for idx in range(len(pair[0])):
                if pair[0][idx] == 'NUM':
                    pair[0][idx] = 'n'
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])
            #if num_stack != []:
            #    print(num_stack)
            #    print('!!!')
            #inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt",add_special_tokens=False)
            inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)

            num_pos = []
            for idx,i in enumerate(inputs['input_ids'].squeeze()):
                if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                    num_pos.append(idx)
            
            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            #print(len(output_cell))
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if output_lang.word2index["UNK"] in output_cell:
                continue
            if len(input_cell) > 100 or len(output_cell) > 20:
                continue
            if len(output_cell) <= 1:
                continue
            if max(pair[3]) >= inputs['input_ids'].squeeze().size(0):
                continue
            train_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                                pair[2], num_pos, num_stack, inputs, op_nums,pair[4]))
        except:
            continue
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    for pair in pairs_tested:
        op_nums = [0,0,0,0,0] # +-*/^
        for i in pair[1]:
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
        try:
            num_stack = []
            for idx in range(len(pair[0])):
                if pair[0][idx] == 'NUM':
                    pair[0][idx] = 'n'
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])
            #if num_stack != []:
            #    print(num_stack)
            #    print('!!!')
            #inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt",add_special_tokens=False)
            inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)

            num_pos = []
            for idx,i in enumerate(inputs['input_ids'].squeeze()):
                if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                    num_pos.append(idx)
            
            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            #print(len(output_cell))
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if output_lang.word2index["UNK"] in output_cell:
                continue
            if len(input_cell) > 100 or len(output_cell) > 20:
                continue
            if len(output_cell) <= 1:
                continue
            if max(pair[3]) >= inputs['input_ids'].squeeze().size(0):
                continue
            test_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                                pair[2], num_pos, num_stack, inputs, op_nums,pair[4]))
        except:
            continue    
    print('Number of testing data %d' % (len(test_pairs)))

    return input_lang, output_lang, train_pairs, test_pairs


def prepare_train_batch_all(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    bert_input_batches = []
    op_num_batches = []
    num_type_batches = []

    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
                    # train_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                    #             pair[2], num_pos, num_stack, inputs, op_nums,pair[4]))
        for _, i, _, j, _, _, _, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
            #label_length.append(len(label))
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        #label_length_max = max(label_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        bert_input_batch = []
        op_num_batch = []
        num_type_batch = []

        for i, li, j, lj, num, num_pos, num_stack, bert_input, op_nums,numtype in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            bert_input_batch.append(bert_input)
            op_num_batch.append(op_nums) #####
            num_type_batch.append(numtype)

        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        bert_input_batches.append(bert_input_batch)
        op_num_batches.append(op_num_batch)
        num_type_batches.append(num_type_batch)

    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, bert_input_batches, op_num_batches,num_type_batches


class PreTrainBert_all_Mac(nn.Module):
    def __init__(self, hidden_size, batch_size):
        super(PreTrainBert_all_Mac, self).__init__()

        # self.bert_model = BertForMaskedLM.from_pretrained("hfl/chinese-bert-wwm-ext")
        # self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        print('Start to load model')
        self.bert_model = BertModel.from_pretrained("/home/yjzhang/Paper/Model/MacBert")
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.5)
        #self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.dist_loss = torch.nn.MSELoss()
        self.fc_num_num = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))
        self.fc_op_num = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=5))                                                              

    def forward(self, bert_input, num_pos,op_num,numtype,device,check=False):
        length_max = max([i['input_ids'].squeeze().size(0) for i in bert_input])
        input_ids = []
        attention_mask = []
        result = []
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
        #print(bert_output.size(),problem_output.size()) #torch.Size([84, 64, 768]) torch.Size([64, 768])
        loss_2 = self.dist_loss(self.fc_op_num(problem_output), torch.FloatTensor(op_num).to(device))#.cuda())
        return loss_2


def transfer_num_pretrain_all(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    #count = 0
    for d in data:
        #count += 1
        #if count == 100:
        #    break
        nums = []
        input_seq = []
        id = d["id"]
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]
        ans = d['ans']

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
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
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        type_nums = [0,0] #小数和整数
        try:
            for n in nums:
                if '/' in n or '%' in n or '.' in n:
                    type_nums[0] += 1 
                else:
                    type_nums[1] += 1 
        except:
            continue

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
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
        out_seq = from_infix_to_prefix(out_seq)
        candi = []
        for token in out_seq:
            if 'N' in token and out_seq.count(token) == 1:
                candi.append(token)

        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos, type_nums,id))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums


def get_train_test_fold_all_pretrain(ori_path,prefix,data,pairs,group,ape_path,ape_id,ape_test_id):
    id_list = open(ape_id, 'r').read().split()
    id_list = [str(i) for i in id_list]
    test_id_list = open(ape_test_id, 'r').read().split()
    test_id_list = [str(i) for i in test_id_list]
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item,pair in zip(data, pairs):
        pair = list(pair)
        pair = tuple(pair)
        if pair[-1] in train_id:
            train_fold.append(pair)
        elif pair[-1] in test_id:
            test_fold.append(pair)
        else:
            train_fold.append(pair)
    return train_fold, test_fold, valid_fold


def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file


def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data_d["type"] = "23k"
            data.append(data_d)
            js = ""

    return data

