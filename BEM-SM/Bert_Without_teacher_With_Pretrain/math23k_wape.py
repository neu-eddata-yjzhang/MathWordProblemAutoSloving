# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *
import json
import torch

torch.cuda.set_device(0)
batch_size = 32
embedding_size = 128
hidden_size = 768
n_epochs = 100
learning_rate = 4e-5
weight_decay = 1e-5
beam_size = 5
n_layers = 2
ori_path = './data/'
prefix = '23k_processed.json'
ape_path = "data/ape_simple_train.json"
ape_id = "data/ape_simple_id.txt"
ape_test_id = "data/ape_simple_test_id.txt"
overlap_id = "data/overlap.txt"
test_overlap_id = "data/test_overlap.txt"

id_list = open(ape_id, 'r').read().split()
id_list = [str(i) for i in id_list]
test_id_list = open(ape_test_id, 'r').read().split()
test_id_list = [str(i) for i in test_id_list]
overlap_list = open(overlap_id, 'r').read().split()
overlap_list = [str(i) for i in overlap_list]
test_overlap_list = open(test_overlap_id, 'r').read().split()
test_overlap_list = [str(i) for i in test_overlap_list]


data = load_raw_data("data/Math_23K.json") + raw_data_new("data/ape_simple_train.json", id_list, overlap_list) + raw_data_new("data/ape_simple_test.json", test_id_list, test_overlap_list)
group_data = read_json("data/Math_23K_processed.json")


pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(ori_path,prefix,data,pairs,group_data,ape_path,ape_id, ape_test_id)


best_acc_fold = []

pairs_tested = test_fold
pairs_tested_ape = valid_fold
pairs_trained = train_fold

input_lang, output_lang, train_pairs, test_pairs, test_pairs_ape = prepare_data(pairs_trained, pairs_tested, pairs_tested_ape, 5, generate_nums,
                                                                copy_nums, tree=True)

encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
# the embedding layer is  only for generated number embeddings, operators, and paddings


encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=1e-2)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=25, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=25, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=25, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=25, gamma=0.5)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

for epoch in range(n_epochs):

    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, bert_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()
    for idx in range(len(input_lengths)):

        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx], bert_batches[idx])
        loss_total += loss
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    if epoch % 5 == 0 or epoch > n_epochs - 10:
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs:
            #print(test_batch)
            #batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                         merge, output_lang, test_batch[5], test_batch[7], beam_size=beam_size)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc_23k", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs_ape:
            #print(test_batch)
            #batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                         merge, output_lang, test_batch[5], test_batch[7], beam_size=beam_size)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc_ape", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
        torch.save(encoder.state_dict(), "models/encoder_new")
        torch.save(predict.state_dict(), "models/predict_new")
        torch.save(generate.state_dict(), "models/generate_new")
        torch.save(merge.state_dict(), "models/merge_new")

