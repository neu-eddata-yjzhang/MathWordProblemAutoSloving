import os
from train_and_evaluate import *
from models import *
import time
import torch.optim
from load_data import *
from num_transfer import *
from expression_tree import *
from log_utils import *
import copy
import random
from itertools import chain

torch.cuda.set_device(0)
batch_size = 32
embedding_size = 128
hidden_size = 768
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
beam_search = True
fold_num = 5
random_seed = 1
# var_nums = ['x','y']
var_nums = []
dataset_name = "Math23K"
ckpt_dir = "Math23K"
data_path = "../dataset/math23k/Math_23K.json"

if dataset_name == "Math23K":
    full_mode = False
    if full_mode:
        var_nums = ['x']
    else:
        var_nums = []
    ckpt_dir = "Math23K_Subtree_SA1"
    data_path = "../dataset/math23k/Math_23K.json"
elif dataset_name == "hmwp":
    hidden_size = 384
    batch_size = 8
    ckpt_dir = "hmwp_Subtree_SA"
    var_nums = ['x','y']
    data_path = "../dataset/hmwp/questions.json"
elif dataset_name == "ALG514":
    batch_size = 32
    ckpt_dir = "ALG514_Subtree_SA"
    var_nums = ['x','y']
    data_path = "../dataset/alg514/questions.json"

save_dir = os.path.join("../models", ckpt_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

log_file = os.path.join(save_dir, 'log')
create_logs(log_file)

for fold_id in range(fold_num):
    if not os.path.exists(os.path.join(save_dir, 'fold-'+str(fold_id))):
        os.mkdir(os.path.join(save_dir, 'fold-'+str(fold_id)))

data = load_math23k_data("../dataset/math23k/Math_23K.json")
data = load_math23k_data(data_path)
pairs, generate_nums, copy_nums = transfer_math23k_num(data)

pairs = None
generate_nums = None
copy_nums = None
if dataset_name == "Math23K":
    data = load_math23k_data(data_path, full_mode=full_mode)
    pairs, generate_nums, copy_nums = transfer_math23k_num(data)
elif dataset_name == "hmwp":
    data = load_hmwp_data(data_path)
    pairs, generate_nums, copy_nums = transfer_hmwp_num(data)
elif dataset_name == "ALG514":
    data = load_alg514_data(data_path)
    pairs, generate_nums, copy_nums = transfer_alg514_num(data)

temp_pairs = []
for p in pairs:
    ept = ExpressionTree()
    # print(p[1])
    ept.build_tree_from_infix_expression(p[1])
    # print(ept.get_prefix_expression())
    if len(p) == 4:
        temp_pairs.append((p[0], ept.get_prefix_expression(), p[2], p[3]))
    else:
        temp_pairs.append((p[0], ept.get_prefix_expression(), p[2], p[3], p[4]))
pairs = temp_pairs

# fold_size = int(len(pairs) * 0.2)
# fold_pairs = []
# for split_fold in range(4):
#     fold_start = fold_size * split_fold
#     fold_end = fold_size * (split_fold + 1)
#     fold_pairs.append(pairs[fold_start:fold_end])
# fold_pairs.append(pairs[(fold_size * 4):])

best_acc_fold = []
best_val_acc_fold = []
all_acc_data = []

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

group_data = read_json("../dataset/math23k/Math_23K_processed.json")

def get_train_test_fold(ori_path,prefix,data,pairs,group):
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
    #test_id = test_id[:1500]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item,pair,g in zip(data, pairs, group):
        pair = list(pair)
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold

ori_path = '../dataset/math23k/'
prefix = '23k_processed.json'
for fold in range(1):
    # pairs_tested = []
    # pairs_trained = []
    # for fold_t in range(5):
    #     if fold_t == fold:
    #         pairs_tested += fold_pairs[fold_t]
    #     else:
    #         pairs_trained += fold_pairs[fold_t]

    train_fold, test_fold, valid_fold = get_train_test_fold(ori_path,prefix,data,pairs,group_data)
    #print(len(train_fold),len(test_fold)) 21162 1000
    pairs_tested = test_fold
    pairs_trained = train_fold
    input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                    copy_nums, tree=True)
    # print(output_lang.index2word)
    # exit()
    # Initialize models
    #####src
    # encoder = Seq2TreeEncoder(vocab_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
    #                      n_layers=n_layers)
    #####Bert
    teacher_encoder = TeacherEncoderAttention(input_size=output_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                        n_layers=n_layers)
    encoder = Bert_EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
    predict = Seq2TreePrediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(var_nums),
                                 vocab_size=len(generate_nums) + len(var_nums))
    generate = Seq2TreeNodeGeneration(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=embedding_size)
    merge = Seq2TreeSubTreeMerge(hidden_size=hidden_size, embedding_size=embedding_size)
    # the embedding layer is  only for generated number embeddings, operators, and paddings
    sementic_alignment = Seq2TreeSemanticAlignment(encoder_hidden_size=hidden_size, decoder_hidden_size=hidden_size, hidden_size=hidden_size)
    teacher_classifier = PreClassify(hidden_size=hidden_size)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-5, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sementic_alignment_optimizer = torch.optim.Adam(sementic_alignment.parameters(), lr=learning_rate, weight_decay=weight_decay)
    teacher_optimizer = torch.optim.Adam(chain(teacher_encoder.parameters(),teacher_classifier.parameters()), lr=learning_rate, weight_decay=weight_decay) 

    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)
    sementic_alignment_scheduler = torch.optim.lr_scheduler.StepLR(sementic_alignment_optimizer, step_size=20, gamma=0.5)
    teacher_scheduler = torch.optim.lr_scheduler.StepLR(teacher_optimizer, step_size=20, gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()
        sementic_alignment.cuda()
        teacher_encoder.cuda()
        teacher_classifier.cuda()

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    var_num_ids = []
    for var in var_nums:
        if var in output_lang.word2index.keys():
            var_num_ids.append(output_lang.word2index[var])

    best_val_acc = 0
    best_equ_acc = 0
    current_save_dir = os.path.join(save_dir, 'fold-'+str(fold))
    current_best_val_acc = (0,0,0)
    for epoch in range(n_epochs):
        if epoch < 25:
            threshold = 0.15
        elif epoch < 50:
            threshold = 0.15
        else:
            threshold = 0.15
        loss_total = 0
        op_list = [1,2,3,4,5]
        # encoder_scheduler.step()
        # predict_scheduler.step()
        # generate_scheduler.step()
        # merge_scheduler.step()
        # sementic_alignment_scheduler.step()
        loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
        num_stack_batches, num_pos_batches, num_size_batches,ans_batches,bert_input_batches = prepare_train_batch(train_pairs, batch_size)
        # print("fold:", fold + 1)
        # print("epoch:", epoch + 1)
        logs_content = "fold: {}".format(fold+1)
        add_log(log_file, logs_content)
        logs_content = "epoch: {}".format(epoch + 1)
        add_log(log_file, logs_content)
        start = time.time()
        fake_batches = copy.deepcopy(output_batches)
        for idx in range(len(input_lengths)):
            loss = train_tree_with_subtree_semantic_alignment(
                epoch,input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], bert_input_batches[idx],fake_batches[idx],teacher_encoder,teacher_classifier, teacher_optimizer,generate_num_ids, encoder, predict, generate, merge, sementic_alignment,
                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, sementic_alignment_optimizer, output_lang, num_pos_batches[idx],
                var_nums=var_num_ids)
            for pos in range(len(num_size_batches[idx])):
                if num_size_batches[idx][pos] == 1:
                    continue
                if output_batches[idx][pos][0] >= 7:
                    continue
                num_list = list(range(7,7+num_size_batches[idx][pos]))
                one_true_record = output_batches[idx][pos]
                one_fake_record = copy.deepcopy(one_true_record)
                while(one_true_record == one_fake_record):
                    for pos_word in range(output_lengths[idx][pos]):
                        p = random.random()
                        if p < threshold:
                            if one_true_record[pos_word] < 5:
                                one_fake_record[pos_word] = random.choice(op_list)
                            else:
                                one_fake_record[pos_word] = random.choice(num_list)
                #print(one_true_record,one_fake_record)
                fake_batches[idx][pos] = one_fake_record
            loss_teacher = teacher_pretrain(
                        input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                        num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, output_lang, 
                        num_pos_batches[idx],fake_batches[idx],teacher_encoder,teacher_classifier, teacher_optimizer, bert_input=bert_input_batches[idx],if_multi=True)
            loss_total += loss

        encoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        sementic_alignment_scheduler.step()

        # print("loss:", loss_total / len(input_lengths))
        # print("training time", time_since(time.time() - start))
        # print("--------------------------------")
        logs_content = "loss: {}".format(loss_total / len(input_lengths))
        add_log(log_file, logs_content)
        logs_content = "training time: {}".format(time_since(time.time() - start))
        add_log(log_file, logs_content)
        logs_content = "--------------------------------"
        add_log(log_file, logs_content)
        if epoch % 1 == 0 or epoch > n_epochs - 5:
        #if epoch == n_epochs - 1:
            value_ac = 0
            equation_ac = 0
            answer_ac = 0
            eval_total = 0
            start = time.time()
            for test_batch in test_pairs:
                if len(test_batch) == 8:
                    test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                         merge, output_lang, test_batch[5],test_batch[7] ,beam_size=beam_size, beam_search=beam_search,
                                         var_nums=var_num_ids)
                else:
                    test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                         merge, output_lang, test_batch[5], test_batch[8],beam_size=beam_size, beam_search=beam_search,
                                         var_nums=var_num_ids)
                # 计算结果的代码要修改
                # val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
                if len(test_batch) == 8:
                    val_ac, equ_ac, ans_ac, _, _ = compute_equations_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[7],
                                                                            ans_list=test_batch[6], tree=True, prefix=True)
                else:
                    val_ac, equ_ac, ans_ac, _, _ = compute_equations_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6],
                                                                            ans_list=[], tree=True, prefix=True)
                if val_ac:
                    value_ac += 1
                if ans_ac:
                    answer_ac += 1
                if equ_ac:
                    equation_ac += 1
                eval_total += 1
            # print(equation_ac, value_ac, eval_total)
            # print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
            # print("testing time", time_since(time.time() - start))
            # print("------------------------------------------------------")
            logs_content = "{}, {}, {}".format(equation_ac, value_ac, eval_total)
            add_log(log_file,logs_content)
            logs_content = "test_answer_acc: {} {}".format(float(equation_ac) / eval_total, float(value_ac) / eval_total)
            add_log(log_file,logs_content)
            logs_content = "testing time: {}".format(time_since(time.time() - start))
            add_log(log_file,logs_content)
            logs_content = "------------------------------------------------------"
            add_log(log_file,logs_content)
            all_acc_data.append((fold, epoch,equation_ac, value_ac, eval_total))

            torch.save(encoder.state_dict(), os.path.join(current_save_dir, "seq2tree_encoder"))
            torch.save(predict.state_dict(), os.path.join(current_save_dir, "seq2tree_predict"))
            torch.save(generate.state_dict(), os.path.join(current_save_dir, "seq2tree_generate"))
            torch.save(merge.state_dict(),  os.path.join(current_save_dir, "seq2tree_merge"))
            torch.save(sementic_alignment.state_dict(),  os.path.join(current_save_dir, "seq2tree_sementic_alignment"))
            if best_val_acc < value_ac:
                best_val_acc = value_ac
                current_best_val_acc = (equation_ac, value_ac, eval_total)
                torch.save(encoder.state_dict(), os.path.join(current_save_dir, "seq2tree_encoder_best_val_acc"))
                torch.save(predict.state_dict(), os.path.join(current_save_dir, "seq2tree_predict_best_val_acc"))
                torch.save(generate.state_dict(), os.path.join(current_save_dir, "seq2tree_generate_best_val_acc"))
                torch.save(merge.state_dict(),  os.path.join(current_save_dir, "seq2tree_merge_best_val_acc"))
                torch.save(sementic_alignment.state_dict(),  os.path.join(current_save_dir, "seq2tree_sementic_alignment_best_val_acc"))
            if best_equ_acc < equation_ac:
                best_equ_acc = equation_ac
                torch.save(encoder.state_dict(), os.path.join(current_save_dir, "seq2tree_encoder_best_equ_acc"))
                torch.save(predict.state_dict(), os.path.join(current_save_dir, "seq2tree_predict_best_equ_acc"))
                torch.save(generate.state_dict(), os.path.join(current_save_dir, "seq2tree_generate_best_equ_acc"))
                torch.save(merge.state_dict(),  os.path.join(current_save_dir, "seq2tree_merge_best_equ_acc"))
                torch.save(sementic_alignment.state_dict(),  os.path.join(current_save_dir, "seq2tree_sementic_alignment_best_equ_acc"))
            if epoch == n_epochs - 1:
                best_acc_fold.append((equation_ac, value_ac, eval_total))
                best_val_acc_fold.append(current_best_val_acc)

a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
# print(a / float(c), b / float(c))
logs_content = "{} {}".format(a / float(c), b / float(c))
add_log(log_file,logs_content)
# print("------------------------------------------------------")
logs_content = "------------------------------------------------------"
add_log(log_file,logs_content)

a, b, c = 0, 0, 0
for bl in range(len(best_val_acc_fold)):
    a += best_val_acc_fold[bl][0]
    b += best_val_acc_fold[bl][1]
    c += best_val_acc_fold[bl][2]
    print(best_val_acc_fold[bl])
# print(a / float(c), b / float(c))
logs_content = "{} {}".format(a / float(c), b / float(c))
add_log(log_file,logs_content)
# print("------------------------------------------------------")
logs_content = "------------------------------------------------------"
add_log(log_file,logs_content)

print(all_acc_data)
logs_content = "{}".format(all_acc_data)
add_log(log_file,logs_content)


