import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer

class GTS_Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(GTS_Encoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, batch_graph, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H

        return pade_outputs, problem_output


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask.bool(), -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)

class Encoder_rbt(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(Encoder_rbt, self).__init__()

        self.hidden_size = hidden_size
        #self.bert_rnn = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
        #self.bert_rnn = BertModel.from_pretrained("/ibex/scratch/lianz0a/bert2tree-master/MWPbert-retrained/checkpoint-10000")
        #self.bert_rnn = BertModel.from_pretrained("/ibex/scratch/lianz0a/bert2tree-master/models/self_epoch_173")
        #self.bert_rnn = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.bert_rnn = BertModel.from_pretrained("/ibex/scratch/lianz0a/bert2tree-master/models/rbt_all_epoch_40")
    def forward(self, input_seqs, input_lengths, bert_encoding, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        length_max = max([i['input_ids'].squeeze().size(0) for i in bert_encoding])
        input_ids = []
        attention_mask = []
        for i in bert_encoding:
            input_id = i['input_ids'].squeeze()
            mask = i['attention_mask'].squeeze()
            zeros = torch.zeros(length_max - input_id.size(0))
            padded = torch.cat([input_id.long(), zeros.long()])
            input_ids.append(padded)
            padded = torch.cat([mask.long(), zeros.long()])
            attention_mask.append(padded)
        input_ids = torch.stack(input_ids,dim = 0).long().cuda()
        attention_mask = torch.stack(attention_mask,dim = 0).long().cuda()
        bert_output = self.bert_rnn(input_ids, attention_mask=attention_mask)[0].transpose(0,1)        

        problem_output = bert_output.mean(0)
        #pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        return bert_output, problem_output
    def evaluate(self, input_seqs, input_lengths, bert_encoding):
        input_ids = bert_encoding['input_ids'].long().cuda()
        attention_mask = bert_encoding['attention_mask'].long().cuda()
        bert_output = self.bert_rnn(input_ids, attention_mask=attention_mask)[0].transpose(0,1) # S x B x E
        problem_output = bert_output.mean(0)
        
        return bert_output, problem_output #seq_len, batch_size, H(768)

class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.hidden_size = hidden_size
        self.bert_rnn = BertModel.from_pretrained("/home/yjzhang/Paper/Model/RoBerta")
        #self.bert_rnn = BertModel.from_pretrained("Your Path")
        self.num_gru=nn.GRU(self.hidden_size,self.hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.tokenizer = BertTokenizer.from_pretrained("/home/yjzhang/Paper/Model/RoBerta")
        #self.ll = torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.gru=nn.GRU(self.hidden_size,self.hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.att = nn.MultiheadAttention(self.hidden_size,8)

    def forward(self, input_seqs, input_lengths, bert_encoding, num_pos,nums,hidden=None):
        #print('num_pos[0]',num_pos[0]) #[25, 44, 52]
        #print('nums[0]',nums[0]) #['1900', '1600', '5%']
        # exit()
        # Note: we run this all at once (over multiple batches of multiple sequences)
        length_max = max([i['input_ids'].squeeze().size(0) for i in bert_encoding])
        input_ids = []
        attention_mask = []
        for i in bert_encoding:
            input_id = i['input_ids'].squeeze()
            mask = i['attention_mask'].squeeze()
            zeros = torch.zeros(length_max - input_id.size(0))
            padded = torch.cat([input_id.long(), zeros.long()])
            input_ids.append(padded)
            padded = torch.cat([mask.long(), zeros.long()])
            attention_mask.append(padded)
        input_ids = torch.stack(input_ids,dim = 0).long().cuda()
        attention_mask = torch.stack(attention_mask,dim = 0).long().cuda()
        bert_output = self.bert_rnn(input_ids, attention_mask=attention_mask)[0]#.transpose(0,1)   

        # nums2T start
        nums_hidden=[]     
        #print('len(nums)',len(nums))
        for b in range(len(nums)):
            nums_prob_hidden=[]
            for num_i in range(len(nums[b])):
                nums_char_list=nums[b][num_i]
                #print('nums_char_list',nums_char_list)
                nums_char_ids = self.tokenizer(nums_char_list,is_split_into_words=True,return_tensors="pt", add_special_tokens=False)
                #print('nums_char_ids',nums_char_ids)
                nums_char_embedded = self.bert_rnn(nums_char_ids['input_ids'].long().cuda())[0].transpose(0,1)  #num_len*1*E
                #print('nums_char_embedded',nums_char_embedded.size())
                nums_gru_out,nums_gru_hidden=self.num_gru(nums_char_embedded)#num_len*1*E,1*1*E
                #print('nums_gru_out',nums_gru_out.size())
                nums_gru_final=nums_gru_out[-1, :, :self.hidden_size] + nums_gru_out[0, :, self.hidden_size:]#1*E
                #print('nums_gru_final',nums_gru_final.size())
                nums_prob_hidden.append(nums_gru_final.squeeze(0))##E
                #print('nums_gru_final.squeeze(0)',nums_gru_final.squeeze(0).size()) #768
            nums_hidden.append(nums_prob_hidden)
            #exit()
        #print('bert_output',bert_output.size()) #torch.Size([32, 89, 768])
        #exit()
        for b in range(len(nums)):
            seq_len = bert_output[b].size(1)
            #print(seq_len)
            for num_i in range(seq_len):
                if num_i in num_pos[b]:
                    num_index=num_pos[b].index(num_i)
                    try:
                    #print(bert_output[b][num_i].size(),nums_hidden[b][num_index].size())
                        bert_output[b][num_i] += nums_hidden[b][num_index]
                        #bert_output[b][num_i] = self.ll(bert_output[b][num_i])
                        #bert_output[b][num_i] = nums_hidden[b][num_index]
                    except:
                        continue
                    #print('bert_output[b][num_i]',bert_output[b][num_i].size()) #768
                    #exit()
        # nums2T finish 

        bert_output = bert_output.transpose(0,1)
        #print(bert_output.size()) #torch.Size([62, 32, 768])
        #bert_output,bert_att = self.att(bert_output,bert_output,bert_output)
        gru_out,gru_hidden=self.num_gru(bert_output)
        #print(gru_out.size())
        #bert_output=gru_out[-1, :, :self.hidden_size] + nums_gru_out[0, :, self.hidden_size:]
        bert_output=gru_out[ :,:, :self.hidden_size] + gru_out[ :,:, self.hidden_size:]
        # print(bert_output.size()) #torch.Size([62, 32, 768])
        # exit()
        problem_output = bert_output.mean(0)
        #pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        # print('bert_output',bert_output.size()) #torch.Size([89, 32, 768]) #torch.Size([32, 768])
        # print('problem_output',problem_output.size())
        # exit()
        return bert_output, problem_output
    def evaluate(self, input_seqs, input_lengths, bert_encoding,num_pos,nums):
        input_ids = bert_encoding['input_ids'].long().cuda()
        attention_mask = bert_encoding['attention_mask'].long().cuda()
        bert_output = self.bert_rnn(input_ids, attention_mask=attention_mask)[0]#.transpose(0,1) # S x B x E
        #print('bert_output',bert_output.size())
         # nums2T start
        #print('bert_output',bert_output.size())   
        #print('len(bert_encoding)',len(bert_encoding)) 
        nums_prob_hidden=[]
        for num_i in range(len(nums)):
            nums_char_list=nums[num_i]
            #print('nums_char_list',nums_char_list)
            nums_char_ids = self.tokenizer(nums_char_list,is_split_into_words=True,return_tensors="pt", add_special_tokens=False)
            #print('nums_char_ids',nums_char_ids)
            nums_char_embedded = self.bert_rnn(nums_char_ids['input_ids'].long().cuda())[0].transpose(0,1)  #num_len*1*E
            #print('nums_char_embedded',nums_char_embedded.size())
            nums_gru_out,nums_gru_hidden=self.num_gru(nums_char_embedded)#num_len*1*E,1*1*E
            #print('nums_gru_out',nums_gru_out.size())
            nums_gru_final=nums_gru_out[-1, :, :self.hidden_size] + nums_gru_out[0, :, self.hidden_size:]#1*E
            #print('nums_gru_final',nums_gru_final.size())
            nums_prob_hidden.append(nums_gru_final.squeeze(0))##E
            #print('nums_gru_final.squeeze(0)',nums_gru_final.squeeze(0).size()) #768
        seq_len = seq_len = bert_output.size(1)
        #print('bert_output',bert_output.size())
        #print(seq_len)
        for num_i in range(seq_len):
            if num_i in num_pos:
                num_index=num_pos.index(num_i)
                try:
                    bert_output[0][num_i] += nums_prob_hidden[num_index]
                    #bert_output[0][num_i] = nums_prob_hidden[num_index]
                except:
                    continue
                #print('bert_output[b][num_i]',bert_output[0][num_i].size()) #768
                    #exit()
        # nums2T finish 
        bert_output = bert_output.transpose(0,1)
        #bert_output,bert_att = self.att(bert_output,bert_output,bert_output)
        gru_out,gru_hidden=self.num_gru(bert_output)
        bert_output=gru_out[ :,:, :self.hidden_size] + gru_out[ :,:, self.hidden_size:]
        problem_output = bert_output.mean(0)
        #print('bert_output',bert_output.size())
        #print('problem_output',problem_output.size())
        return bert_output, problem_output #seq_len, batch_size, H(768)

class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask) #seq_len, batch_size, 768
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree
